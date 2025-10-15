import logging
from typing import Dict, List, Optional

import torch
from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules.attention import CausalAttention
from rtp_llm.models_py.modules.embedding import Embedding
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear import Linear
from rtp_llm.models_py.modules.moe import FusedMoe
from rtp_llm.models_py.modules.moe.fused_moe_factory import FusedMoeFactory
from rtp_llm.models_py.modules.norm import RMSNorm
from rtp_llm.ops import KVCache, PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.batch_overlap.stage_executor import StateDict
from rtp_llm.models_py.batch_overlap.micro_batch_executor import MicroBatchExecutor

try:
    from libth_transformer.rtp_llm_ops import SelectTopkOp
except ImportError:
    logging.info("SelectTopkOp not available")


class Qwen3MoeLayer(nn.Module):
    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.config = config

        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.moe_inter_padding_size
        self.num_experts = config.expert_num
        self.top_k = config.moe_k
        self.gate = Linear(weights[W.moe_gate], None)
        self.select_topk_op = SelectTopkOp(config)
        self.fused_moe: FusedMoe = FusedMoeFactory.create_fused_moe(config, weights)
        self.w1 = weights.get(W.moe_w1, None)
        self.w2 = weights.get(W.moe_w2, None)
        assert self.w1 is not None and self.w2 is not None, "Weights w1 and w2 must be provided"
        self.num_local_experts = self.w1.shape[0]

        self.expert_map = self.build_expert_map()

    def build_expert_map(self):
        num_local_experts = self.num_local_experts
        global_num_experts = self.num_experts
        expert_map = torch.full((global_num_experts,), fill_value=-1, dtype=torch.int32)
        start_id = self.config.ep_rank * num_local_experts
        end_id = start_id + num_local_experts
        expert_map[start_id:end_id] = torch.tensor(list(range(num_local_experts)))
        return expert_map.to(device=torch.cuda.current_device(), dtype=torch.int32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, _ = hidden_states.shape
        router_logits = self.gate(hidden_states)

        router_logits_fp32 = router_logits.float()
        topk_weights = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        self.select_topk_op.forward(router_logits_fp32, topk_ids, topk_weights)

        return self.fused_moe(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation="SiGLU",
            expert_map=self.expert_map,
        )

    def op_gate(self, state: StateDict, hidden_states: torch.Tensor):
        router_logits = self.gate(hidden_states).float()
        return {"router_logits": router_logits, "hidden_states": hidden_states}

    def op_select_topk_experts(self, state: StateDict, router_logits: torch.Tensor, hidden_states: torch.Tensor):
        num_tokens = hidden_states.shape[0]
        topk_weights = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        topk_ids = torch.zeros(
            (num_tokens, self.top_k),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        self.select_topk_op.forward(router_logits, topk_ids, topk_weights)
        state.topk_ids = topk_ids
        state.topk_weights = topk_weights
        state.quant_config = self.fused_moe.fused_experts.quant_config
        return {"hidden_states": hidden_states}


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = CausalAttention(config, weights)
        self.mlp = Qwen3MoeLayer(config, weights)

        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(weights[W.post_ln_gamma], eps=config.layernorm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states=hidden_states, fmha_impl=fmha_impl, kv_cache=kv_cache)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def op_store_residual(self, state: StateDict, hidden_states: torch.Tensor):
        state.residual = hidden_states
        return {"hidden_states": hidden_states}

    def op_add_residual(self, state: StateDict, hidden_states: torch.Tensor):
        residual = state.pop("residual")
        hidden_states = residual + hidden_states
        return {"hidden_states": hidden_states}

    def op_input_layernorm(self, state: StateDict, hidden_states: torch.Tensor):
        hidden_states = self.input_layernorm(hidden_states)
        return {"hidden_states": hidden_states}

    def op_post_attention_layernorm(self, state: StateDict, hidden_states: torch.Tensor):
        hidden_states = self.post_attention_layernorm(hidden_states)
        return {"hidden_states": hidden_states}


class Qwen3MoeModel(GptModelBase):
    def __init__(self, config: GptInitModelParameters, weights: ModelWeights):
        super().__init__(config, weights)
        self.embed_tokens = Embedding(config, weights.get_global_weight(W.embedding))
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, weights.weights[idx], idx) for idx in range(self.layer_num)]
        )
        if config.gpt_init_params.device_resource_config.enable_layer_micro_batch == 0:
            self.norm = RMSNorm(weights.get_global_weight(W.final_ln_gamma), eps=config.layernorm_eps)
        else:
            self.micro_batch_executor = MicroBatchExecutor(config, self, self.layers)

    def forward(self, inputs: PyModelInputs) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs: PyAttentionInputs = inputs.attention_inputs
        fmha_impl = self.get_fmha_impl(attention_inputs)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states = decoder_layer(
                hidden_states,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states = self.norm(hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)

    def op_prepare_forward(self, state: StateDict, stage_input: PyModelInputs):
        state.update(
            {
                "fmha_impl": self.get_fmha_impl(stage_input.attention_inputs),
                "kv_cache": self.kv_cache if self.kv_cache else None,
                "layer_idx": 0,
            }
        )
        return {"input_ids": stage_input.input_ids}

    def op_embed_tokens(self, state: StateDict, input_ids: torch.Tensor):
        return {"hidden_states": self.embed_tokens(input_ids)}

    def op_finalize_forward(self, state: StateDict, hidden_states: torch.Tensor):
        state.pop("fmha_impl")
        state.pop("kv_cache")
        state.pop("layer_idx")
        stage_output = PyModelOutputs(hidden_states)
        return {"stage_output": stage_output}

    def forward_micro_batch(self, inputs: List[PyModelInputs]) -> List[PyModelOutputs]:
        return self.micro_batch_executor.execute(inputs)