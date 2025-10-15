from typing import Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.distribute.collective import Group, all_reduce
from rtp_llm.models_py.modules import FusedQKRMSNorm
from rtp_llm.models_py.modules.fmha import FMHAImplBase
from rtp_llm.models_py.modules.linear_factory import LinearFactory
from rtp_llm.ops import KVCache
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.batch_overlap.stage_executor import StateDict


class CausalAttention(nn.Module):

    def __init__(self, config: GptInitModelParameters, weights: Dict[str, torch.Tensor]):
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.head_num
        self.head_num = config.head_num
        self.num_key_value_groups = config.head_num // config.head_num_kv
        self.q_size = config.head_num * self.head_dim

        # Create linear layers using LinearFactory
        self.qkv_proj = LinearFactory.create_linear_from_weights(
            weights, W.attn_qkv_w, W.attn_qkv_s, W.attn_qkv_b, config
        )
        self.o_proj = LinearFactory.create_linear_from_weights(weights, W.attn_o_w, W.attn_o_s, W.attn_o_b, config)
        self.qk_fuse_norm = None
        if W.q_ln_gamma in weights and W.k_ln_gamma in weights:
            self.qk_fuse_norm = FusedQKRMSNorm(
                weights[W.q_ln_gamma],
                weights[W.k_ln_gamma],
                config.head_num // config.tp_size,
                config.head_num_kv // config.tp_size,
                config.size_per_head,
                config.layernorm_eps,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[KVCache],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        qkv = self.qkv_proj(hidden_states)
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        attn_output = fmha_impl.forward(qkv, kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        output = self.o_proj(attn_output)
        if self.config.tp_size > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def op_qkv_proj(self, state: StateDict, hidden_states: torch.Tensor):
        state.input_shape = hidden_states.shape[:-1]
        return {"qkv": self.qkv_proj(hidden_states)}

    def op_qk_fuse_norm(self, state: StateDict, qkv: torch.Tensor):
        if self.qk_fuse_norm is not None:
            qkv = self.qk_fuse_norm(qkv)
        return {"qkv": qkv}

    def op_fmha_impl(self, state: StateDict, qkv: torch.Tensor):
        input_shape = state.pop("input_shape")
        current_layer_kv_cache = state.kv_cache.get_layer_cache(state.layer_idx) if state.kv_cache else None
        state.layer_idx += 1
        attn_output = state.fmha_impl.forward(qkv, current_layer_kv_cache)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return {"attn_output": attn_output}

    def op_o_proj(self, state: StateDict, attn_output: torch.Tensor):
        hidden_states = self.o_proj(attn_output)
        return {"hidden_states": hidden_states}

    def op_all_reduce(self, state: StateDict, hidden_states: torch.Tensor):
        if self.config.tp_size > 1:
            hidden_states = all_reduce(hidden_states, group=Group.TP)
        return {"hidden_states": hidden_states}
