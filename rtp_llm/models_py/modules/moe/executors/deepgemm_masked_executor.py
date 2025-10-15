from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.kernels.activation import (
    silu_mul_bf16_deep_gemm_masked,
    silu_mul_fp8_quant_deep_gemm_masked,
)
from rtp_llm.models_py.kernels.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
    m_grouped_bf16_gemm_nt_masked,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
    TopKWeightAndReduce,
)
from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
from rtp_llm.utils.model_weight import W
from rtp_llm.models_py.batch_overlap.stage_executor import StateDict


class DeepGemmMaskedExecutor(FusedMoeExpertExecutor):

    # The Deep Gemm kernels only support block size of 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
        quant_config: FusedMoEQuantConfig,
    ):
        """Initialize the DeepGemmMaskedExecutor.
        Args:
            config: Model configuration.
            weights: Dictionary containing model weights.
            quant_config: Quantization configuration.
        """
        super().__init__(quant_config=quant_config)
        self._config = config
        self._weights = weights
        self._num_experts = config.expert_num
        # init weights
        self._w1 = self._weights.get(W.moe_w1, None)
        self._w2 = self._weights.get(W.moe_w2, None)
        self._w1_scale = self._weights.get(W.moe_s1, None)
        self._w2_scale = self._weights.get(W.moe_s2, None)
        assert self._w1 is not None and self._w2 is not None
        # check fp8 block quantization
        self._use_fp8 = True
        if self.quant_config.is_quantized:
            if self.quant_config.quant_dtype == torch.float8_e4m3fn and self.quant_config.is_block_quantized:
                if self.quant_config.block_shape != self.DEEPGEMM_BLOCK_SHAPE:
                    raise NotImplementedError(
                        "DeepGemmMaskedExecutor only supports fp8 block quantization with block shape 128x128"
                    )
                self._use_fp8 = True
                assert self._w1_scale is not None and self._w2_scale is not None
            else:
                raise NotImplementedError("DeepGemmMaskedExecutor only supports fp8 block quantization or bf16")
        else:
            self._use_fp8 = False

    @property
    def local_num_experts(self) -> int:
        assert self._w1 is not None
        return self._w1.size(0)

    def finalize_weight_and_reduce_impl(self) -> TopKWeightAndReduce:
        # Let PrepareAndFinalize::finalize() decide the impl.
        return TopKWeightAndReduceDelegate()

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        global_num_experts: int,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:

        assert self._w1 is not None and self._w2 is not None
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None

        expert_x = payload.expert_x
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        assert expert_num_tokens is not None

        assert expert_x.ndim == 3
        E, M, K = expert_x.size()

        _, N, _ = self._w1.size()
        assert N % 2 == 0
        assert self._w1.size(0) == E
        assert self._w1.size(2) == K
        assert self._w2.size(0) == E
        assert self._w2.size(1) == K
        assert self._w2.size(2) == N // 2

        workspace = torch.empty((E, M, N), device=expert_x.device, dtype=torch.bfloat16)
        output = torch.empty((E, M, K), device=expert_x.device, dtype=torch.bfloat16)

        if self._use_fp8:
            assert self._w1_scale is not None and self._w2_scale is not None
            assert payload.expert_x_scale is not None

            expert_x_scale = payload.expert_x_scale

            assert expert_x_scale.size(0) == E
            assert expert_x_scale.size(1) == M
            assert expert_x_scale.size(2) == K // self.DEEPGEMM_BLOCK_SHAPE[0]

            m_grouped_fp8_gemm_nt_masked(
                (expert_x, expert_x_scale),
                (self._w1, self._w1_scale),
                workspace,
                expert_num_tokens,
                M,
            )
            a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_masked(
                workspace,
                expert_num_tokens,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[1],
                use_ue8m0=is_deep_gemm_e8m0_used(),
                eps=1e-10,
            )
            m_grouped_fp8_gemm_nt_masked(
                (a2q, a2q_scale),
                (self._w2, self._w2_scale),
                output,
                expert_num_tokens,
                M,
            )
        else:
            m_grouped_bf16_gemm_nt_masked(expert_x, self._w1, workspace, expert_num_tokens, M)
            a2q = silu_mul_bf16_deep_gemm_masked(workspace, expert_num_tokens, group_size=256)
            m_grouped_bf16_gemm_nt_masked(a2q, self._w2, output, expert_num_tokens, M)

        return output

    def op_gemm1(self, state: StateDict, payload: ExpertForwardPayload):
        # assert
        assert self._w1 is not None
        assert payload.expert_x is not None
        assert payload.expert_tokens_meta is not None
        expert_x = payload.expert_x
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens
        assert expert_num_tokens is not None
        assert expert_x.ndim == 3
        E, M, K = expert_x.size()
        _, N, _ = self._w1.size()
        assert N % 2 == 0
        assert self._w1.size(0) == E
        assert self._w1.size(2) == K
        # GEMM1
        workspace = torch.empty((E, M, N), device=expert_x.device, dtype=torch.bfloat16)
        if self._use_fp8:
            assert self._w1_scale is not None
            assert payload.expert_x_scale is not None
            expert_x_scale = payload.expert_x_scale
            assert expert_x_scale.size(0) == E
            assert expert_x_scale.size(1) == M
            assert expert_x_scale.size(2) == K // self.DEEPGEMM_BLOCK_SHAPE[0]
            m_grouped_fp8_gemm_nt_masked(
                (expert_x, expert_x_scale),
                (self._w1, self._w1_scale),
                workspace,
                expert_num_tokens,
                M,
            )
        else:
            m_grouped_bf16_gemm_nt_masked(expert_x, self._w1, workspace, expert_num_tokens, M)
        state.E = E
        state.M = M
        state.K = K
        state.N = N
        state.expert_num_tokens = expert_num_tokens
        return {"workspace": workspace}

    def op_silu_mul(self, state: StateDict, workspace: torch.Tensor):
        # fused silu and mul
        if self._use_fp8:
            a2q, a2q_scale = silu_mul_fp8_quant_deep_gemm_masked(
                workspace,
                state.expert_num_tokens,
                group_size=self.DEEPGEMM_BLOCK_SHAPE[1],
                use_ue8m0=is_deep_gemm_e8m0_used(),
                eps=1e-10,
            )
            return {"a2q": a2q, "a2q_scale": a2q_scale}
        else:
            a2q = silu_mul_bf16_deep_gemm_masked(workspace, state.expert_num_tokens, group_size=256)
            return {"a2q": a2q}

    def op_gemm2(self, state: StateDict, a2q: torch.Tensor, a2q_scale: Optional[torch.Tensor] = None):
        E = state.pop("E")
        M = state.pop("M")
        K = state.pop("K")
        N = state.pop("N")
        expert_num_tokens = state.pop("expert_num_tokens")
        # assert
        assert self._w2 is not None
        assert self._w2.size(0) == E
        assert self._w2.size(1) == K
        assert self._w2.size(2) == N // 2
        # GEMM2
        output = torch.empty((E, M, K), device=a2q.device, dtype=torch.bfloat16)
        if self._use_fp8:
            assert a2q_scale is not None
            assert self._w2_scale is not None
            m_grouped_fp8_gemm_nt_masked(
                (a2q, a2q_scale),
                (self._w2, self._w2_scale),
                output,
                expert_num_tokens,
                M,
            )
        else:
            m_grouped_bf16_gemm_nt_masked(a2q, self._w2, output, expert_num_tokens, M)
        return {"expert_output": output}
