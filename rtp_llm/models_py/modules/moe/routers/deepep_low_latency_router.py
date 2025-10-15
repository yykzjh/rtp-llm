import os
from typing import Any, Callable, Optional, Tuple

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.distributed.deepep_wrapper import get_deepep_wrapper
from rtp_llm.models_py.modules.moe.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
    TopKWeightAndReduce,
)
from rtp_llm.models_py.modules.moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from rtp_llm.models_py.modules.moe.utils import FusedMoEQuantConfig
from rtp_llm.models_py.batch_overlap.stage_executor import StateDict

# DeepEP Low-Latency supports hidden sizes
SUPPORTED_HIDDEN_SIZES = [1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]


class DeepEpLowLatencyRouter(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's low-latency communication primitives.

    This router dispatches tokens to experts and receives results from experts across all ep ranks.
    """

    def __init__(
        self,
        config: GptInitModelParameters,
        use_fp8_dispatch: bool = True,
        return_recv_hook: bool = False,
    ):
        super().__init__()
        self._config = config
        self._num_experts = get_deepep_wrapper().num_experts
        self._buffer = get_deepep_wrapper().buffer
        self._num_max_dispatch_tokens_per_rank = get_deepep_wrapper().ll_num_max_token_per_rank
        self._use_fp8_dispatch = use_fp8_dispatch
        self._return_recv_hook = return_recv_hook
        self._opt_level = int(os.environ.get("ACCL_LOW_LATENCY_OPTIMIZE", 1))
        self._handle: Optional[Tuple[Any, ...]] = None

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # assert
        assert a1.dim() == 2 and topk_ids.dim() == 2
        assert a1.size(0) == topk_ids.size(0)
        num_tokens = a1.size(0)
        num_dispatch_tokens_per_rank = (num_tokens + self._config.tp_size - 1) // self._config.tp_size
        assert (
            num_dispatch_tokens_per_rank <= self._num_max_dispatch_tokens_per_rank
        ), f"Number of dispatch tokens {num_dispatch_tokens_per_rank} exceeds the maximum number of dispatch tokens per rank {self._num_max_dispatch_tokens_per_rank}."
        hidden_dim = a1.size(1)
        assert (
            hidden_dim in SUPPORTED_HIDDEN_SIZES
        ), f"Hidden Size {hidden_dim} not in supported list of hidden sizes: {SUPPORTED_HIDDEN_SIZES}."
        if self._use_fp8_dispatch and quant_config.block_shape is not None:
            assert (
                hidden_dim % quant_config.block_shape[0] == 0
            ), f"DeepEP Low-Latency only supports hidden sizes that are divisible by {quant_config.block_shape[0]}."
        assert self._use_fp8_dispatch and (
            (quant_config.block_shape is not None) ^ (quant_config.is_per_act_token)
        ), "DeepEP Low-Latency only supports block quantization or per-token quantization for fp8 dispatch."
        assert (
            not self._use_fp8_dispatch and quant_config.block_shape is None and not quant_config.is_per_act_token
        ), "DeepEP Low-Latency only supports fp8 block quantization or per-token quantization."
        assert self._handle is None, "DeepEP Low-latency dispatch handle should be clean before prepare()."
        # dispatch
        topk_ids = topk_ids.to(torch.int64)
        expert_x, expert_num_tokens, self._handle, _, hook = self._buffer.low_latency_dispatch(
            a1,
            topk_ids,
            self._num_max_dispatch_tokens_per_rank,
            num_experts,
            use_fp8=self._use_fp8_dispatch,
            async_finish=False,
            return_recv_hook=self._return_recv_hook,
            opt_level=self._opt_level,
            pertoken_quant=quant_config.is_per_act_token,
        )
        if self._return_recv_hook and hook is not None:
            hook()
        if quant_config.is_per_act_token:
            assert expert_x[0].shape[1] == expert_x[1].shape[1]
            assert expert_x[1].shape[-1] == 1
        # return payload
        return ExpertForwardPayload(
            expert_x=expert_x[0] if self._use_fp8_dispatch else expert_x,
            expert_x_scale=expert_x[1] if self._use_fp8_dispatch else None,
            expert_x_origin_dtype=a1.dtype,
            expert_topk_weights=None,
            expert_topk_ids=None,
            expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None),
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: TopKWeightAndReduce,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        """
        Combines expert outputs back to all original ranks.
        """
        # assert
        assert isinstance(
            weight_and_reduce_impl, TopKWeightAndReduceDelegate
        ), "Weight application and reduction happens in the combine kernel."
        assert self._handle is not None, "DeepEP Low-latency combine handle is missing for finalize()."
        # combine
        topk_ids = topk_ids.to(torch.int64)
        combined_x, _, hook = self._buffer.low_latency_combine(
            fused_expert_output,
            topk_ids,
            topk_weights,
            self._handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=self._return_recv_hook,
            opt_level=self._opt_level,
        )
        if self._return_recv_hook and hook is not None:
            hook()
        # reset handle
        self._handle = None
        return combined_x

    def op_dispatch_send(self, state: StateDict, hidden_states: torch.Tensor):
        quant_config = state.quant_config
        topk_ids = state.topk_ids
        # assert
        assert hidden_states.dim() == 2 and topk_ids.dim() == 2
        assert hidden_states.size(0) == topk_ids.size(0)
        num_tokens = hidden_states.size(0)
        num_dispatch_tokens_per_rank = (num_tokens + self._config.tp_size - 1) // self._config.tp_size
        assert (
            num_dispatch_tokens_per_rank <= self._num_max_dispatch_tokens_per_rank
        ), f"Number of dispatch tokens {num_dispatch_tokens_per_rank} exceeds the maximum number of dispatch tokens per rank {self._num_max_dispatch_tokens_per_rank}."
        hidden_dim = hidden_states.size(1)
        assert (
            hidden_dim in SUPPORTED_HIDDEN_SIZES
        ), f"Hidden Size {hidden_dim} not in supported list of hidden sizes: {SUPPORTED_HIDDEN_SIZES}."
        if self._use_fp8_dispatch and quant_config.block_shape is not None:
            assert (
                hidden_dim % quant_config.block_shape[0] == 0
            ), f"DeepEP Low-Latency only supports hidden sizes that are divisible by {quant_config.block_shape[0]}."
        assert self._use_fp8_dispatch and (
            (quant_config.block_shape is not None) ^ (quant_config.is_per_act_token)
        ), "DeepEP Low-Latency only supports block quantization or per-token quantization for fp8 dispatch."
        assert (
            not self._use_fp8_dispatch and quant_config.block_shape is None and not quant_config.is_per_act_token
        ), "DeepEP Low-Latency only supports fp8 block quantization or per-token quantization."
        # dispatch
        topk_ids = topk_ids.to(torch.int64)
        expert_x, expert_num_tokens, handle, _, hook = self._buffer.low_latency_dispatch(
            hidden_states,
            topk_ids,
            self._num_max_dispatch_tokens_per_rank,
            self._num_experts,
            use_fp8=self._use_fp8_dispatch,
            async_finish=False,
            return_recv_hook=self._return_recv_hook,
            opt_level=self._opt_level,
            pertoken_quant=quant_config.is_per_act_token,
        )
        state.handle = handle
        state.expert_x_origin_dtype = hidden_states.dtype
        return {
            "expert_x": expert_x,
            "expert_num_tokens": expert_num_tokens,
            "hook": hook,
        }

    def op_dispatch_recv(
        self,
        state: StateDict,
        expert_x: torch.Tensor,
        expert_num_tokens: torch.Tensor,
        hook: Optional[Callable[[], None]],
    ):
        if self._return_recv_hook and hook is not None:
            hook()
        quant_config = state.pop("quant_config")
        expert_x_origin_dtype = state.pop("expert_x_origin_dtype")
        if quant_config.is_per_act_token:
            assert expert_x[0].shape[1] == expert_x[1].shape[1]
            assert expert_x[1].shape[-1] == 1
        payload = ExpertForwardPayload(
            expert_x=expert_x[0] if self._use_fp8_dispatch else expert_x,
            expert_x_scale=expert_x[1] if self._use_fp8_dispatch else None,
            expert_x_origin_dtype=expert_x_origin_dtype,
            expert_topk_weights=None,
            expert_topk_ids=None,
            expert_tokens_meta=ExpertTokensMetadata(expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None),
        )
        return {
            "payload": payload,
        }

    def op_combine_send(self, state: StateDict, expert_output: torch.Tensor):
        topk_ids = state.pop("topk_ids")
        topk_weights = state.pop("topk_weights")
        handle = state.pop("handle")
        # combine
        topk_ids = topk_ids.to(torch.int64)
        combined_x, _, hook = self._buffer.low_latency_combine(
            expert_output,
            topk_ids,
            topk_weights,
            handle,
            zero_copy=False,
            async_finish=False,
            return_recv_hook=self._return_recv_hook,
            opt_level=self._opt_level,
        )
        return {"combined_x": combined_x, "hook": hook}

    def op_combine_recv(self, state: StateDict, combined_x: torch.Tensor, hook: Optional[Callable[[], None]]):
        if self._return_recv_hook and hook is not None:
            hook()
        return {
            "hidden_states": combined_x,
        }
