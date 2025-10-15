from typing import List
from dataclasses import dataclass

from torch import nn

from rtp_llm.config.generate_config import RoleType
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.batch_overlap.operations import Operation, Stage, YieldOperation


__all__ = [
    "OperationStrategy",
    "StageStrategy",
    "create_qwen3_moe_layer_operation_strategy",
]


@dataclass
class OperationStrategy:
    operations: List[Operation]
    stage_delta: int = 0

    @classmethod
    def concat(cls, operations_strategies: List["OperationStrategy"]) -> "OperationStrategy":
        return OperationStrategy(
            operations=[x for operations_strategy in operations_strategies for x in operations_strategy.operations],
            stage_delta=_assert_all_same(
                [operations_strategy.stage_delta for operations_strategy in operations_strategies]
            ),
        )


@dataclass
class StageStrategy:
    stages: List[Stage]
    stage_delta: int = 0


def _assert_all_same(items: List[int]) -> int:
    assert all(item == items[0] for item in items)
    return items[0]


def create_qwen3_moe_layer_operation_strategy(
    model: GptModelBase, layer: nn.Module, layer_id: int, num_layers: int, role_type: RoleType
) -> OperationStrategy:
    if role_type == RoleType.PREFILL:
        return _create_qwen3_moe_prefill_layer_operation_strategy(model, layer, layer_id, num_layers)
    elif role_type == RoleType.DECODE or role_type == RoleType.PDFUSION:
        return _create_qwen3_moe_decode_layer_operation_strategy(model, layer, layer_id, num_layers)
    else:
        raise NotImplementedError(f"Role type {role_type} is not implemented")


def _create_qwen3_moe_prefill_layer_operation_strategy(
    model: GptModelBase, layer: nn.Module, layer_id: int, num_layers: int
) -> OperationStrategy:
    return OperationStrategy(
        stage_delta=0,
        operations=[
            YieldOperation(),
            layer.forward,
        ],
    )


def _create_qwen3_moe_decode_layer_operation_strategy(
    model: GptModelBase, layer: nn.Module, layer_id: int, num_layers: int
) -> OperationStrategy:
    return OperationStrategy(
        stage_delta=2,
        operations=(
            [
                model.op_prepare_forward,
                model.op_embed_tokens,
            ]
            if layer_id == 0
            else []
        )
        + [
            layer.op_store_residual,
            layer.op_input_layernorm,
            layer.self_attn.op_qkv_proj,
            layer.self_attn.op_qk_fuse_norm,
            YieldOperation(),
            layer.self_attn.op_fmha_impl,
            layer.self_attn.op_o_proj,
            layer.self_attn.op_all_reduce,
            layer.op_add_residual,
            layer.op_store_residual,
            layer.op_post_attention_layernorm,
            layer.mlp.op_gate,
            layer.mlp.op_select_topk_experts,
            YieldOperation(),
            layer.mlp.fused_moe.router.op_dispatch_send,
            YieldOperation(),
            layer.mlp.fused_moe.router.op_dispatch_recv,
            layer.mlp.fused_moe.fused_experts.op_gemm1,
            layer.mlp.fused_moe.fused_experts.op_silu_mul,
            layer.mlp.fused_moe.fused_experts.op_gemm2,
            layer.mlp.fused_moe.router.op_combine_send,
            YieldOperation(),
            layer.mlp.fused_moe.router.op_combine_recv,
            YieldOperation(),
            layer.op_add_residual,
        ]
        + (
            [
                model.op_finalize_forward,
            ]
            if layer_id == num_layers - 1
            else []
        ),
    )
