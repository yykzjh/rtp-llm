from typing import List
from dataclasses import dataclass

from torch import nn

from rtp_llm.config.generate_config import RoleType
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
    layer: nn.Module, layer_id: int, role_type: RoleType
) -> OperationStrategy:
    if role_type == RoleType.PREFILL or role_type == RoleType.PDFUSION:
        return _create_qwen3_moe_prefill_layer_operation_strategy(layer, layer_id)
    elif role_type == RoleType.DECODE:
        return _create_qwen3_moe_decode_layer_operation_strategy(layer, layer_id)
    else:
        raise NotImplementedError(f"Role type {role_type} is not implemented")


def _create_qwen3_moe_prefill_layer_operation_strategy(layer: nn.Module, layer_id: int) -> OperationStrategy:
    return OperationStrategy(
        stage_delta=0,
        operations=[
            YieldOperation(),
            layer.forward,
        ],
    )


def _create_qwen3_moe_decode_layer_operation_strategy(layer: nn.Module, layer_id: int) -> OperationStrategy:
    return OperationStrategy(
        stage_delta=0,
        operations=[
            YieldOperation(),
            layer.forward,
        ],
    )
