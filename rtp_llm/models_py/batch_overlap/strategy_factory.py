from typing import Optional

from torch import nn

from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.batch_overlap.operations import convert_operations_to_stages
from rtp_llm.models_py.batch_overlap.strategy import (
    OperationStrategy,
    StageStrategy,
    create_qwen3_moe_layer_operation_strategy,
)


__all__ = [
    "StrategyFactory",
]


class StrategyFactory:

    @staticmethod
    def create_stage_strategy(
        config: GptInitModelParameters, model: GptModelBase, layers: nn.ModuleList, stage_delta: Optional[int] = None
    ) -> StageStrategy:
        layer_name = layers[0].__class__.__name__
        if layer_name == "Qwen3MoeDecoderLayer":
            qwen3_moe_model_operation_strategy = OperationStrategy.concat(
                [
                    create_qwen3_moe_layer_operation_strategy(model, layer, layer_id, len(layers), config.role_type)
                    for layer_id, layer in enumerate(layers)
                ]
            )
            if stage_delta is not None:
                qwen3_moe_model_operation_strategy.stage_delta = stage_delta
            qwen3_moe_model_stage_strategy = StageStrategy(
                stages=convert_operations_to_stages(qwen3_moe_model_operation_strategy.operations),
                stage_delta=qwen3_moe_model_operation_strategy.stage_delta,
            )
            return qwen3_moe_model_stage_strategy
        else:
            raise NotImplementedError(f"Strategy for {layer_name} is not implemented")
