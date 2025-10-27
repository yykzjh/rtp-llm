from typing import List, Optional, Sequence

from torch import nn

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.batch_overlap.stage_executor import StageExecutor
from rtp_llm.models_py.batch_overlap.strategy_factory import StrategyFactory
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.ops import PyModelInputs, PyModelOutputs


class MicroBatchExecutor:

    def __init__(
        self,
        config: GptInitModelParameters,
        model: GptModelBase,
        layers: nn.ModuleList,
        stage_deltas: Optional[Sequence[int]] = None,
    ):
        # Initialize arguments
        self._config = config
        self._num_micro_batches = (
            self._config.gpt_init_params.device_resource_config.enable_layer_micro_batch
        )
        # Check arguments
        assert (
            self._num_micro_batches > 1
        ), "enable_layer_micro_batch must be greater than 1"
        if stage_deltas is not None:
            assert self._num_micro_batches == len(
                stage_deltas
            ), f"stage_deltas length {len(stage_deltas)} is not equal to enable_layer_micro_batch {self._num_micro_batches}"
            assert stage_deltas[0] == 0, "stage_deltas[0] must be 0"

        # Initialize strategy with specified stage_deltas
        self._stage_strategies = [
            StrategyFactory.create_stage_strategy(
                self._config,
                model,
                layers,
                stage_deltas[micro_batch_idx] if stage_deltas is not None else None,
            )
            for micro_batch_idx in range(self._num_micro_batches)
        ]
        if stage_deltas is None:
            self._stage_strategies[0].stage_delta = 0

        # Initialize stage executors corresponding to each micro-batch
        self._stage_executors = [
            StageExecutor(stage_strategy, debug_name=f"micro_batch_{micro_batch_idx}")
            for micro_batch_idx, stage_strategy in enumerate(self._stage_strategies)
        ]

    def tbo_execute(
        self,
        stage_inputs: List[PyModelInputs],
    ) -> List[PyModelOutputs]:
        # Assert
        assert (
            len(stage_inputs) == 2 and self._num_micro_batches == 2
        ), "stage_inputs length must be 2 and enable_layer_micro_batch must be 2"

        # Prepare input data for each stage executor
        self._stage_executors[0].initialize(stage_inputs[0])
        self._stage_executors[1].initialize(stage_inputs[1])

        # Orchestrate micro-batch pipeline
        # Pipeline fill
        for _ in range(self._stage_executors[1].stage_delta):
            self._stage_executors[0].next()

        # Pipeline steady-state
        for _ in range(
            self._stage_executors[0].num_stages
            - self._stage_executors[0].current_stage_index
        ):
            self._stage_executors[0].next()
            self._stage_executors[1].next()

        # Pipeline flush
        for _ in range(self._stage_executors[1].stage_delta):
            self._stage_executors[1].next()

        assert all(
            stage_executor.done for stage_executor in self._stage_executors
        ), "all stage executors must be done"
        return [stage_executor.output for stage_executor in self._stage_executors]
