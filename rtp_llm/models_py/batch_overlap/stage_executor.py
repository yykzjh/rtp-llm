import os
from typing import Any, Dict, Sequence
from contextlib import contextmanager
import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.batch_overlap.strategy import StageStrategy
from rtp_llm.ops.libth_transformer import PyModelInputs


__all__ = [
    "StateDict",
    "StageExecutor",
]


_ENABLE_PROFILE = bool(int(os.environ.get("FT_SERVER_TEST", "0")))


@contextmanager
def _annotate_region(debug_name: str):
    if _ENABLE_PROFILE:
        with torch.autograd.profiler.record_function(debug_name):
            yield
    else:
        yield


class StateDict:
    def __init__(self):
        self._data: Dict[str, Any] = {}

    def __setattr__(self, key: str, value: Any):
        if key == "_data":
            super().__setattr__(key, value)
            return
        assert key not in self._data, f"`{key}` already exist, are you sure you want to override it?"
        self._data[key] = value

    def __getattr__(self, item: str):
        return self._data[item]

    def __delattr__(self, item: str):
        del self._data[item]

    def pop(self, item: str):
        return self._data.pop(item)

    def update(self, values: Dict[str, Any]):
        for k, v in values.items():
            setattr(self, k, v)

    def get(self, item: str):
        return self._data.get(item)

    def clear(self, expect_keys: Sequence[str]):
        if set(self._data.keys()) != set(expect_keys):
            raise Exception(
                f"Unexpected keys when clearning. This may indicate you do not release memory early enough but leave it to here. {list(self._data.keys())=} {expect_keys=}"
            )
        self._data.clear()


class StageExecutor:
    def __init__(self, stage_strategy: StageStrategy, debug_name: str = ""):
        self._debug_name = debug_name
        self._stages = stage_strategy.stages
        self._stage_delta = stage_strategy.stage_delta
        self._index = 0
        self._stage_output = None
        self._stage_state = StateDict()

    def next(self):
        assert not self.done, f"{self._debug_name} has done but still execute next"

        stage = self._stages[self._index]

        with _annotate_region(debug_name=f"{self._debug_name}{self._index}"):
            for op in stage:
                with _annotate_region(debug_name=op.debug_name):
                    self._stage_output = op.fn(
                        state=self._stage_state,
                        **(self._stage_output if self._stage_output is not None else {}),
                    )

        self._index += 1

    def initialize(self, stage_input: PyModelInputs):
        self._index = 0
        self._stage_output = {"stage_input": stage_input}
        self._stage_state.clear(expect_keys=[])

    @property
    def output(self):
        assert self.done
        return self._stage_output["stage_output"]

    @property
    def done(self):
        return self._index >= self.num_stages

    @property
    def num_stages(self):
        return len(self._stages)

    @property
    def current_stage_index(self):
        return self._index

    @property
    def stage_delta(self):
        return self._stage_delta
