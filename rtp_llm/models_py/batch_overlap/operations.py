from dataclasses import dataclass
from typing import Any, Callable, Generator, List, Union


__all__ = [
    "Stage",
    "Operation",
    "YieldOperation",
    "ExecutionOperation",
    "convert_operations_to_stages",
]


class YieldOperation:
    pass


@dataclass
class ExecutionOperation:
    debug_name: str
    fn: Callable[..., Any]


Stage = List[ExecutionOperation]
Operation = Union[YieldOperation, Callable[..., Any]]
DecoratedOperation = Union[YieldOperation, ExecutionOperation]


def convert_operations_to_stages(operations: List[Operation]) -> List[Stage]:
    decorated_operations: List[DecoratedOperation] = _decorate_operations(operations)
    operation_chunks = list(_chunk_by_separator(decorated_operations, lambda op: isinstance(op, YieldOperation)))
    assert all(len(chunk) > 0 for chunk in operation_chunks)
    return operation_chunks


def _chunk_by_separator(
    items: List[DecoratedOperation],
    is_separator: Callable[[DecoratedOperation], bool],
) -> Generator[Stage, None, None]:
    pending_items: Stage = []
    for item in items:
        if is_separator(item):
            yield pending_items
            pending_items = []
        else:
            if isinstance(item, ExecutionOperation):
                pending_items.append(item)
    if len(pending_items) > 0:
        yield pending_items


def _decorate_operations(operations: List[Operation], debug_name_prefix: str = "") -> List[DecoratedOperation]:
    return [_decorate_operation(op, debug_name_prefix) for op in operations]


def _decorate_operation(operation: Operation, debug_name_prefix: str) -> DecoratedOperation:
    if isinstance(operation, YieldOperation):
        return operation
    return ExecutionOperation(
        debug_name=debug_name_prefix + getattr(operation, "__name__", "unknown").replace("op_", ""),
        fn=operation,
    )
