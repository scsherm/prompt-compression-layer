from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def curriculum_subset(items: Sequence[T], epoch: int) -> list[T]:
    values = list(items)
    if not values:
        return []
    warmup_count = min(len(values), 5)
    if epoch <= 2:
        return values[:warmup_count]
    if epoch <= 5:
        return values[: max(warmup_count, int(len(values) * 0.3))]
    return values
