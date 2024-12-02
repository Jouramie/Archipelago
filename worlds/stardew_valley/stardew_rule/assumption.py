from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import ClassVar, Hashable, Iterable


@dataclass(frozen=True)
class AssumptionState:
    UNKNOWN_BOUNDS: ClassVar[tuple[int, int]] = (0, sys.maxsize)
    combinable_values: dict[Hashable, tuple[int, int]] = field(default_factory=dict)
    """Lower bound is inclusive, upper bound is exclusive.
    """
    spots: dict[Hashable, bool] = field(default_factory=dict)

    def add_combinable_lower_bounds(self, lower_bounds: Iterable[tuple[Hashable, int]]) -> AssumptionState:
        new_bounds = {}

        for key, value in lower_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert upper_bound >= value

            if lower_bound >= value:
                continue

            new_bounds[key] = (value, upper_bound)

        return AssumptionState(self.combinable_values | new_bounds, self.spots)

    def add_combinable_upper_bounds(self, upper_bounds: Iterable[tuple[Hashable, int]]) -> AssumptionState:
        new_bounds = {}

        for key, value in upper_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert lower_bound <= value

            if upper_bound <= value:
                continue

            new_bounds[key] = (lower_bound, value)

        return AssumptionState(self.combinable_values | new_bounds, self.spots)

    def get_spot_state(self, resolution_hint: str, spot: str) -> bool | None:
        return self.spots.get((resolution_hint, spot))

    def set_spot_available(self, resolution_hint: str, spot: str) -> AssumptionState:
        return AssumptionState(self.combinable_values, {**self.spots, (resolution_hint, spot): True})

    def set_spot_unavailable(self, resolution_hint: str, spot: str) -> AssumptionState:
        return AssumptionState(self.combinable_values, {**self.spots, (resolution_hint, spot): False})

    def __str__(self):
        return (f"{{{', '.join(f'{key}: {self.str_bound(*bound)}' for key, bound in self.combinable_values.items())}}}|"
                f"{{{', '.join(f'{spot}: {available}' for spot, available in self.spots.items())}}}")

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def str_bound(lower_bound: int, upper_bound: int) -> str:
        if lower_bound == 0:
            return f"[0, {upper_bound})"
        if upper_bound == sys.maxsize:
            return f"[{lower_bound}, ~)"
        if lower_bound + 1 == upper_bound:
            return f"{lower_bound}"
        return f"[{lower_bound}, {upper_bound})"
