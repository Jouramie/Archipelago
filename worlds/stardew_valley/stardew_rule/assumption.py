from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import ClassVar, Tuple, Dict, Hashable, Iterable


@dataclass(frozen=True)
class AssumptionState:
    UNKNOWN_BOUNDS: ClassVar[Tuple[int, int]] = (0, sys.maxsize)
    """Lower bound is inclusive, upper bound is exclusive.
    """
    combinable_values: Dict[Hashable, Tuple[int, int]] = field(default_factory=dict)
    spots: Dict[Hashable, bool] = field(default_factory=dict)

    def add_combinable_lower_bounds(self, lower_bounds: Iterable[Tuple[Hashable, int]]) -> AssumptionState:
        new_bounds = {}

        for key, value in lower_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert upper_bound >= value

            if lower_bound >= value:
                continue

            new_bounds[key] = (value, upper_bound)

        return AssumptionState(self.combinable_values | new_bounds, self.spots)

    def add_combinable_upper_bounds(self, upper_bounds: Iterable[Tuple[Hashable, int]]) -> AssumptionState:
        new_bounds = {}

        for key, value in upper_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert lower_bound <= value

            if upper_bound <= value:
                continue

            new_bounds[key] = (lower_bound, value)

        return AssumptionState(self.combinable_values | new_bounds, self.spots)

    def set_spot_available(self, spot: Hashable) -> AssumptionState:
        return AssumptionState(self.combinable_values, {**self.spots, spot: True})

    def set_spot_unavailable(self, spot: Hashable) -> AssumptionState:
        return AssumptionState(self.combinable_values, {**self.spots, spot: False})

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
