from dataclasses import dataclass
from functools import cached_property
from typing import Iterable, Union, List, Tuple, Hashable, TYPE_CHECKING

from BaseClasses import CollectionState
from .base import BaseStardewRule, CombinableStardewRule, AssumptionState
from .literal import false_, true_
from .protocol import StardewRule
from ..strings.ap_names.event_names import Event

if TYPE_CHECKING:
    from .. import StardewValleyWorld


class TotalReceived(BaseStardewRule):
    count: int
    items: Iterable[str]
    player: int

    def __init__(self, count: int, items: Union[str, Iterable[str]], player: int):
        items_list: List[str]

        if isinstance(items, Iterable):
            items_list = [*items]
        else:
            items_list = [items]

        self.player = player
        self.items = items_list
        self.count = count

    def __call__(self, state: CollectionState) -> bool:
        c = 0
        for item in self.items:
            c += state.count(item, self.player)
            if c >= self.count:
                return True
        return False

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        return f"Received {self.count} {self.items}"


@dataclass(frozen=True)
class Received(CombinableStardewRule):
    item: str
    player: int
    count: int
    event: bool = False
    """Helps `explain` to know it can dig into a location with the same name."""

    @property
    def combination_key(self) -> Hashable:
        return self.item

    @property
    def value(self):
        return self.count

    def __call__(self, state: CollectionState) -> bool:
        return state.has(self.item, self.player, self.count)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        if self.count == 1:
            return f"Received {'event ' if self.event else ''}{self.item}"
        return f"Received {'event ' if self.event else ''}{self.count} {self.item}"


@dataclass(frozen=True)
class Reach(BaseStardewRule):
    spot: str
    resolution_hint: str
    player: int

    def __call__(self, state: CollectionState) -> bool:
        if self.resolution_hint == 'Region' and self.spot not in state.multiworld.regions.region_cache[self.player]:
            return False
        return state.can_reach(self.spot, self.resolution_hint, self.player)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def simplify_knowing(self, assumption_state: AssumptionState) -> StardewRule:
        is_available = assumption_state.spots.get(self.key)
        if is_available is None:
            return self

        if is_available:
            return true_
        return false_

    def add_lower_bounds(self, assumption_state: AssumptionState) -> AssumptionState:
        return assumption_state.set_spot_available(self.key)

    def add_upper_bounds(self, assumption_state: AssumptionState) -> AssumptionState:
        return assumption_state.set_spot_unavailable(self.key)

    @cached_property
    def key(self) -> Hashable:
        return f"{self.resolution_hint}/{self.spot}"

    def __repr__(self):
        return f"Reach {self.resolution_hint} {self.spot}"


class HasProgressionPercent(Received):
    def __init__(self, player: int, percent: int):
        super().__init__(Event.received_progression_percent, player, percent, event=True)

    def __post_init__(self):
        assert self.count > 0, "HasProgressionPercent rule must be above 0%"
        assert self.count <= 100, "HasProgressionPercent rule can't require more than 100% of items"

    def __repr__(self):
        return f"Received {self.count}% progression items"
