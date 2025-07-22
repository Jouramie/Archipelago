from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable, Tuple

from BaseClasses import CollectionState


@runtime_checkable
class StardewRule(Protocol):

    @abstractmethod
    def __call__(self, state: CollectionState) -> bool:
        ...

    @abstractmethod
    def __and__(self, other: StardewRule):
        ...

    @abstractmethod
    def __or__(self, other: StardewRule):
        ...

    # TODO move this to some kind of internal place
    @abstractmethod
    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        ...
