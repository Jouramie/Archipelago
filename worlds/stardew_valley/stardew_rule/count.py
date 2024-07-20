from __future__ import annotations

import enum
from collections import Counter
from functools import cached_property
from typing import List, Callable, Optional, Dict, Tuple, Protocol, runtime_checkable, Hashable, Collection

from BaseClasses import CollectionState
from .base import BaseStardewRule, CombinableStardewRule, And, Or
from .protocol import StardewRule


class ShortCircuitPropagation(enum.Enum):
    NONE = enum.auto()
    POSITIVE = enum.auto()
    NEGATIVE = enum.auto()
    EQUAL = enum.auto()

    @property
    def opposite(self):
        if self is ShortCircuitPropagation.POSITIVE:
            return ShortCircuitPropagation.NEGATIVE
        elif self is ShortCircuitPropagation.NEGATIVE:
            return ShortCircuitPropagation.POSITIVE
        return self


@runtime_checkable
class CanShortCircuitLink(Protocol):

    def calculate_short_circuit_propagation(self, other: CanShortCircuitLink) -> ShortCircuitPropagation:
        """Return the link between two rules.
        NONE if there is no possible short-circuit propagation;
        POSITIVE if a True result from evaluating self short-circuit other;
        NEGATIVE if a False result from evaluating self short-circuit other;
        EQUAL if both resul;ts from evaluating self short-circuit other. In other words, both rules have the same combinable part.

        A POSITIVE or a NEGATIVE implies that other will short-circuit self in the opposite way.

        And/Or rules will always have a NONE link one another.
        """
        ...


class CombinableCanShortCircuitLink(CombinableStardewRule, CanShortCircuitLink):
    delegate: CombinableStardewRule

    def __init__(self, rule: CombinableStardewRule):
        self.delegate = rule

    def calculate_short_circuit_propagation(self, other: CanShortCircuitLink) -> ShortCircuitPropagation:
        if not isinstance(other, CombinableCanShortCircuitLink):
            return other.calculate_short_circuit_propagation(self)

        # Different key means nothing in common, so no short-circuit propagation.
        if self.combination_key != other.combination_key:
            return ShortCircuitPropagation.NONE

        # Both have same value, so rule is in fact the same.
        if self.value == other.value:
            return ShortCircuitPropagation.EQUAL

        # Self has a higher value, so self evaluating to True mean that other will be True was well.
        if self.value > other.value:
            return ShortCircuitPropagation.POSITIVE

        # Self has a lower value, so self evaluating to False mean that other will be False was well.
        return ShortCircuitPropagation.NEGATIVE

    @property
    def combination_key(self) -> Hashable:
        return self.delegate.combination_key

    @property
    def value(self):
        return self.delegate.value
        pass

    def __call__(self, state: CollectionState) -> bool:
        return self.delegate(state)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self.delegate.evaluate_while_simplifying(state)

    def __str__(self):
        return str(self.delegate)

    def __repr__(self):
        return repr(self.delegate)


class AndCanShortCircuitLink(And, CanShortCircuitLink):

    def __init__(self, rule: And):
        super().__init__(_combinable_rules=rule.combinable_rules, _simplification_state=rule.simplification_state)

    def calculate_short_circuit_propagation(self, other: CanShortCircuitLink) -> ShortCircuitPropagation:
        if isinstance(other, CombinableCanShortCircuitLink):
            return self.__calculate_short_circuit_propagation_combinable(other)
        elif isinstance(other, AndCanShortCircuitLink):
            return self.__calculate_short_circuit_propagation_and(other)
        return ShortCircuitPropagation.NONE

    def __calculate_short_circuit_propagation_combinable(self, other: CombinableCanShortCircuitLink):
        # Different key means not enough in common, so no short-circuit propagation.
        if other.combination_key not in self.combinable_rules:
            return ShortCircuitPropagation.NONE

        # Self has a higher value, meaning it is more restrictive than the other rule.
        value = self.combinable_rules[other.combination_key].value
        if value == other.value:
            return ShortCircuitPropagation.EQUAL

        if value > other.value:
            return ShortCircuitPropagation.POSITIVE

        # Self has a lower value, so self evaluating to False mean that other will be False was well.
        if len(self.combinable_rules) == 1 and value < other.value:
            return ShortCircuitPropagation.NEGATIVE

        # Values are diverging, so no short-circuit propagation.
        return ShortCircuitPropagation.NONE

    def __calculate_short_circuit_propagation_and(self, other: AndCanShortCircuitLink):
        # No combinable rules, so no short-circuit propagation.
        if not self.combinable_rules or not other.combinable_rules:
            return ShortCircuitPropagation.NONE

        if self.combinable_rules == other.combinable_rules:
            return ShortCircuitPropagation.EQUAL

        # No intersection means rules are diverging, so no short-circuit propagation.
        intersection = self.combinable_rules.keys() & other.combinable_rules.keys()
        if not intersection:
            return ShortCircuitPropagation.NONE

        if len(intersection) == len(self.combinable_rules):
            smaller = self
            larger = other
        elif len(intersection) == len(other.combinable_rules):
            smaller = other
            larger = self
        else:
            # Both have different keys, which means rules are diverging. No short-circuit propagation.
            return ShortCircuitPropagation.NONE

        # larger has a higher value, meaning it is more restrictive than the other rule.
        more_restrictive = all(larger.combinable_rules[key].value >= smaller.combinable_rules[key].value for key in smaller.combinable_rules)
        if more_restrictive:
            if self is larger:
                return ShortCircuitPropagation.POSITIVE
            return ShortCircuitPropagation.NEGATIVE

        # self has a lower value, so it is less restrictive that other rule.
        less_restrictive = all(larger.combinable_rules[key].value <= smaller.combinable_rules[key].value for key in self.combinable_rules)
        if len(self.combinable_rules) == len(other.combinable_rules) and less_restrictive:
            if self is larger:
                return ShortCircuitPropagation.NEGATIVE
            return ShortCircuitPropagation.POSITIVE

        print("man I never thought this would happen...")
        # Self has a lower or diverging values, so self evaluating to False mean that other will be False was well.
        return ShortCircuitPropagation.NONE


class OrCanShortCircuitLink(Or, CanShortCircuitLink):

    def __init__(self, rule: Or):
        super().__init__(_combinable_rules=rule.combinable_rules, _simplification_state=rule.simplification_state)

    def calculate_short_circuit_propagation(self, other: CanShortCircuitLink) -> ShortCircuitPropagation:
        # TODO see that later
        print("hey maybe you should implement or short circuit propagation...")
        return ShortCircuitPropagation.NONE


class CannotShortCircuitLink(BaseStardewRule, CanShortCircuitLink):
    delegate: StardewRule

    def __init__(self, rule: StardewRule):
        self.delegate = rule

    def __call__(self, state: CollectionState) -> bool:
        return self.delegate(state)

    def __and__(self, other: StardewRule):
        return self.delegate & other

    def __or__(self, other: StardewRule):
        return self.delegate | other

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self.delegate.evaluate_while_simplifying(state)

    def calculate_short_circuit_propagation(self, other: CanShortCircuitLink) -> ShortCircuitPropagation:
        return ShortCircuitPropagation.NONE


def to_can_short_circuit_link(rule: StardewRule) -> CanShortCircuitLink:
    if isinstance(rule, And):
        return AndCanShortCircuitLink(rule)
    elif isinstance(rule, Or):
        return OrCanShortCircuitLink(rule)
    elif isinstance(rule, CombinableStardewRule):
        return CombinableCanShortCircuitLink(rule)
    return CannotShortCircuitLink(rule)


def create_special_count(rules: Collection[StardewRule], count: int) -> Count:
    short_circuit_links = [to_can_short_circuit_link(rule) for rule in rules]

    link_results = {}

    for ri in short_circuit_links:
        for rj in short_circuit_links:
            link_results[ri, rj] = ri.calculate_short_circuit_propagation(rj)

    return Count(list(rules), count)


class Count(BaseStardewRule):
    count: int
    rules: List[StardewRule]
    counter: Counter[StardewRule]
    evaluate: Callable[[CollectionState], bool]

    total: Optional[int]
    rule_mapping: Optional[Dict[StardewRule, StardewRule]]

    def __init__(self, rules: List[StardewRule], count: int):
        self.count = count
        self.counter = Counter(rules)

        if len(self.counter) / len(rules) < .66:
            # Checking if it's worth using the count operation with shortcircuit or not. Value should be fine-tuned when Count has more usage.
            self.total = sum(self.counter.values())
            self.rules = sorted(self.counter.keys(), key=lambda x: self.counter[x], reverse=True)
            self.rule_mapping = {}
            self.evaluate = self.evaluate_with_shortcircuit
        else:
            self.rules = rules
            self.evaluate = self.evaluate_without_shortcircuit

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate(state)

    def evaluate_without_shortcircuit(self, state: CollectionState) -> bool:
        c = 0
        for i in range(self.rules_count):
            self.rules[i], value = self.rules[i].evaluate_while_simplifying(state)
            if value:
                c += 1

            if c >= self.count:
                return True
            if c + self.rules_count - i < self.count:
                break

        return False

    def evaluate_with_shortcircuit(self, state: CollectionState) -> bool:
        c = 0
        t = self.total

        for rule in self.rules:
            evaluation_value = self.call_evaluate_while_simplifying_cached(rule, state)
            rule_value = self.counter[rule]

            if evaluation_value:
                c += rule_value
            else:
                t -= rule_value

            if c >= self.count:
                return True
            elif t < self.count:
                break

        return False

    def call_evaluate_while_simplifying_cached(self, rule: StardewRule, state: CollectionState) -> bool:
        try:
            # A mapping table with the original rule is used here because two rules could resolve to the same rule.
            #  This would require to change the counter to merge both rules, and quickly become complicated.
            return self.rule_mapping[rule](state)
        except KeyError:
            self.rule_mapping[rule], value = rule.evaluate_while_simplifying(state)
            return value

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    @cached_property
    def rules_count(self):
        return len(self.rules)

    def __repr__(self):
        return f"Received {self.count} [{', '.join(f'{value}x {repr(rule)}' for rule, value in self.counter.items())}]"
