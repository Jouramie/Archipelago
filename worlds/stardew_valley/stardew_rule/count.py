from __future__ import annotations

from collections import Counter
from typing import List, Callable, Optional, Tuple, cast, Collection

from BaseClasses import CollectionState
from .base import BaseStardewRule, AssumptionState, Or, And
from .literal import false_, true_
from .protocol import StardewRule


class Count(BaseStardewRule):
    count: int
    rules: List[StardewRule]
    rules_and_points: List[Tuple[StardewRule, int]]
    evaluate: Callable[[CollectionState], bool]

    total: int

    def __init__(self, rules: Collection[StardewRule], count: int, _rules_and_points: Optional[List[Tuple[StardewRule, int]]] = None):
        self.count = count
        if _rules_and_points is not None:
            self.rules_and_points = _rules_and_points
        else:
            self.rules_and_points = sorted(Counter(rules).items(), key=lambda x: x[1], reverse=True)
        self.total = sum(point for _, point in self.rules_and_points)
        self.rules = [rule for rule, _ in self.rules_and_points]

    @staticmethod
    def from_leftovers(leftovers: List[Tuple[StardewRule, int]], count: int):
        return Count([], count, _rules_and_points=leftovers)

    def __call__(self, state: CollectionState) -> bool:
        min_points = 0
        max_points = self.total
        goal = self.count
        rules = self.rules

        for i, (_, points) in enumerate(self.rules_and_points):
            rules[i], evaluation_value = rules[i].evaluate_while_simplifying(state)

            if evaluation_value:
                min_points += points
                if min_points >= goal:
                    return True
            else:
                max_points -= points
                if max_points < goal:
                    return False

        assert False, "Should have returned before."

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def deep_simplify_knowing(self, assumption_state: AssumptionState) -> StardewRule:
        min_points = 0
        max_points = self.total
        goal = self.count
        simplified_rules = Counter()

        for rule, points in self.rules_and_points:
            simplified_rule = cast(BaseStardewRule, rule).deep_simplify_knowing(assumption_state)

            if simplified_rule is true_:
                min_points += points
                if min_points >= goal:
                    return true_
                continue
            elif simplified_rule is false_:
                max_points -= points
                if max_points < goal:
                    return false_
                continue
            else:
                simplified_rules.update({simplified_rule: points})

        if len(simplified_rules) == 1:
            return next(iter(simplified_rules.keys()))

        new_goal = goal - min_points
        if new_goal == 1:
            return Or(*simplified_rules.keys())

        if new_goal == sum(points for _, points in simplified_rules.items()):
            return And(*simplified_rules.keys())

        return Count.from_leftovers(list(simplified_rules.items()), new_goal)

    def __repr__(self):
        return f"Received {self.count} [{', '.join(f'{value}x {repr(rule)}' for rule, value in self.rules_and_points)}]"
