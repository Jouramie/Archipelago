from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Optional, Tuple, List, Union, Dict, Hashable, Set

import networkx as nx

from BaseClasses import CollectionState
from .base import BaseStardewRule, ShortCircuitPropagation, CombinableStardewRule, Or, And
from .protocol import StardewRule
from .state import HasProgressionPercent, Reach, Received


@dataclass(frozen=True)
class Node:
    true_edge: Optional[Edge]
    false_edge: Optional[Edge]
    rule: BaseStardewRule

    @staticmethod
    def leaf(rule: BaseStardewRule):
        return Node(None, None, rule)

    @cached_property
    def is_leaf(self):
        return self.true_edge is self.false_edge is None


@dataclass(frozen=True)
class Edge:
    current_state: Tuple[int, int]
    points: int
    """Points are to be added or subtracted depending on there the edge is placed on the node. 
    - true edge will add points to the total;
    - false edge will subtract points from the maximum reachable."""
    leftovers: List[Tuple[StardewRule, int]]
    """Leftovers are the rules that could not be resolved by the short-circuit evaluation. They will be evaluated afterward."""
    node: Node

    def __str__(self):
        leftovers_points = sum(x[1] for x in self.leftovers)
        return (f"{{{'+' if self.points > 0 else ''}{self.points} -> "
                f"{self.current_state} + {leftovers_points} leftovers"
                f"{' [LEAF]' if self.node.is_leaf else ''}}}")

    def __repr__(self):
        return self.__str__()


class EvaluationTreeStardewRule:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    original: StardewRule
    evaluation_tree: Node

    def __init__(self, original: StardewRule, evaluation_tree: Node):
        self.original = original
        self.evaluation_tree = evaluation_tree

    def __call__(self, state: CollectionState) -> bool:
        current_node = self.evaluation_tree
        while not current_node.is_leaf:
            if current_node.rule(state):
                current_node = current_node.true_edge.node
            else:
                current_node = current_node.false_edge.node

        return current_node.rule(state)

    def __or__(self, other: StardewRule):
        raise NotImplementedError

    def __and__(self, other: StardewRule):
        raise NotImplementedError

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        return repr(self.original)


# TODO try slots, cuz there will be a lot of these. Maybe
@dataclass(frozen=True)
class ShortCircuitScore:
    true: Union[int, float]
    false: Union[int, float]

    def __add__(self, other: ShortCircuitScore) -> ShortCircuitScore:
        # Union
        return ShortCircuitScore(self.true + other.true, self.false + other.false)

    @cached_property
    def min(self):
        return min(self.true, self.false)

    @cached_property
    def total(self) -> Union[int, float]:
        return self.true + self.false


@singledispatch
def to_rule_map(
        rule: StardewRule,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    if _rule_map is None:
        _rule_map = nx.DiGraph()
    elif rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
        return _rule_map

    _rule_map.add_node(rule, priority=0, score=_score)

    return _rule_map


@to_rule_map.register
def _(
        rule: And,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    root = False
    if _rule_map is None:
        _rule_map = nx.DiGraph()
        _combinable_rules = {}
        root = True

    if rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
    else:
        _rule_map.add_node(rule, priority=0, score=_score)

    subrules = rule.original_rules
    propagated_score = ShortCircuitScore(_score.min / len(subrules), _score.false)
    for subrule in subrules:
        to_rule_map(subrule, _rule_map, propagated_score, _combinable_rules)
        _rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NEGATIVE)

    if root:
        _propagate_combinable_scores(_rule_map, _combinable_rules)

    return _rule_map


@to_rule_map.register
def _(
        rule: Or,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    root = False
    if _rule_map is None:
        _rule_map = nx.DiGraph()
        _combinable_rules = {}
        root = True

    if rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
    else:
        _rule_map.add_node(rule, priority=0, score=_score)

    subrules = rule.original_rules
    propagated_score = ShortCircuitScore(_score.true, _score.min / len(subrules))
    for subrule in subrules:
        to_rule_map(subrule, _rule_map, propagated_score, _combinable_rules)
        _rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.POSITIVE)

    if root:
        _propagate_combinable_scores(_rule_map, _combinable_rules)

    return _rule_map


@to_rule_map.register
def _(
        rule: Received,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    if _rule_map is None:
        _rule_map = nx.DiGraph()
    elif rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
        return _rule_map

    _rule_map.add_node(rule, priority=5, score=_score)

    if _combinable_rules is not None:
        _combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return _rule_map


@to_rule_map.register
def _(
        rule: HasProgressionPercent,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    if _rule_map is None:
        _rule_map = nx.DiGraph()
    elif rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
        return _rule_map

    _rule_map.add_node(rule, priority=4, score=_score)

    if _combinable_rules is not None:
        _combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return _rule_map


@to_rule_map.register
def _(
        rule: Reach,
        _rule_map: nx.DiGraph = None,
        _score=ShortCircuitScore(1, 1),
        _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None
) -> nx.DiGraph:
    if _rule_map is None:
        _rule_map = nx.DiGraph()
    elif rule in _rule_map.nodes:
        _rule_map.nodes[rule]["score"] += _score
        return _rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 1.
    _rule_map.add_node(rule, priority=1, score=_score)

    return _rule_map


def _propagate_combinable_scores(rule_map: nx.DiGraph, combinable_rules: Dict[Hashable, Set[CombinableStardewRule]]):
    for rules in combinable_rules.values():
        original_scores_by_rule = [(rule, rule_map.nodes[rule]["score"]) for rule in rules]

        for i, (rule, score) in enumerate(original_scores_by_rule):
            for other_rule, other_score in original_scores_by_rule[i + 1:]:

                if rule.value > other_rule.value:
                    rule_map.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.POSITIVE)
                    rule_map.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.NEGATIVE)
                    rule_map.nodes[rule]["score"] += ShortCircuitScore(other_score.true, 0)
                    rule_map.nodes[other_rule]["score"] += ShortCircuitScore(0, score.false)
                else:
                    rule_map.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.NEGATIVE)
                    rule_map.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.POSITIVE)
                    rule_map.nodes[rule]["score"] += ShortCircuitScore(0, other_score.false)
                    rule_map.nodes[other_rule]["score"] += ShortCircuitScore(score.true, 0)
