from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Optional, Tuple, List, Union, Dict, Hashable, Set

import networkx as nx

from BaseClasses import CollectionState
from .base import BaseStardewRule, And, Or, ShortCircuitPropagation, CombinableStardewRule
from .protocol import StardewRule
from .state import HasProgressionPercent, Received, Reach


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
        pass

    def __and__(self, other: StardewRule):
        pass

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        return repr(self.original)


# TODO try slots, cuz there will be a lot of these. Maybe
@dataclass(frozen=True)
class ShortCircuitScore:
    true: Union[int, float]
    false: Union[int, float]

    def __or__(self, other: ShortCircuitScore) -> ShortCircuitScore:
        # Union
        # TODO find a way to use min so scores don't need to be floats
        return ShortCircuitScore(max(self.true, other.true), max(self.false, other.false))

    def total(self) -> Union[int, float]:
        return self.true + self.false


@singledispatch
def to_rule_map(rule: StardewRule, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    """
    if _graph is None:
        _graph = nx.DiGraph()

    _graph.add_node(rule, priority=0)

    return _graph


@to_rule_map.register
def _(rule: And, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None):
    if _graph is None:
        _graph = nx.DiGraph()
    if _combinable_rules is None:
        _combinable_rules = {}
    _graph.add_node(rule, priority=0)

    for subrule in rule.original_rules:
        to_rule_map(subrule, _graph, _combinable_rules)
        _graph.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NEGATIVE)

    return _graph


@to_rule_map.register
def _(rule: Or, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None):
    if _graph is None:
        _graph = nx.DiGraph()
    if _combinable_rules is None:
        _combinable_rules = {}

    _graph.add_node(rule, priority=0)

    for subrule in rule.original_rules:
        to_rule_map(subrule, _graph, _combinable_rules)
        _graph.add_edge(subrule, rule, propagation=ShortCircuitPropagation.POSITIVE)

    return _graph


@to_rule_map.register
def _(rule: Received, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None):
    if _graph is None:
        _graph = nx.DiGraph()
    if _combinable_rules is None:
        _combinable_rules = {}

    _graph.add_node(rule, priority=5)
    _link_combinable_rule(rule, _graph, _combinable_rules)

    return _graph


@to_rule_map.register
def _(rule: HasProgressionPercent, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None):
    if _graph is None:
        _graph = nx.DiGraph()
    if _combinable_rules is None:
        _combinable_rules = {}

    _graph.add_node(rule, priority=4)
    _link_combinable_rule(rule, _graph, _combinable_rules)

    return _graph


@to_rule_map.register
def _(rule: Reach, _graph: nx.DiGraph = None, _combinable_rules: Dict[Hashable, Set[CombinableStardewRule]] = None):
    if _graph is None:
        _graph = nx.DiGraph()

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 1.
    _graph.add_node(rule, priority=1)

    return _graph


def _link_combinable_rule(rule: CombinableStardewRule, graph: nx.DiGraph, combinable_rules: Dict[Hashable, Set[CombinableStardewRule]]):
    other_rules_with_same_key = combinable_rules.get(rule.combination_key, set())

    if rule in other_rules_with_same_key:
        return

    for other_rule in other_rules_with_same_key:
        if rule.value > other_rule.value:
            graph.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.POSITIVE)
            graph.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.NEGATIVE)
        else:
            graph.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.NEGATIVE)
            graph.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.POSITIVE)

    combinable_rules[rule.combination_key] = other_rules_with_same_key | {rule}
