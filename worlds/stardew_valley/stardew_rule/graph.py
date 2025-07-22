from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Optional, Tuple, Union, Dict, Hashable, Set, List, cast

import networkx as nx

from BaseClasses import CollectionState
from . import LiteralStardewRule
from .base import ShortCircuitPropagation, CombinableStardewRule, Or, And, AssumptionState, BaseStardewRule
from .protocol import StardewRule
from .state import Received, Reach, HasProgressionPercent, TotalReceived

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Node:
    true_edge: Optional[Edge]
    false_edge: Optional[Edge]
    rule: StardewRule

    @staticmethod
    def leaf(rule: StardewRule):
        return Node(None, None, rule)

    @cached_property
    def is_leaf(self):
        return self.true_edge is self.false_edge is None

    def __str__(self, depth=0):
        if self.is_leaf:
            return f"{self.rule}"

        padding = '  ' * depth
        return (f"{self.rule} \n"
                f"{padding} T {self.true_edge.__str__(depth=depth + 1)}\n"
                f"{padding} F {self.false_edge.__str__(depth=depth + 1)}")

    def __repr__(self):
        return self.__str__()


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

    @staticmethod
    def simple_edge(node: Node):
        return Edge((0, 0), 0, [], node)

    def __str__(self, depth: int = 0):
        leftovers_points = sum(x[1] for x in self.leftovers)
        return (f"{{{'+' if self.points > 0 else ''}{self.points} -> {self.current_state} + {leftovers_points} leftovers"
                f" {self.node.__str__(depth=depth + 1)}}}")

    def __repr__(self):
        return self.__str__()


class OptimizedStardewRule:
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


def to_rule_map(rule: StardewRule) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    combinable_rules = {}
    rule_map = _recursive_to_rule_map(rule, nx.DiGraph(), ShortCircuitScore(1, 1), combinable_rules, True)
    _propagate_combinable_scores(rule_map, combinable_rules)
    return rule_map


@singledispatch
def _recursive_to_rule_map(
        rule: StardewRule,
        rule_map: nx.DiGraph,
        score,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    raise NotImplementedError(f"Rule {rule} is not supported.")


@_recursive_to_rule_map.register
def _(rule: LiteralStardewRule, rule_map: nx.DiGraph, score, *_) -> nx.DiGraph:
    rule_map.add_node(rule, priority=9, score=score)
    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: And,
        rule_map: nx.DiGraph,
        score,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, root=root)

    subrules = rule.original_rules
    propagated_score = ShortCircuitScore(score.min / len(subrules), score.false)
    for subrule in subrules:
        # TODO simplify subrule while adding in state, with the assumption that other subrules did not short-circuit
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NEGATIVE)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Or,
        rule_map: nx.DiGraph,
        score,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, root=root)

    subrules = rule.original_rules
    propagated_score = ShortCircuitScore(score.true, score.min / len(subrules))
    for subrule in subrules:
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.POSITIVE)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Received,
        rule_map: nx.DiGraph,
        score,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    rule_map.add_node(rule, priority=5, score=score)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: HasProgressionPercent,
        rule_map: nx.DiGraph,
        score,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    rule_map.add_node(rule, priority=4, score=score)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: TotalReceived,
        rule_map: nx.DiGraph,
        score,
        *_
) -> nx.DiGraph:
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    rule_map.add_node(rule, priority=2, score=score)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Reach,
        rule_map: nx.DiGraph,
        score,
        *_
) -> nx.DiGraph:
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 1.
    rule_map.add_node(rule, priority=1, score=score)

    return rule_map


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


def to_evaluation_tree(root: BaseStardewRule) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.
    :returns: The root node.
    """

    return _recursive_create_evaluation_tree(root, AssumptionState())


def _recursive_create_evaluation_tree(root: BaseStardewRule, assumption_state: AssumptionState) -> Node:
    root = root.deep_simplify_knowing(assumption_state)
    rule_map = to_rule_map(root)
    if rule_map.number_of_nodes() == 1:
        return Node.leaf(root)

    most_significant_rule: BaseStardewRule = max((node for node in rule_map.nodes.items() if node[1]["priority"] != 0),
                                                 key=lambda x: (x[1]["score"].total, x[1]["priority"]))[0]

    false_assumption_state = most_significant_rule.add_upper_bounds(assumption_state)
    false_evaluation_tree = _recursive_create_evaluation_tree(root, false_assumption_state)
    false_edge = Edge.simple_edge(false_evaluation_tree)

    true_assumption_state = most_significant_rule.add_lower_bounds(assumption_state)
    true_evaluation_tree = _recursive_create_evaluation_tree(root, true_assumption_state)
    true_edge = Edge.simple_edge(true_evaluation_tree)

    return Node(true_edge, false_edge, most_significant_rule)


def to_optimized(rule: StardewRule) -> StardewRule:
    # TODO allow Count, multiply score by weight of rule
    if not isinstance(rule, (And, Or)):
        return rule
    rule = cast(BaseStardewRule, rule)

    evaluation_tree = to_evaluation_tree(rule)
    return OptimizedStardewRule(rule, evaluation_tree)
