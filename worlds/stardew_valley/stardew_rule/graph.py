from __future__ import annotations

import enum
import logging
import time
import typing
from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property, singledispatch
from typing import Tuple, Dict, Hashable, Set, List, cast, Collection, ClassVar

import networkx as nx

from BaseClasses import CollectionState
from .assumption import AssumptionState
from .base import ShortCircuitPropagation, CombinableStardewRule, Or, And, BaseStardewRule, Has
from .count import Count
from .literal import LiteralStardewRule, true_, false_
from .protocol import StardewRule
from .state import Received, Reach, HasProgressionPercent, TotalReceived, CombinableReach

logger = logging.getLogger(__name__)


class RuleLinkType(enum.Enum):
    COMBINABLE = enum.auto()
    INCLUSION = enum.auto()


@dataclass(frozen=True)
class Node:
    true_edge: Edge | None
    false_edge: Edge | None
    rule: StardewRule

    @staticmethod
    def leaf(rule: StardewRule):
        return Node(None, None, rule)

    @cached_property
    def is_leaf(self):
        return self.true_edge is self.false_edge is None

    @property
    def depth(self):
        if self.is_leaf:
            return 0
        return max(self.true_edge.node.depth, self.false_edge.node.depth) + 1

    def list_leaf_depth(self, _depth=0) -> List[int]:
        if self.is_leaf:
            return [_depth]

        return self.true_edge.node.list_leaf_depth(_depth + 1) + self.false_edge.node.list_leaf_depth(_depth + 1)

    @property
    def average_leaf_depth(self):
        leafs = self.list_leaf_depth()
        return sum(leafs) / len(leafs)

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
    current_state: tuple[int, int]
    points: int
    """Points are to be added or subtracted depending on there the edge is placed on the node. 
    - true edge will add points to the total;
    - false edge will subtract points from the maximum reachable."""
    node: Node

    @staticmethod
    def simple_edge(node: Node):
        return Edge((0, 0), 0, node)

    @staticmethod
    def simple_leaf(rule: StardewRule):
        return Edge((0, 0), 0, Node.leaf(rule))

    def __str__(self, depth: int = 0):
        return (f"{{{'+' if self.points > 0 else ''}{self.points} -> {self.current_state}"
                f" {self.node.__str__(depth=depth + 1)}}}")

    def __repr__(self):
        return self.__str__()


@dataclass(frozen=True)
class TreeStardewRule:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    original: StardewRule
    evaluation_tree: Node

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


@dataclass(frozen=True)
class SequenceCountStardewRule:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    original: list[tuple[StardewRule, int]] = field(repr=False, hash=False, compare=False)
    evaluation_sequence: list[tuple[StardewRule, int, int]]
    count: int
    total: int

    def __call__(self, state: CollectionState) -> bool:
        min_points = 0
        max_points = self.total
        goal = self.count

        for rule, true_points, false_points in self.evaluation_sequence:
            if rule(state):
                min_points += true_points
                if min_points >= goal:
                    return True
            else:
                max_points -= false_points
                if max_points < goal:
                    return False

        assert False, "Should have returned before."

    def __or__(self, other: StardewRule):
        raise NotImplementedError

    def __and__(self, other: StardewRule):
        raise NotImplementedError

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        return repr(Count([], self.count, _rules_and_points=self.original))


@dataclass(frozen=True)
class CompressedStardewRule:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    original: StardewRule = field(repr=False, hash=False, compare=False)
    compressed: StardewRule

    def __call__(self, state: CollectionState) -> bool:
        return self.compressed(state)

    def __or__(self, other: StardewRule):
        raise NotImplementedError

    def __and__(self, other: StardewRule):
        raise NotImplementedError

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)


@dataclass(frozen=True)
class EvaluationTreeStardewRule:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    evaluation_tree: Node

    def __call__(self, state: CollectionState) -> bool:
        current_node = self.evaluation_tree
        while not current_node.is_leaf:
            if current_node.rule(state):
                current_node = current_node.true_edge.node
            else:
                current_node = current_node.false_edge.node

        return current_node.rule(state)

    def __or__(self, other: StardewRule):
        return Or(self, other)

    def __and__(self, other: StardewRule):
        return And(self, other)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)


# TODO try slots, cuz there will be a lot of these. Maybe
@dataclass(frozen=True)
class RuleValue:
    none: ClassVar[RuleValue] = None
    true: int | float
    false: int | float

    def __post_init__(self):
        assert self.true >= 0, "True value must be non-negative."
        assert self.false >= 0, "False value must be non-negative."

    def __add__(self, other: RuleValue) -> RuleValue:
        return RuleValue(self.true + other.true, self.false + other.false)

    def __sub__(self, other: RuleValue) -> RuleValue:
        return RuleValue(self.true - other.true, self.false - other.false)

    # TODO Does caching this improve performance?
    @cached_property
    def min(self):
        return min(self.true, self.false)

    @cached_property
    def total(self) -> int | float:
        # A score of more than 1 is overkill, it's better to evaluate a more balanced rule.
        return min(self.true, 1) + min(self.false, 1)

    @cached_property
    def avg(self) -> float:
        # A score of more than 1 is overkill, it's better to evaluate a more balanced rule.
        # TODO or maybe multiply instead of min?
        return (min(self.true, 1) + min(self.false, 1)) / 2

    @property
    def balance_score(self) -> tuple[int | float, int | float]:
        return self.true + self.false, -abs(self.true - self.false)

    def get(self, value: bool) -> int | float:
        """Returns the value of the rule based on the boolean value."""
        if value:
            return self.true
        return self.false


RuleValue.none = RuleValue(0, 0)


def to_rule_map(rule: StardewRule) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    combinable_rules = {}
    rule_map = _recursive_to_rule_map(rule, nx.DiGraph(), RuleValue(1, 1), combinable_rules, value=RuleValue(1, 1))
    _propagate_combinable_scores(rule_map, combinable_rules)
    return rule_map


@singledispatch
def _recursive_to_rule_map(
        rule: StardewRule,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        /,
        *,
        value: RuleValue = RuleValue.none,
        allowed_depth: int = 0
) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    rule_map.add_node(rule, priority=1, score=score, short_circuit_score=score, value=value)

    return rule_map


@_recursive_to_rule_map.register
def _(rule: LiteralStardewRule, rule_map: nx.DiGraph, score: RuleValue, *_, value: RuleValue = RuleValue.none, **__) -> nx.DiGraph:
    rule_map.add_node(rule, priority=9, score=score, short_circuit_score=score, value=value)
    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Has,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        allowed_depth: int = 0,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
    else:
        rule_map.add_node(rule, priority=1, score=score, short_circuit_score=score, value=value)

    subrule = rule.other_rules.get(rule.item)
    if subrule is None:
        # Subrule is not known at this point, so we can't add it to the rule map.
        # FIXME So I guess simplification process should happen later when all the rules are known?
        return rule_map

    _recursive_to_rule_map(subrule, rule_map, score, combinable_rules, allowed_depth=allowed_depth)
    rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.EQUAL, link=RuleLinkType.INCLUSION)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: And,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        allowed_depth: int = 0,
        **__

) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, short_circuit_score=score, value=value)

    if allowed_depth <= 0:
        return rule_map

    subrules = rule.original_rules
    propagated_score = RuleValue(score.true / len(subrules), score.false)

    for subrule in subrules:
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules, allowed_depth=allowed_depth - 1)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NEGATIVE, link=RuleLinkType.INCLUSION)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Or,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        allowed_depth: int = 0,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, short_circuit_score=score, value=value)

    if allowed_depth <= 0:
        return rule_map

    subrules = rule.original_rules
    propagated_score = RuleValue(score.true, score.false / len(subrules))

    for subrule in subrules:
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules, allowed_depth=allowed_depth - 1)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.POSITIVE, link=RuleLinkType.INCLUSION)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Received,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    rule_map.add_node(rule, priority=5, score=score, short_circuit_score=score, value=value)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: HasProgressionPercent,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    rule_map.add_node(rule, priority=4, score=score, short_circuit_score=score, value=value)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: TotalReceived,
        rule_map: nx.DiGraph,
        score: RuleValue,
        *_,
        value: RuleValue = RuleValue.none,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    rule_map.add_node(rule, priority=4, score=score, short_circuit_score=score, value=value)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: CombinableReach,
        rule_map: nx.DiGraph,
        score: RuleValue,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_,
        value: RuleValue = RuleValue.none,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 2.
    rule_map.add_node(rule, priority=2, score=score, short_circuit_score=score, value=value)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Reach,
        rule_map: nx.DiGraph,
        score: RuleValue,
        *_,
        value: RuleValue = RuleValue.none,
        **__
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["value"] += value
        rule_map.nodes[rule]["score"] += score
        rule_map.nodes[rule]["short_circuit_score"] += score
        return rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 2.
    rule_map.add_node(rule, priority=2, score=score, short_circuit_score=score, value=value)

    return rule_map


def _propagate_combinable_scores(rule_map: nx.DiGraph, combinable_rules: Dict[Hashable, Set[CombinableStardewRule]]):
    for rules in combinable_rules.values():
        original_scores_by_rule = [(rule, rule_map.nodes[rule]["score"]) for rule in rules]

        for i, (rule, score) in enumerate(original_scores_by_rule):
            for other_rule, other_score in original_scores_by_rule[i + 1:]:

                if rule.value > other_rule.value:
                    rule_map.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.POSITIVE, link=RuleLinkType.COMBINABLE)
                    rule_map.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.NEGATIVE, link=RuleLinkType.COMBINABLE)
                    rule_map.nodes[rule]["short_circuit_score"] += RuleValue(other_score.true, 0)
                    rule_map.nodes[other_rule]["short_circuit_score"] += RuleValue(0, score.false)
                else:
                    rule_map.add_edge(rule, other_rule, propagation=ShortCircuitPropagation.NEGATIVE, link=RuleLinkType.COMBINABLE)
                    rule_map.add_edge(other_rule, rule, propagation=ShortCircuitPropagation.POSITIVE, link=RuleLinkType.COMBINABLE)
                    rule_map.nodes[rule]["short_circuit_score"] += RuleValue(0, other_score.false)
                    rule_map.nodes[other_rule]["short_circuit_score"] += RuleValue(score.true, 0)


def to_evaluation_tree(root: BaseStardewRule) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.
    :returns: The root node.
    """

    return _recursive_create_evaluation_tree(root, AssumptionState())


def _recursive_create_evaluation_tree(root: BaseStardewRule, assumption_state: AssumptionState) -> Node:
    root = root.deep_simplify_knowing(assumption_state)
    # TODO maybe reuse use the same map while score is still significant?
    rule_map = to_rule_map(root)
    if rule_map.number_of_nodes() == 1:
        return Node.leaf(root)
    root = cast(BaseStardewRule, root)

    most_significant_rule = None
    while most_significant_rule is None:
        most_significant_rule, attrs = max((node for node in rule_map.nodes.items() if node[1]["priority"] != 0),
                                           key=lambda x: (x[1]["score"].total, x[1]["priority"]),
                                           default=(None, None))

        if most_significant_rule is None:
            # Skipping the first node since it's the root and is by definition not priority.
            iterator = iter(rule_map.nodes.items())
            next(iterator)
            rule_map = to_rule_map(next(iterator)[0])

    false_assumption_state = most_significant_rule.add_upper_bounds(assumption_state)
    false_evaluation_tree = _recursive_create_evaluation_tree(root, false_assumption_state)
    false_edge = Edge.simple_edge(false_evaluation_tree)

    true_assumption_state = most_significant_rule.add_lower_bounds(assumption_state)
    # TODO might be interesting to explore switching nodes in the tree when a lower node removes rule form the count than the parent (1 node removed when not having a recipe is bad compared to 17 node when not having farming level 1)
    #   just sorting the rules to have those with the most combinable first would be a good start.

    # TODO should also try simplifying while evaluating
    true_evaluation_tree = _recursive_create_evaluation_tree(root, true_assumption_state)
    true_edge = Edge.simple_edge(true_evaluation_tree)

    return Node(true_edge, false_edge, most_significant_rule)


def to_optimized_v1(rule: StardewRule) -> StardewRule | TreeStardewRule:
    # TODO allow Count, multiply score by weight of rule
    # TODO do something to reduce Reach(location) into access_rule + region, since it access_rule won't be optimized with current rule.
    if not isinstance(rule, (And, Or, Count, Has)):
        return rule
    rule = cast(BaseStardewRule, rule)

    evaluation_tree = to_evaluation_tree(rule)
    return TreeStardewRule(rule, evaluation_tree)


def to_optimized_v2(rule: StardewRule) -> StardewRule | CompressedStardewRule:
    """Compress the evaluation tree to reduce the number of branches. It makes it easier to real, will most likely be used for display purposes."""
    if not isinstance(rule, (And, Or, Count, Has)):
        return rule
    rule = cast(BaseStardewRule, rule)

    evaluation_tree = to_evaluation_tree(rule)
    compressed = compress_evaluation_tree(evaluation_tree)
    return CompressedStardewRule(rule, compressed)


def compress_evaluation_tree(evaluation_tree: Node) -> StardewRule:
    return _compress_evaluation_tree_recursive(evaluation_tree)


def _compress_evaluation_tree_recursive(evaluation_tree: Node) -> StardewRule:
    if evaluation_tree.is_leaf:
        return evaluation_tree.rule

    if evaluation_tree.true_edge.node.rule == true_:
        return evaluation_tree.rule | _compress_evaluation_tree_recursive(evaluation_tree.false_edge.node)

    compressed_true_branch = _compress_evaluation_tree_recursive(evaluation_tree.true_edge.node)
    if evaluation_tree.false_edge.node.rule == false_:
        return evaluation_tree.rule & compressed_true_branch

    if isinstance(compressed_true_branch, EvaluationTreeStardewRule):
        true_branch = Edge.simple_edge(compressed_true_branch.evaluation_tree)
    else:
        true_branch = Edge.simple_leaf(compressed_true_branch)

    compressed_false_branch = _compress_evaluation_tree_recursive(evaluation_tree.false_edge.node)
    if isinstance(compressed_false_branch, EvaluationTreeStardewRule):
        false_branch = Edge.simple_edge(compressed_false_branch.evaluation_tree)
    else:
        false_branch = Edge.simple_leaf(compressed_false_branch)

    compressed_sub_tree = Node(true_branch, false_branch, evaluation_tree.rule)
    return EvaluationTreeStardewRule(compressed_sub_tree)


def create_count_rule_map(rules_and_points: Collection[tuple[StardewRule, int]], count: int, depth=1) -> nx.DiGraph:
    combinable_rules = {}
    rule_map = nx.DiGraph()

    rules_count = sum(r[1] for r in rules_and_points)
    goal_to_true = count
    goal_to_false = rules_count - goal_to_true + 1
    for subrule, points in rules_and_points:
        true_ratio = points / goal_to_true
        false_ratio = points / goal_to_false
        score = RuleValue(true_ratio, false_ratio)

        _recursive_to_rule_map(subrule, rule_map, score, combinable_rules, value=RuleValue(points, points), allowed_depth=depth)

    _propagate_combinable_scores(rule_map, combinable_rules)
    return rule_map


def create_count_evaluation_sequence(rule_map: nx.DiGraph) -> list[tuple[StardewRule, int, int]]:
    """Create a sequence of rules to evaluate based on the rule map.
    The sequence is created by traversing the rule map and selecting the most significant rules first.
    FIXME ça marche pas vraiment, ça arrive qu'on claim des pts de True alors que les False sont déjà claimed pour la même règle...
    """
    assert rule_map.number_of_nodes() is not None

    evaluation_sequence = []

    # TODO is there a way to stop adding rules to sequence by knowing we've evaluated enough?
    #  yes, when the values of all rules have been added, even tho they are not evaluated (graph should be empty if that happens).
    #  or maybe of the total cumulative value of all nodes removed from the graph is greater than min/max point, we can stop
    while rule_map.number_of_nodes() != 0:
        current_most_significant_rule, attrs = max(rule_map.nodes.items(), key=lambda x: (x[1]["short_circuit_score"].avg, x[1]["priority"]))
        points_counted: RuleValue = attrs["value"]

        edges_to_explore = list((r, e, True) for r, e in rule_map[current_most_significant_rule].items())
        explored_nodes = set()
        nodes_to_remove = [current_most_significant_rule]
        edges_to_update = []  # TODO fill to recalculate scores of every toughed nodes once they are claimed
        while edges_to_explore:
            linked_rule, edge_attrs, is_direct_edge = edges_to_explore.pop()
            explored_nodes.add(linked_rule)

            node_attrs = rule_map.nodes[linked_rule]
            added_value: RuleValue = node_attrs["value"]
            propagation = edge_attrs["propagation"]

            if propagation == ShortCircuitPropagation.POSITIVE:
                if added_value.true > 0:
                    points_counted = RuleValue(points_counted.true + added_value.false, points_counted.false)
                    added_value = RuleValue(0, added_value.false)
                    node_attrs["value"] = added_value
                    node_attrs["short_circuit_score"] = RuleValue(0, node_attrs["short_circuit_score"].false)

                new_edges = list((r, {**e, "propagation": ShortCircuitPropagation.POSITIVE}, False)
                                 for r, e in rule_map[linked_rule].items()
                                 if e["propagation"] in (ShortCircuitPropagation.POSITIVE, ShortCircuitPropagation.EQUAL)
                                 and r not in explored_nodes)
                edges_to_explore.extend(new_edges)

            elif propagation == ShortCircuitPropagation.NEGATIVE:
                if added_value.false > 0:
                    points_counted = RuleValue(points_counted.true, points_counted.false + added_value.false)
                    added_value = RuleValue(added_value.true, 0)
                    node_attrs["value"] = added_value
                    node_attrs["short_circuit_score"] = RuleValue(node_attrs["short_circuit_score"].true, 0)
                    # TODO probably should also update the score

                new_edges = list((r, {**e, "propagation": ShortCircuitPropagation.NEGATIVE}, False)
                                 for r, e in rule_map[linked_rule].items()
                                 if e["propagation"] in (ShortCircuitPropagation.NEGATIVE, ShortCircuitPropagation.EQUAL)
                                 and r not in explored_nodes)
                edges_to_explore.extend(new_edges)

            elif propagation == ShortCircuitPropagation.EQUAL:
                if added_value.true > 0 or added_value.false > 0:
                    points_counted = RuleValue(points_counted.true + added_value.true, points_counted.false + added_value.false)
                    added_value = RuleValue(0, 0)
                    node_attrs["value"] = added_value
                    node_attrs["short_circuit_score"] = RuleValue(0, 0)

                new_edges = list((r, e, is_direct_edge)
                                 for r, e in rule_map[linked_rule].items()
                                 if e["propagation"] in (ShortCircuitPropagation.NEGATIVE, ShortCircuitPropagation.POSITIVE, ShortCircuitPropagation.EQUAL)
                                 and r not in explored_nodes)
                edges_to_explore.extend(new_edges)
            else:
                raise ValueError(f"Unknown propagation type {propagation} for rule {linked_rule}.")

            if is_direct_edge and edge_attrs["link"] == RuleLinkType.INCLUSION:
                if len(rule_map.in_edges(linked_rule)) == 1:
                    points_counted = points_counted + added_value
                    nodes_to_remove.append(linked_rule)

        evaluation_sequence.append((current_most_significant_rule, points_counted.true, points_counted.false))
        for node in nodes_to_remove:
            removed_score = rule_map.nodes[node]["score"]
            for short_circuiter, _ in rule_map.in_edges(node):
                short_circuit_score = rule_map.nodes[short_circuiter]["short_circuit_score"]
                short_circuit_propagation = rule_map.edges[short_circuiter, node]["propagation"]

                if short_circuit_propagation == ShortCircuitPropagation.POSITIVE:
                    new_score = RuleValue(short_circuit_score.true - removed_score.true, short_circuit_score.false)
                elif short_circuit_propagation == ShortCircuitPropagation.NEGATIVE:
                    new_score = RuleValue(short_circuit_score.true, short_circuit_score.false - removed_score.false)
                elif short_circuit_propagation == ShortCircuitPropagation.EQUAL:
                    new_score = RuleValue(short_circuit_score.true - removed_score.true, short_circuit_score.false - removed_score.false)
                else:
                    raise ValueError(f"Unknown propagation type {short_circuit_propagation} for rule {short_circuiter}.")

                rule_map.nodes[short_circuiter]["short_circuit_score"] = new_score
        rule_map.remove_nodes_from(nodes_to_remove)

    return evaluation_sequence


def create_optimized_count_v3(rules: Collection[StardewRule], count: int):
    rules_and_points = sorted(Counter(rules).items(), key=lambda x: x[1], reverse=True)

    start = time.perf_counter_ns()
    rule_map = create_count_rule_map(rules_and_points, count, depth=2)
    print(f"Size is {rule_map.number_of_nodes()} nodes, {rule_map.number_of_edges()} edges.")

    if rule_map.number_of_edges() == 0:
        print("No edges in the rule map, returning unsimplified count.")
        return Count(rules, count, _rules_and_points=rules_and_points)

    node_edge_ratio = rule_map.number_of_nodes() / rule_map.number_of_edges()
    if node_edge_ratio > 1:
        print(f"Node to edge ratio is {node_edge_ratio:.2f}. Simplification will probably be a waste of time.")
        return Count(rules, count, _rules_and_points=rules_and_points)

    sequence = create_count_evaluation_sequence(rule_map)
    end = time.perf_counter_ns()
    print(f"Creating count of depth {2} took {(end - start) / 1_000_000:.2f} ms. ")

    assert sum(t for _, t, _ in sequence) == len(rules) and sum(f for _, _, f in sequence) == len(rules)

    return SequenceCountStardewRule(rules_and_points, sequence, count, len(rules))


@dataclass
class GrowingNode:
    rule: typing.Union[StardewRule, None] = None

    rule_map: nx.DiGraph | None = None
    attrs: dict[str, typing.Any] | None = None
    true_points: int | None = None
    false_points: int | None = None
    goal: int | None = None

    true_branch: GrowingNode | None = None
    false_branch: GrowingNode | None = None

    @staticmethod
    def leaf(rule: StardewRule) -> GrowingNode:
        return GrowingNode(rule)

    @staticmethod
    def branch(rule_map: nx.DiGraph, true_points: int, false_points: int, goal: int) -> GrowingNode:
        most_significant_rule, attrs = max(rule_map.nodes.items(), key=lambda x: (x[1]["short_circuit_score"].balance_score, x[1]["priority"]))
        return GrowingNode(most_significant_rule, rule_map, attrs, true_points, false_points, goal)

    @cached_property
    def is_leaf(self) -> bool:
        return self.rule_map is self.true_branch is self.false_branch is None

    def replace_true_branch(self, new_branch: GrowingNode) -> None:
        self.true_branch = new_branch

    def replace_false_branch(self, new_branch: GrowingNode) -> None:
        self.false_branch = new_branch

    def grow_true_branch(self) -> GrowingNode:
        if self.false_branch is not None:
            rule_map = self.rule_map
        else:
            rule_map = self.rule_map.copy()

        points = self._collect_points(rule_map, True)
        new_points = self.true_points + points

        if new_points >= self.goal:
            self.true_branch = GrowingNode.leaf(true_)
        else:
            self.true_branch = GrowingNode.branch(rule_map, new_points, self.false_points, self.goal)

        if self.false_branch is not None:
            del self.rule_map
            del self.attrs

        return self.true_branch

    def grow_false_branch(self) -> GrowingNode:
        if self.true_branch is not None:
            rule_map = self.rule_map
        else:
            rule_map = self.rule_map.copy()

        points = self._collect_points(rule_map, False)
        new_points = self.false_points - points

        # TODO maybe switch to normal count when there is no longer an high enough node/edge ratio?
        if new_points < self.goal:
            self.false_branch = GrowingNode.leaf(false_)
        else:
            self.false_branch = GrowingNode.branch(rule_map, self.true_points, new_points, self.goal)

        if self.true_branch is not None:
            del self.rule_map
            del self.attrs

        return self.false_branch

    def _collect_points(self, rule_map: nx.DiGraph, result: bool) -> int:
        rule = self.rule
        attrs = self.attrs

        points_to_add: int = attrs["value"].get(result)
        short_circuited_nodes = {rule}

        # FIXME remove nested functions?
        def explore_short_circuiting_rules():
            nonlocal points_to_add
            if result:
                propagations = (ShortCircuitPropagation.POSITIVE, ShortCircuitPropagation.EQUAL)
            else:
                propagations = (ShortCircuitPropagation.NEGATIVE, ShortCircuitPropagation.EQUAL)

            edges_to_explore = [
                r
                for r, e in rule_map[rule].items()
                if e["propagation"] in propagations
            ]
            explored_nodes = {rule}

            while edges_to_explore:
                linked_rule = edges_to_explore.pop()
                explored_nodes.add(linked_rule)

                short_circuited_nodes.add(linked_rule)
                points_to_add += rule_map.nodes[linked_rule]["value"].get(result)

                edges_to_explore.extend(
                    r
                    for r, e in rule_map[linked_rule].items()
                    if e["propagation"] in propagations
                    and r not in explored_nodes
                )

        explore_short_circuiting_rules()

        # TODO merge the two functions
        def remove_short_circuited_nodes():

            for node in short_circuited_nodes:
                score_to_remove = rule_map.nodes[node]["score"]

                for short_circuiter in rule_map.predecessors(node):
                    if short_circuiter in short_circuited_nodes:
                        continue

                    short_circuit_score = rule_map.nodes[short_circuiter]["short_circuit_score"]
                    short_circuit_propagation = rule_map.edges[short_circuiter, node]["propagation"]

                    if short_circuit_propagation == ShortCircuitPropagation.POSITIVE:
                        new_score = RuleValue(max(0, short_circuit_score.true - score_to_remove.true), short_circuit_score.false)
                    elif short_circuit_propagation == ShortCircuitPropagation.NEGATIVE:
                        new_score = RuleValue(short_circuit_score.true, max(0, short_circuit_score.false - score_to_remove.false))
                    elif short_circuit_propagation == ShortCircuitPropagation.EQUAL:
                        new_score = RuleValue(max(0, short_circuit_score.true - score_to_remove.true),
                                              max(0, short_circuit_score.false - score_to_remove.false))
                    else:
                        raise ValueError(f"Unknown propagation type {short_circuit_propagation} for rule {short_circuiter}.")

                    rule_map.nodes[short_circuiter]["short_circuit_score"] = new_score

                # Recalculate the score of the nodes with included relationships
                impacted_edges = list(rule_map.out_edges(node))
                for subrule, parent in impacted_edges:
                    # FIXME should check if parent has real value, otherwise we might recalculate multiple times the same node.
                    if parent in short_circuited_nodes or rule_map.edges[subrule, parent]["link"] != RuleLinkType.INCLUSION:
                        continue

                    subrules = list(
                        c
                        for c in rule_map.predecessors(parent)
                        # FIXME Only removing subrule will trigger multiple recalculation of multiple childs are to be short-circuited.
                        #   Can happen where two Has link to the same item behind the scene.
                        if c is not subrule
                        and rule_map.edges[c, parent]["link"] == RuleLinkType.INCLUSION
                    )
                    if not subrules:
                        continue

                    score = rule_map.nodes[parent]["score"]
                    if result:
                        assert isinstance(parent, And)
                        propagated_score = RuleValue(score.true / len(subrules), score.false)
                        diff = propagated_score - RuleValue(score.true / (len(subrules) + 1), score.false)
                    else:
                        assert isinstance(parent, Or)
                        propagated_score = RuleValue(score.true, score.false / len(subrules))
                        diff = propagated_score - RuleValue(score.true, score.false / (len(subrules) + 1))

                    for child in subrules:
                        rule_map.nodes[child]["score"] += diff
                        rule_map.nodes[child]["short_circuit_score"] += diff

                    if len(subrules) == 1:
                        rule_map.edges[subrules[0], parent]["propagation"] = ShortCircuitPropagation.EQUAL

                rule_map.remove_node(node)

        # FIXME skip this if we already reached the goal
        remove_short_circuited_nodes()

        return points_to_add


@dataclass(frozen=True)
class GrowingTreeCount:
    """A rule that can be evaluated with an evaluation tree. Should only be used for evaluation."""
    original: list[tuple[StardewRule, int]] = field(repr=False, hash=False, compare=False)
    count: int = field(repr=False, hash=False, compare=False)
    evaluation_tree: GrowingNode

    def __call__(self, state: CollectionState) -> bool:
        current_node = self.evaluation_tree

        while not current_node.is_leaf:
            if current_node.rule(state):
                if current_node.true_branch is None:
                    current_node = current_node.grow_true_branch()
                else:
                    current_node = current_node.true_branch
            else:
                if current_node.false_branch is None:
                    current_node = current_node.grow_false_branch()
                else:
                    current_node = current_node.false_branch

        return current_node.rule(state)

    def __or__(self, other: StardewRule):
        return Or(self, other)

    def __and__(self, other: StardewRule):
        return And(self, other)

    def __hash__(self):
        return id(self)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    def __repr__(self):
        return repr(self.original)


def create_optimized_count(rules: Collection[StardewRule], count: int):
    start = time.perf_counter_ns()

    rules_and_points = sorted(Counter(rules).items(), key=lambda x: x[1], reverse=True)
    rule_map = create_count_rule_map(rules_and_points, count, depth=2)
    print(f"Size is {rule_map.number_of_nodes()} nodes, {rule_map.number_of_edges()} edges.")

    if rule_map.number_of_edges() == 0:
        print("No edges in the rule map, returning unsimplified count.")
        return Count(rules, count, _rules_and_points=rules_and_points)

    node_edge_ratio = rule_map.number_of_nodes() / rule_map.number_of_edges()
    if node_edge_ratio > 1:
        print(f"Node to edge ratio is {node_edge_ratio:.2f}. Simplification will probably be a waste of time.")
        return Count(rules, count, _rules_and_points=rules_and_points)

    rule = GrowingTreeCount(rules_and_points, count, GrowingNode.branch(rule_map, 0, len(rules), count))
    end = time.perf_counter_ns()
    print(f"Creating count of depth {2} took {(end - start) / 1_000_000:.2f} ms. ")

    return rule
