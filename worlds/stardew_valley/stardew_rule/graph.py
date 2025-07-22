from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import cached_property, singledispatch
from typing import Tuple, Union, Dict, Hashable, Set, List, cast, Collection

import networkx as nx

from BaseClasses import CollectionState
from .assumption import AssumptionState
from .base import ShortCircuitPropagation, CombinableStardewRule, Or, And, BaseStardewRule
from .count import Count
from .literal import LiteralStardewRule, true_, false_
from .protocol import StardewRule
from .state import Received, Reach, HasProgressionPercent, TotalReceived, CombinableReach

logger = logging.getLogger(__name__)


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
class OptimizedStardewRule:
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
        # A score of more than 1 is overkill, it's better to evaluate a more balanced rule.
        return min(self.true, 1) + min(self.false, 1)

    def is_significant(self):
        return self.true >= 0.1 or self.false >= 0.1


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
        score: ShortCircuitScore,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    """Converts a rule to a graph representation.
    priority: rules with higher priority are to evaluate first. Priority is assigned per rule type, based on how long it takes to evaluate it.
    propagation: which result are propagated between rules. Short circuit goes from the starting node to the ending node.
    scores: the percentage of the rule that will be resolved when this rule is evaluated.
    """
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    rule_map.add_node(rule, priority=1, score=score)

    return rule_map


@_recursive_to_rule_map.register
def _(rule: LiteralStardewRule, rule_map: nx.DiGraph, score, *_) -> nx.DiGraph:
    rule_map.add_node(rule, priority=9, score=score)
    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: And,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, root=root)

    subrules = rule.original_rules
    # I checked empirically that using min here reduce the average depth of the tree, which should on average reduce the evaluation time.
    propagated_score = ShortCircuitScore(score.min / len(subrules), score.false)

    if not propagated_score.is_significant():
        return rule_map

    for subrule in subrules:
        # TODO simplify subrule while adding in state, with the assumption that other subrules did not short-circuit
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NEGATIVE)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Or,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, root=root)

    subrules = rule.original_rules
    propagated_score = ShortCircuitScore(score.true, score.min / len(subrules))

    if not propagated_score.is_significant():
        return rule_map

    for subrule in subrules:
        _recursive_to_rule_map(subrule, rule_map, propagated_score, combinable_rules)
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.POSITIVE)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Count,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        root: bool = False
) -> nx.DiGraph:
    if rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
    else:
        rule_map.add_node(rule, priority=0, score=score, root=root)

    subrules = rule.rules_and_points
    rules_count = rule.total
    goal_to_true = rule.count
    goal_to_false = rules_count - goal_to_true + 1
    for subrule, points in subrules:
        true_ratio = points / goal_to_true
        false_ratio = points / goal_to_false
        rule_propagated_score = ShortCircuitScore(score.true * true_ratio, score.false * false_ratio)

        if not rule_propagated_score.is_significant():
            continue

        _recursive_to_rule_map(subrule, rule_map, rule_propagated_score, combinable_rules)
        # TODO do we really use propagation?
        rule_map.add_edge(subrule, rule, propagation=ShortCircuitPropagation.NONE)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Received,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
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
        score: ShortCircuitScore,
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
        score: ShortCircuitScore,
        *_
) -> nx.DiGraph:
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    rule_map.add_node(rule, priority=4, score=score)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: CombinableReach,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
        combinable_rules: Dict[Hashable, Set[CombinableStardewRule]],
        *_
) -> nx.DiGraph:
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 2.
    rule_map.add_node(rule, priority=2, score=score)

    if combinable_rules is not None:
        combinable_rules.setdefault(rule.combination_key, set()).add(rule)

    return rule_map


@_recursive_to_rule_map.register
def _(
        rule: Reach,
        rule_map: nx.DiGraph,
        score: ShortCircuitScore,
        *_
) -> nx.DiGraph:
    if rule_map is None:
        rule_map = nx.DiGraph()
    elif rule in rule_map.nodes:
        rule_map.nodes[rule]["score"] += score
        return rule_map

    # Reach can trigger cache refresh, which takes time... So, it's better to avoid it, hence the priority at 2.
    rule_map.add_node(rule, priority=2, score=score)

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
    true_evaluation_tree = _recursive_create_evaluation_tree(root, true_assumption_state)
    true_edge = Edge.simple_edge(true_evaluation_tree)

    return Node(true_edge, false_edge, most_significant_rule)


def to_optimized_v1(rule: StardewRule) -> StardewRule | OptimizedStardewRule:
    # TODO allow Count, multiply score by weight of rule
    # TODO do something to reduce Reach(location) into access_rule + region, since it access_rule won't be optimized with current rule.
    if not isinstance(rule, (And, Or, Count)):
        return rule
    rule = cast(BaseStardewRule, rule)

    evaluation_tree = to_evaluation_tree(rule)
    return OptimizedStardewRule(rule, evaluation_tree)


def create_optimized_count(rules: Collection[StardewRule], count: int) -> OptimizedStardewRule:
    return to_optimized_v1(Count(rules, count))


def to_optimized_v2(rule: StardewRule) -> StardewRule | CompressedStardewRule:
    """Compress the evaluation tree to reduce the number of branches. It makes it easier to real, will most likely be used for display purposes."""
    if not isinstance(rule, (And, Or, Count)):
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
