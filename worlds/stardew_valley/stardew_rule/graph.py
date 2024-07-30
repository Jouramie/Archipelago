from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Optional, Tuple, Union, Dict, Hashable, Set, Deque, List

import networkx as nx

from BaseClasses import CollectionState
from .base import ShortCircuitPropagation, CombinableStardewRule, Or, And, AssumptionState, BaseStardewRule
from .literal import LiteralStardewRule
from .protocol import StardewRule
from .state import HasProgressionPercent, Reach, Received


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

    def __str__(self):
        leftovers_points = sum(x[1] for x in self.leftovers)
        return (f"{{{'+' if self.points > 0 else ''}{self.points} -> "
                f"{self.current_state} + {leftovers_points} leftovers"
                f"{' [LEAF]' if self.node.is_leaf else ''}}}")

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


# TODO split finding root and recursive part
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
        # TODO simplify subrule while adding in state, with the assumption that other subrules did not short-circuit
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


def to_evaluation_tree(rule_map: nx.DiGraph, root: BaseStardewRule) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.

    TODO New algo to create evaluation tree
        Find the node that can short-circuit the most rules. Sum true short-circuit rules and false short-circuit rules.
        If decomposable (And and Or), add to killed nodes and

    TODO handle weights ? -> Keep all nodes with equal links so center see the other rules. Add weight on equal link depending on point.
        Add node matching exact rule to true_, so weight of that node can be considered.
        Build graph by calling simplify_knowing() on each equal node.
        Then we could evaluate one state subrule at the time. Prioritize received rule.

    :returns: The root node.
    """

    ordered_nodes = sorted(rule_map.nodes.items(), key=lambda x: (x[1]["priority"], x[1]["score"].total))
    rules = (node[0] for node in ordered_nodes)

    return _recursive_create_evaluation_tree(deque(rules), root, AssumptionState())


def _recursive_create_evaluation_tree(evaluation_queue: Deque[BaseStardewRule], root: BaseStardewRule, assumption_state: AssumptionState) -> Node:
    if isinstance(root, LiteralStardewRule):
        return Node.leaf(root)

    evaluation_queue = evaluation_queue.copy()
    center_rule = evaluation_queue.pop()
    if not evaluation_queue:
        return Node.leaf(center_rule)

    # FALSE branch
    false_assumption_state = assumption_state.add_upper_bounds(center_rule)
    false_simplified_root = root.deep_simplify_knowing(false_assumption_state)

    false_evaluation_tree = _recursive_create_evaluation_tree(evaluation_queue, false_simplified_root, false_assumption_state)
    false_edge = Edge.simple_edge(false_evaluation_tree)

    # TRUE branch

    true_assumption_state = assumption_state.add_lower_bounds(center_rule)
    true_simplified_root = root.deep_simplify_knowing(true_assumption_state)

    true_evaluation_tree = _recursive_create_evaluation_tree(evaluation_queue, true_simplified_root, true_assumption_state)
    true_edge = Edge.simple_edge(true_evaluation_tree)

    return Node(true_edge, false_edge, center_rule)
