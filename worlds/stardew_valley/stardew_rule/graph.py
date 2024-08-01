from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from functools import cached_property, singledispatch
from typing import Optional, Tuple, Union, Dict, Hashable, Set, List, cast, Collection

import networkx as nx

from BaseClasses import CollectionState
from .base import ShortCircuitPropagation, CombinableStardewRule, Or, And, AssumptionState, BaseStardewRule, CanShortCircuit
from .count import Count
from .literal import LiteralStardewRule, false_, true_
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
    rule_map = to_rule_map(root)
    if rule_map.number_of_nodes() == 1:
        return Node.leaf(root)
    root = cast(BaseStardewRule, root)

    most_significant_rule, attrs = max((node for node in rule_map.nodes.items() if node[1]["priority"] != 0),
                                       key=lambda x: (x[1]["score"].total, x[1]["priority"]))

    false_assumption_state = most_significant_rule.add_upper_bounds(assumption_state)
    false_evaluation_tree = _recursive_create_evaluation_tree(root, false_assumption_state)
    false_edge = Edge.simple_edge(false_evaluation_tree)

    true_assumption_state = most_significant_rule.add_lower_bounds(assumption_state)
    true_evaluation_tree = _recursive_create_evaluation_tree(root, true_assumption_state)
    true_edge = Edge.simple_edge(true_evaluation_tree)

    return Node(true_edge, false_edge, most_significant_rule)


def to_optimized(rule: StardewRule) -> Union[StardewRule, OptimizedStardewRule]:
    # TODO allow Count, multiply score by weight of rule
    # TODO do something to reduce Reach(location) into access_rule + region, since it access_rule won't be optimized with current rule.
    if not isinstance(rule, (And, Or, Count)):
        return rule
    rule = cast(BaseStardewRule, rule)

    evaluation_tree = to_evaluation_tree(rule)
    return OptimizedStardewRule(rule, evaluation_tree)


def create_evaluation_tree(full_rule_graph: nx.DiGraph,
                           weights: Dict[CanShortCircuit, int],
                           rules: Dict[Optional[CanShortCircuit], Counter],
                           count: int,
                           current_state: Tuple[int, int],
                           starting_leftovers: List[Tuple[StardewRule, int]],
                           _simplification_state: AssumptionState = AssumptionState()) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.

    TODO New algo to create evaluation tree
    Find the node that can short-circuit the most rules. Sum true short-circuit rules and false short-circuit rules.
    If decomposable (And and Or), add to killed nodes and


    :returns: The root node.
    """

    if current_state[0] >= count:
        return Node.leaf(true_)
    if current_state[1] < count:
        return Node.leaf(false_)
    if not full_rule_graph:
        assert current_state[0] + sum(points for _, points in starting_leftovers) == current_state[1], "Looks like we lost some leftovers."
        return Node.leaf(Count.from_leftovers(starting_leftovers, count - current_state[0]))

    components = list(nx.weakly_connected_components(full_rule_graph))
    main_component = max(components, key=lambda x: sum(weights[i] for i in x))
    components.remove(main_component)

    main_rule_graph: nx.DiGraph = full_rule_graph.subgraph(main_component)

    # TODO handle weights ? -> Keep all nodes with equal links so center see the other rules. Add weight on equal link depending on point.
    #  Add node matching exact rule to true_, so weight of that node can be considered.
    #  Build graph by calling simplify_knowing() on each equal node.
    #  Then we could evaluate one state subrule at the time. Prioritize received rule.
    center_rule: BaseStardewRule = nx.center(main_rule_graph)[0]

    # FALSE branch

    false_killed_nodes = [center_rule]
    for _, short_circuited_node in nx.generic_bfs_edges(main_rule_graph,
                                                        center_rule,
                                                        neighbors=lambda x: (v
                                                                             for u, v, d in main_rule_graph.out_edges(x, data=True)
                                                                             if d["propagation"] is False)):
        false_killed_nodes.append(short_circuited_node)
    false_simplification_state = center_rule.add_upper_bounds(_simplification_state)

    false_surviving_graph: nx.DiGraph = full_rule_graph.copy()
    false_surviving_graph.remove_nodes_from(false_killed_nodes)

    false_weight = 0
    false_leftovers = list(starting_leftovers)
    for short_circuited in false_killed_nodes:
        for leftover, weight in rules[short_circuited].items():
            simplified = leftover.simplify_knowing(false_simplification_state)

            if simplified is false_:
                false_weight += weight
                continue

            false_leftovers.append((simplified, weight))
    false_state = (current_state[0], current_state[1] - false_weight)

    false_evaluation_tree = create_evaluation_tree(false_surviving_graph, weights, rules, count, false_state, false_leftovers, false_simplification_state)
    false_edge = Edge(false_state, -false_weight, false_leftovers, false_evaluation_tree)

    # TRUE branch

    true_killed_nodes = [center_rule]
    for _, short_circuited_node in nx.generic_bfs_edges(main_rule_graph,
                                                        center_rule,
                                                        neighbors=lambda x: (v
                                                                             for u, v, d in main_rule_graph.out_edges(x, data=True)
                                                                             if d["propagation"] is True)):
        true_killed_nodes.append(short_circuited_node)
    true_simplification_state = center_rule.add_lower_bounds(_simplification_state)

    true_surviving_graph: nx.DiGraph = full_rule_graph.copy()
    true_surviving_graph.remove_nodes_from(true_killed_nodes)

    true_weight = 0
    true_leftovers = list(starting_leftovers)
    for short_circuited in true_killed_nodes:
        for leftover, weight in rules[short_circuited].items():
            simplified = leftover.simplify_knowing(true_simplification_state)

            if simplified is true_:
                true_weight += weight
                continue

            true_leftovers.append((simplified, weight))
    true_state = (current_state[0] + true_weight, current_state[1])

    true_evaluation_tree = create_evaluation_tree(true_surviving_graph, weights, rules, count, true_state, true_leftovers, true_simplification_state)
    true_edge = Edge(true_state, true_weight, true_leftovers, true_evaluation_tree)

    return Node(true_edge, false_edge, center_rule)


def create_optimized_count(rules: Collection[StardewRule], count: int) -> OptimizedStardewRule:
    return to_optimized(Count(rules, count))


def create_special_count(rules: Collection[StardewRule], count: int) -> Union[SpecialCount, Count]:
    """
    TODO New algorithm to link rules together
    Put all rules in a graph
    Put all rules in a queue
    For each rule A in the queue
      For each rule B in the graph where there is no link with A
        Calculate common part -> { A fully included in B, B fully included in A, A and B are disjoint, A and B intersect as C }
        Add link between A and B
        If C is not empty
          Add link between A and C
          Add link between B and C
          Push C in queue
    Prune disjoint links
    Prune non final decomposable rules
    Transitive reduction
    """

    short_circuit_links = cast(Collection[CanShortCircuit], rules)

    grouped_by_component = {}
    for rule in short_circuit_links:
        grouped_by_component.setdefault(rule.short_circuit_able_component, Counter()).update((rule,))

    link_results = {}
    short_circuit_able_keys: List[CanShortCircuit] = list(grouped_by_component.keys())
    for i, ri in enumerate(short_circuit_able_keys):
        if ri is None:
            continue

        for rj in short_circuit_able_keys[i + 1:]:
            if rj is None:
                continue

            link_results[ri, rj] = ri.calculate_short_circuit_propagation(rj)

    nodes = set()
    edges = []
    for link, direction in link_results.items():
        nodes.update(link)

        if direction is ShortCircuitPropagation.POSITIVE:
            edges.append((link[1], link[0]))
        elif direction is ShortCircuitPropagation.NEGATIVE:
            edges.append(link)

    starting_graph = nx.DiGraph()
    starting_graph.add_nodes_from(nodes)
    starting_graph.add_edges_from(edges)
    reduced_graph: nx.DiGraph = nx.transitive_reduction(starting_graph)

    final_graph = nx.DiGraph()
    final_graph.add_nodes_from(reduced_graph.nodes)
    for u, v in reduced_graph.edges:
        final_graph.add_edge(u, v, propagation=False)
        final_graph.add_edge(v, u, propagation=True)

    weights = {rule: sum(counter.values()) for rule, counter in grouped_by_component.items()}
    starting_state = (0, sum(weights.values()))
    starting_leftovers = [(rule, points) for rule, points in grouped_by_component.get(None, {}).items()]
    evaluation_tree = create_evaluation_tree(final_graph, weights, grouped_by_component, count, starting_state, starting_leftovers)

    return SpecialCount(grouped_by_component, evaluation_tree, count)


class SpecialCount(BaseStardewRule):
    count: int
    rules_and_points: List[Tuple[StardewRule, int]]
    evaluation_tree: Node

    def __init__(self, rules: Dict[CanShortCircuit, Counter], evaluation_tree: Node, count: int):
        self.count = count
        self.rules_and_points = sorted([(rule, value) for counter in rules.values() for rule, value in counter.items()], key=lambda x: x[1], reverse=True)
        self.evaluation_tree = evaluation_tree

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate_with_shortcircuit(state)

    def evaluate_with_shortcircuit(self, state: CollectionState) -> bool:
        current_node = self.evaluation_tree
        while not current_node.is_leaf:
            if current_node.rule(state):
                current_node = current_node.true_edge.node
            else:
                current_node = current_node.false_edge.node

        return current_node.rule(state)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    @cached_property
    def rules_count(self):
        return len(self.rules)

    def __repr__(self):
        return f"Received {self.count} [{', '.join(f'{value}x {repr(rule)}' for rule, value in self.rules_and_points)}]"

    @cached_property
    def rules(self):
        return [rule for rule, _ in self.rules_and_points]
