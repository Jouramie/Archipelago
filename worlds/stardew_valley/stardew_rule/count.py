from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import cached_property
from typing import List, Callable, Optional, Dict, Tuple, Collection, Union, cast

import networkx as nx

from BaseClasses import CollectionState
from .base import BaseStardewRule, CanShortCircuitLink, ShortCircuitPropagation, AssumptionState
from .literal import false_, true_
from .protocol import StardewRule


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


def create_evaluation_tree(full_rule_graph: nx.DiGraph,
                           weights: Dict[CanShortCircuitLink, int],
                           rules: Dict[Optional[CanShortCircuitLink], Counter],
                           count: int,
                           current_state: Tuple[int, int],
                           starting_leftovers: List[Tuple[StardewRule, int]],
                           _simplification_state: AssumptionState = AssumptionState()) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.

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
    center_rule = nx.center(main_rule_graph)[0]

    # FALSE branch

    false_killed_nodes = [center_rule]
    for _, short_circuited_node in nx.generic_bfs_edges(main_rule_graph,
                                                        center_rule,
                                                        neighbors=lambda x: (v
                                                                             for u, v, d in main_rule_graph.out_edges(x, data=True)
                                                                             if d["propagation"] is False)):
        false_killed_nodes.append(short_circuited_node)
    false_simplification_state = _simplification_state.add_upper_bounds(center_rule)

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
    true_simplification_state = _simplification_state.add_lower_bounds(center_rule)

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


def create_special_count(rules: Collection[StardewRule], count: int) -> Union[SpecialCount, Count]:
    short_circuit_links = cast(Collection[CanShortCircuitLink], rules)

    grouped_by_component = {}
    for rule in short_circuit_links:
        grouped_by_component.setdefault(rule.short_circuit_able_component, Counter()).update((rule,))

    link_results = {}
    short_circuit_able_keys = list(grouped_by_component.keys())
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

    total: int

    def __init__(self, rules: Dict[CanShortCircuitLink, Counter], evaluation_tree: Node, count: int):
        self.count = count
        self.rules_and_points = sorted([(rule, value) for counter in rules.values() for rule, value in counter.items()], key=lambda x: x[1], reverse=True)
        self.evaluation_tree = evaluation_tree

        self.total = sum(sum(counter.values()) for _, counter in rules.items())

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


class Count(BaseStardewRule):
    count: int
    rules: List[StardewRule]
    rules_and_points: List[Tuple[StardewRule, int]]
    evaluate: Callable[[CollectionState], bool]

    total: Optional[int]
    rule_mapping: Optional[Dict[StardewRule, StardewRule]]

    def __init__(self, rules: List[StardewRule], count: int, _rules_and_points: Optional[List[Tuple[StardewRule, int]]] = None):
        self.count = count
        if _rules_and_points is not None:
            self.rules_and_points = _rules_and_points
        self.rules_and_points = sorted(Counter(rules).items(), key=lambda x: x[1], reverse=True)
        self.total = sum(point for _, point in self.rules_and_points)
        self.rules = [rule for rule, _ in self.rules_and_points]

    @staticmethod
    def from_leftovers(leftovers: List[Tuple[StardewRule, int]], count: int):
        # FIXME this is really suboptimal... I should remove the other implementation anyways
        rules = [rule for rule, value in leftovers for _ in range(value)]
        return Count(rules, count, _rules_and_points=leftovers)

    def __call__(self, state: CollectionState) -> bool:
        min_points = 0
        max_points = self.total
        goal = self.count
        rules = self.rules

        for i, (_, point) in enumerate(self.rules_and_points):
            rules[i], evaluation_value = rules[i].evaluate_while_simplifying(state)

            if evaluation_value:
                min_points += point
                if min_points >= goal:
                    return True
            else:
                max_points -= point
                if max_points < goal:
                    return False

        assert False, "Should have returned before."

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    @cached_property
    def rules_count(self):
        return len(self.rules)

    def __repr__(self):
        return f"Received {self.count} [{', '.join(f'{value}x {repr(rule)}' for rule, value in self.rules_and_points)}]"
