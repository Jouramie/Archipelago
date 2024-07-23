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


# TODO make sure it's faster with slots
@dataclass(frozen=True)
class Node:
    true_edge: Edge
    false_edge: Edge
    rule: BaseStardewRule


@dataclass(frozen=True)
class Edge:
    points: int
    """Points are to be added or subtracted depending on there the edge is placed on the node. 
    - true edge will add points to the total;
    - false edge will subtract points from the maximum reachable."""
    leftovers: List[Tuple[StardewRule, int]]
    """Leftovers are the rules that could not be resolved by the short-circuit evaluation. They will be evaluated afterward."""
    node: Node

    def __str__(self):
        return f"{{{self.points} + {self.leftovers}}}"

    def __repr__(self):
        return self.__str__()


EVALUATION_END_SENTINEL = Node(None, None, None)


def create_evaluation_tree(full_rule_graph: nx.DiGraph,
                           weights: Dict[CanShortCircuitLink, int],
                           rules: Dict[Optional[CanShortCircuitLink], Counter],
                           _simplification_state: AssumptionState = AssumptionState()) -> Node:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules, as long as the leftovers are evaluated afterward.

    :returns: The root node.
    """
    if not full_rule_graph:
        return EVALUATION_END_SENTINEL

    components = list(nx.weakly_connected_components(full_rule_graph))
    main_component = max(components, key=lambda x: sum(weights[i] for i in x))
    components.remove(main_component)

    # Rules that do not have short-circuit-able part are added to the leftovers directly.
    starting_leftovers = [(rule, points) for rule, points in rules.get(None, Counter()).items()]
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
    false_leftovers = sorted(false_leftovers, key=lambda x: x[1])

    # TODO knowing current amount of points from the assumption taken up to this point,
    #  we could literally calculate the result of the rule if we meet the count. We already calculate every state possible anyways.
    #  When creating the leaf, put leftovers in legacy count in rules, or a literal.
    false_evaluation_tree = create_evaluation_tree(false_surviving_graph, weights, rules, false_simplification_state)
    false_edge = Edge(false_weight, false_leftovers, false_evaluation_tree)

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
    true_leftovers = sorted(true_leftovers, key=lambda x: x[1])

    true_evaluation_tree = create_evaluation_tree(true_surviving_graph, weights, rules, true_simplification_state)
    true_edge = Edge(true_weight, true_leftovers, true_evaluation_tree)

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
    evaluation_tree = create_evaluation_tree(final_graph, weights, grouped_by_component)

    return SpecialCount(grouped_by_component, evaluation_tree, count)


class SpecialCount(BaseStardewRule):
    count: int
    rules: Dict[CanShortCircuitLink, Counter]
    evaluation_tree: Node

    weight: Dict[CanShortCircuitLink, int]
    total: int
    simplify_rule_mapping: Dict[StardewRule, StardewRule]

    def __init__(self, rules: Dict[CanShortCircuitLink, Counter], evaluation_tree: Node, count: int):
        self.count = count
        self.rules = rules
        self.evaluation_tree = evaluation_tree

        self.weight = {rule: sum(counter.values()) for rule, counter in rules.items()}
        self.total = sum(sum(counter.values()) for _, counter in rules.items())
        self.simplify_rule_mapping = {}

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate_with_shortcircuit(state)

    def evaluate_with_shortcircuit(self, state: CollectionState) -> bool:
        target_points = self.count

        leftovers: List[Tuple[StardewRule, int]] = []
        min_points = 0
        max_points = self.total

        # Do a first pass without evaluating all the rules completely, just the short-circuit part.
        current_node = self.evaluation_tree
        while current_node != EVALUATION_END_SENTINEL:
            evaluation = current_node.rule(state)

            if evaluation:
                edge = current_node.true_edge
                min_points += edge.points
                if min_points >= target_points:
                    return True

            else:
                edge = current_node.false_edge
                max_points -= edge.points
                if max_points < target_points:
                    return False

            leftovers.extend(edge.leftovers)
            current_node = edge.node

        return self.evaluate_leftovers(state, min_points, max_points, leftovers)

    def evaluate_leftovers(self,
                           state: CollectionState,
                           min_points: int,
                           max_points: int,
                           leftovers: List[Tuple[StardewRule, int]]) -> bool:
        """
        Do a second pass to evaluate the rules that were not short-circuited.
        """

        target_points = self.count

        for rule, points in leftovers:
            evaluation = self.call_evaluate_while_simplifying_cached(rule, state)
            if evaluation:
                min_points += points
                if min_points >= target_points:
                    return True
            else:
                max_points -= points
                if max_points < target_points:
                    return False

        assert min_points == max_points, "Seems like some rules were not evaluated correctly."
        return False

    def call_evaluate_while_simplifying_cached(self, rule: StardewRule, state: CollectionState) -> bool:
        try:
            # A mapping table with the original rule is used here because two rules could resolve to the same rule.
            #  This would require to change the counter to merge both rules, and quickly become complicated.
            return self.simplify_rule_mapping[rule](state)
        except KeyError:
            self.simplify_rule_mapping[rule], value = rule.evaluate_while_simplifying(state)
            return value

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self, self(state)

    @cached_property
    def rules_count(self):
        return len(self.rules)

    def __repr__(self):
        return f"Received {self.count} [{', '.join(f'{value}x {repr(rule)}' for counter in self.rules.values() for rule, value in counter.items())}]"


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
