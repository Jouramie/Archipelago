from __future__ import annotations

from collections import Counter
from functools import cached_property
from typing import List, Callable, Optional, Dict, Tuple, Collection, Union, Any, cast

import networkx as nx

from BaseClasses import CollectionState
from .base import BaseStardewRule, CanShortCircuitLink, ShortCircuitPropagation
from .protocol import StardewRule

EVALUATION_END_SENTINEL = "EVALUATION_END_SENTINEL"


def create_evaluation_tree(short_circuit_graph: nx.DiGraph, weights: Dict[CanShortCircuitLink, int], rules: Dict[CanShortCircuitLink, Counter]) -> Tuple[
    nx.DiGraph, Any]:
    """ Precalculate the evaluation tree based on the possible results of each rule (recursively).
    Going from the root to one leaf should evaluate all the rules.

    :returns: First element is the tree, second is the root.
    """
    if not short_circuit_graph:
        tree = nx.DiGraph()
        tree.add_node(EVALUATION_END_SENTINEL)
        return tree, EVALUATION_END_SENTINEL

    evaluation_tree = nx.DiGraph()
    # TODO account for disconnected graphs
    # TODO handle weights ? -> Keep all nodes with equal links so center see the other rules
    center_rule = nx.center(short_circuit_graph)[0]

    false_short_circuited_nodes = [center_rule]
    for _, short_circuited_node in nx.generic_bfs_edges(short_circuit_graph,
                                                        center_rule,
                                                        neighbors=lambda x: (v
                                                                             for u, v, d in short_circuit_graph.out_edges(x, data=True)
                                                                             if d["propagation"] is False)):
        false_short_circuited_nodes.append(short_circuited_node)

    false_surviving_graph = nx.DiGraph(short_circuit_graph)
    false_surviving_graph.remove_nodes_from(false_short_circuited_nodes)
    false_weight = sum(weights[rule] for rule in false_short_circuited_nodes)
    # FIXME this is assuming a AND but will not work with OR. Should add some kind of "resolve knowing" method.
    false_leftovers = []
    false_evaluation_tree, false_evaluation_root = create_evaluation_tree(false_surviving_graph, weights, rules)
    evaluation_tree.update(false_evaluation_tree)
    evaluation_tree.add_edge(center_rule,
                             false_evaluation_root,
                             short_circuit=false_short_circuited_nodes,
                             weight=(false_weight, None),
                             leftovers=(false_leftovers, []))

    true_short_circuited_nodes = [center_rule]
    for _, short_circuited_node in nx.generic_bfs_edges(short_circuit_graph,
                                                        center_rule,
                                                        neighbors=lambda x: (v
                                                                             for u, v, d in short_circuit_graph.out_edges(x, data=True)
                                                                             if d["propagation"] is True)):
        true_short_circuited_nodes.append(short_circuited_node)

    true_surviving_graph = nx.DiGraph(short_circuit_graph)
    true_surviving_graph.remove_nodes_from(true_short_circuited_nodes)
    true_weight = sum(rules[rule].get(rule, 0) for rule in true_short_circuited_nodes)
    true_leftovers = sorted([(leftover, weight)
                             for short_circuited in true_short_circuited_nodes
                             for leftover, weight in rules[short_circuited].items()
                             if leftover != short_circuited],
                            key=lambda x: x[1])
    true_evaluation_tree, true_evaluation_root = create_evaluation_tree(true_surviving_graph, weights, rules)
    evaluation_tree.update(true_evaluation_tree)
    evaluation_tree.add_edge(center_rule,
                             true_evaluation_root,
                             short_circuit=true_short_circuited_nodes,
                             weight=(None, true_weight),
                             leftovers=([], true_leftovers))

    if true_evaluation_root == false_evaluation_root == EVALUATION_END_SENTINEL:
        evaluation_tree.add_edge(center_rule,
                                 EVALUATION_END_SENTINEL,
                                 short_circuit=[center_rule],
                                 weight=(false_weight, true_weight),
                                 leftovers=(false_leftovers, true_leftovers))

    return evaluation_tree, center_rule


def create_special_count(rules: Collection[StardewRule], count: int) -> Union[SpecialCount, Count]:
    short_circuit_links = cast(Collection[CanShortCircuitLink], rules)

    grouped_by_component = {}
    for rule in short_circuit_links:
        grouped_by_component.setdefault(rule.short_circuit_able_component, Counter()).update((rule,))

    link_results = {}
    short_circuit_able_keys = list(grouped_by_component.keys())
    for i, ri in enumerate(short_circuit_able_keys):
        for rj in short_circuit_able_keys[i + 1:]:
            link_results[ri, rj] = ri.calculate_short_circuit_propagation(rj)

    directed_links = []
    for link, direction in link_results.items():
        if direction is ShortCircuitPropagation.POSITIVE:
            directed_links.append((link[1], link[0]))
        elif direction is ShortCircuitPropagation.NEGATIVE:
            directed_links.append(link)

    if not directed_links:
        return Count(list(rules), count)

    g = nx.DiGraph(directed_links)
    tr: nx.DiGraph = nx.transitive_reduction(g)

    final_graph = nx.DiGraph()
    for u, v in tr.edges:
        final_graph.add_edge(u, v, propagation=False)
        final_graph.add_edge(v, u, propagation=True)

    weights = {rule: sum(counter.values()) for rule, counter in grouped_by_component.items()}
    evaluation_tree = create_evaluation_tree(final_graph, weights, grouped_by_component)

    return SpecialCount(grouped_by_component, evaluation_tree, count)


class SpecialCount(BaseStardewRule):
    count: int
    rules: Dict[CanShortCircuitLink, Counter]
    evaluation_tree: Tuple[nx.DiGraph, Union[CanShortCircuitLink, StardewRule]]

    weight: Dict[CanShortCircuitLink, int]
    total: int
    simplify_rule_mapping: Dict[StardewRule, StardewRule]

    def __init__(self, rules: Dict[CanShortCircuitLink, Counter], evaluation_tree: Tuple[nx.DiGraph, CanShortCircuitLink],
                 count: int):
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

        evaluated: List[bool] = []
        leftovers: List[Tuple[StardewRule, int]] = []
        min_points = 0
        max_points = self.total

        # Do a first pass without evaluating all the rules completely, just the short-circuit part.
        tree, current_rule = self.evaluation_tree
        while current_rule != EVALUATION_END_SENTINEL:
            evaluation = current_rule(state)
            weight_index = 1 if evaluation else 0

            current_rule, data = self.__find_next_node(tree, current_rule, weight_index)
            points = data["weight"][weight_index]

            if evaluation:
                min_points += points
                if min_points >= target_points:
                    return True
            else:
                max_points -= points
                if max_points < target_points:
                    return False

            leftovers.extend(data["leftovers"][weight_index])
            evaluated.append(evaluation)

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
        tree, root = self.evaluation_tree

        for rule, points in leftovers:
            if rule == root:
                continue

            evaluation = self.call_evaluate_while_simplifying_cached(rule, state)
            if evaluation:
                min_points += points
                if min_points >= target_points:
                    return True
            else:
                max_points -= points
                if max_points < target_points:
                    return False

        assert min_points == max_points
        return False

    @staticmethod
    def __find_next_node(tree: nx.DiGraph, current_node: CanShortCircuitLink, weight_index: int) -> Tuple[CanShortCircuitLink, Dict]:
        for v, d in tree[current_node].items():
            if d["weight"][weight_index] is not None:
                return v, d

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
