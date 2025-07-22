import unittest

import networkx as nx

from ...stardew_rule import Received, Reach, ShortCircuitPropagation, to_rule_map, ShortCircuitScore, to_evaluation_tree, Node, false_, true_, Edge


class NetworkXAssertMixin(unittest.TestCase):

    def assert_graph_equals(self, expected: nx.Graph, actual: nx.Graph):
        self.assertEqual(set(expected.nodes), set(actual.nodes), "Nodes are different")
        for node in expected.nodes:
            self.assertEqual(expected.nodes[node], actual.nodes[node], f"Node {node} is different")
        self.assertEqual(set(expected.edges), set(actual.edges), "Edges are different")
        for edge in expected.edges:
            self.assertEqual(expected.edges[edge], actual.edges[edge], f"Edge {edge} is different")


class TestToRuleMap(NetworkXAssertMixin, unittest.TestCase):

    def test_given_received_when_convert_to_rule_map_then_single_node_with_high_priority(self):
        rule = Received("Carrot", 1, 1)

        graph = to_rule_map(rule)

        expected = nx.DiGraph()
        expected.add_node(rule, priority=5, score=ShortCircuitScore(1, 1))
        self.assert_graph_equals(expected, graph)

    def test_given_reach_when_convert_to_rule_map_then_single_node_with_resolvable(self):
        rule = Reach("Carrot Field", "Location", 1)

        graph = to_rule_map(rule)

        expected = nx.DiGraph()
        expected.add_node(rule, priority=2, score=ShortCircuitScore(1, 1))
        self.assert_graph_equals(expected, graph)

    def test_given_or_of_duplicated_ands_when_convert_to_rule_map_then_ands_are_merged(self):
        carrot = Received("Carrot", 1, 1)
        potato = Received("Potato", 1, 1)
        and_rule = carrot & potato
        rule = and_rule | and_rule

        graph = to_rule_map(rule)

        expected = nx.DiGraph()
        expected.add_node(carrot, priority=5, score=ShortCircuitScore(1 / 2, 1))
        expected.add_node(potato, priority=5, score=ShortCircuitScore(1 / 2, 1))
        expected.add_node(and_rule, priority=0, score=ShortCircuitScore(2, 1), root=False)
        expected.add_node(rule, priority=0, score=ShortCircuitScore(1, 1), root=True)
        expected.add_edge(carrot, and_rule, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(potato, and_rule, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(and_rule, rule, propagation=ShortCircuitPropagation.POSITIVE)
        self.assert_graph_equals(expected, graph)

    def test_given_and_of_duplicated_ors_when_convert_to_rule_map_then_ors_are_merged(self):
        carrot = Received("Carrot", 1, 1)
        potato = Received("Potato", 1, 1)
        or_rule = carrot | potato
        rule = or_rule & or_rule

        graph = to_rule_map(rule)

        expected = nx.DiGraph()
        expected.add_node(carrot, priority=5, score=ShortCircuitScore(1, 1 / 2))
        expected.add_node(potato, priority=5, score=ShortCircuitScore(1, 1 / 2))
        expected.add_node(or_rule, priority=0, score=ShortCircuitScore(1, 2), root=False)
        expected.add_node(rule, priority=0, score=ShortCircuitScore(1, 1), root=True)
        expected.add_edge(carrot, or_rule, propagation=ShortCircuitPropagation.POSITIVE)
        expected.add_edge(potato, or_rule, propagation=ShortCircuitPropagation.POSITIVE)
        expected.add_edge(or_rule, rule, propagation=ShortCircuitPropagation.NEGATIVE)
        self.assert_graph_equals(expected, graph)

    def test_given_complex_case_of_or_of_ands_with_received_more_than_once_when_convert_to_rule_map_then_graph_is_correct(self):
        received_carrot = Received("Carrot", 1, 1)
        received_two_carrots = Received("Carrot", 1, 2)
        received_potato = Received("Potato", 1, 1)
        reach_kitchen = Reach("Kitchen", "Region", 1)
        kitchen_and_two_carrots = received_two_carrots & reach_kitchen
        carrot_and_potato = received_carrot & received_potato
        rule = received_potato | kitchen_and_two_carrots | carrot_and_potato

        graph = to_rule_map(rule)

        expected = nx.DiGraph()
        expected.add_node(received_carrot, priority=5, score=ShortCircuitScore(1 / 6, 1 / 3 + 1 / 3))
        expected.add_node(received_two_carrots, priority=5, score=ShortCircuitScore(1 / 6 + 1 / 6, 1 / 3))
        expected.add_node(received_potato, priority=5, score=ShortCircuitScore(1 + 1 / 6, 1 / 3 + 1 / 3))
        expected.add_node(reach_kitchen, priority=2, score=ShortCircuitScore(1 / 6, 1 / 3))
        expected.add_node(kitchen_and_two_carrots, priority=0, score=ShortCircuitScore(1, 1 / 3), root=False)
        expected.add_node(carrot_and_potato, priority=0, score=ShortCircuitScore(1, 1 / 3), root=False)
        expected.add_node(rule, priority=0, score=ShortCircuitScore(1, 1), root=True)

        expected.add_edge(received_carrot, carrot_and_potato, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(received_potato, carrot_and_potato, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(received_two_carrots, kitchen_and_two_carrots, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(reach_kitchen, kitchen_and_two_carrots, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(received_potato, rule, propagation=ShortCircuitPropagation.POSITIVE)
        expected.add_edge(kitchen_and_two_carrots, rule, propagation=ShortCircuitPropagation.POSITIVE)
        expected.add_edge(carrot_and_potato, rule, propagation=ShortCircuitPropagation.POSITIVE)

        expected.add_edge(received_carrot, received_two_carrots, propagation=ShortCircuitPropagation.NEGATIVE)
        expected.add_edge(received_two_carrots, received_carrot, propagation=ShortCircuitPropagation.POSITIVE)

        self.assert_graph_equals(expected, graph)


class TestEvaluationTree(unittest.TestCase):

    def test_given_received_when_convert_to_rule_map_then_single_node_with_high_priority(self):
        rule = Received("Carrot", 1, 1)

        evaluation_tree = to_evaluation_tree(rule)

        expected = Node.leaf(rule)
        self.assertEqual(expected, evaluation_tree)

    def test_given_reach_when_convert_to_rule_map_then_single_node_with_resolvable(self):
        rule = Reach("Carrot Field", "Location", 1)

        evaluation_tree = to_evaluation_tree(rule)

        expected = Node.leaf(rule)
        self.assertEqual(expected, evaluation_tree)

    def test_given_or_of_duplicated_ands_when_convert_to_rule_map_then_ands_are_merged(self):
        carrot = Received("Carrot", 1, 1)
        potato = Received("Potato", 1, 1)
        and_rule = carrot & potato
        rule = and_rule | and_rule

        evaluation_tree = to_evaluation_tree(rule)

        self.assertEqual(false_, evaluation_tree.false_edge.node.rule)
        self.assertEqual(false_, evaluation_tree.true_edge.node.false_edge.node.rule)
        self.assertEqual(true_, evaluation_tree.true_edge.node.true_edge.node.rule)

    def test_given_and_of_duplicated_ors_when_convert_to_rule_map_then_ors_are_merged(self):
        carrot = Received("Carrot", 1, 1)
        potato = Received("Potato", 1, 1)
        or_rule = carrot | potato
        rule = or_rule & or_rule

        evaluation_tree = to_evaluation_tree(rule)

        self.assertEqual(true_, evaluation_tree.true_edge.node.rule)
        self.assertEqual(true_, evaluation_tree.false_edge.node.true_edge.node.rule)
        self.assertEqual(false_, evaluation_tree.false_edge.node.false_edge.node.rule)

    def test_given_complex_case_of_or_of_ands_with_received_more_than_once_when_convert_to_rule_map_then_graph_is_correct(self):
        received_carrot = Received("Carrot", 1, 1)
        received_two_carrots = Received("Carrot", 1, 2)
        received_potato = Received("Potato", 1, 1)
        reach_kitchen = Reach("Kitchen", "Region", 1)
        kitchen_and_two_carrots = received_two_carrots & reach_kitchen
        carrot_and_potato = received_carrot & received_potato
        rule = received_potato | kitchen_and_two_carrots | carrot_and_potato

        evaluation_tree = to_evaluation_tree(rule)

        two_carrot_node = Node(Edge.simple_edge(Node.leaf(reach_kitchen)), Edge.simple_edge(Node.leaf(false_)), received_two_carrots)
        potato_node = Node(Edge.simple_edge(Node.leaf(true_)), Edge.simple_edge(two_carrot_node), received_potato)
        self.assertEqual(potato_node, evaluation_tree)
