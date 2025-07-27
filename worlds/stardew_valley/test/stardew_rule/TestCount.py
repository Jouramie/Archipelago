import unittest
from typing import cast
from unittest.mock import MagicMock, Mock

from ...stardew_rule import StardewRule, Received, Count, Reach, create_optimized_count, Has


class TestCount(unittest.TestCase):

    def test_duplicate_rule_count_double(self):
        expected_result = True
        collection_state = MagicMock()
        simplified_rule = Mock()
        other_rule = Mock(spec=StardewRule)
        other_rule.evaluate_while_simplifying = Mock(return_value=(simplified_rule, expected_result))
        rule = Count([cast(StardewRule, other_rule), other_rule, other_rule], 2)

        actual_result = rule(collection_state)

        other_rule.evaluate_while_simplifying.assert_called_once_with(collection_state)
        self.assertEqual(expected_result, actual_result)

    def test_simplified_rule_is_reused(self):
        expected_result = False
        collection_state = MagicMock()
        simplified_rule = Mock()
        simplified_rule.evaluate_while_simplifying = Mock(return_value=(simplified_rule, expected_result))
        other_rule = Mock(spec=StardewRule)
        other_rule.evaluate_while_simplifying = Mock(return_value=(simplified_rule, expected_result))
        rule = Count([cast(StardewRule, other_rule), cast(StardewRule, other_rule), cast(StardewRule, other_rule)], 2)

        actual_result = rule(collection_state)

        other_rule.evaluate_while_simplifying.assert_called_once_with(collection_state)
        self.assertEqual(expected_result, actual_result)

        other_rule.evaluate_while_simplifying.reset_mock()

        actual_result = rule(collection_state)

        other_rule.evaluate_while_simplifying.assert_not_called()
        simplified_rule.evaluate_while_simplifying.assert_called()
        self.assertEqual(expected_result, actual_result)

    def test_break_if_not_enough_rule_to_complete(self):
        expected_result = False
        collection_state = MagicMock()
        simplified_rule = Mock()
        never_called_rule = Mock()
        other_rule = Mock(spec=StardewRule)
        other_rule.evaluate_while_simplifying = Mock(return_value=(simplified_rule, expected_result))
        rule = Count([cast(StardewRule, other_rule)] * 4, 2)

        actual_result = rule(collection_state)

        other_rule.evaluate_while_simplifying.assert_called_once_with(collection_state)
        never_called_rule.assert_not_called()
        never_called_rule.evaluate_while_simplifying.assert_not_called()
        self.assertEqual(expected_result, actual_result)


class TestSuperCount(unittest.TestCase):

    def test_can_count_dependent_rules(self):
        collection_state = Mock()
        special_count = create_optimized_count([
            Received("Potato", 1, 1),
            Received("Potato", 1, 2),
            Received("Potato", 1, 3),
        ], 2)

        collection_state.has = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)

        collection_state.has = Mock(return_value=True)
        self.assertTrue(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)

    def test_can_count_dependent_and_rules(self):
        collection_state = Mock()
        special_count = create_optimized_count([
            Received("Potato", 1, 1) & Reach("Potato Field", "Location", 1),
            Received("Potato", 1, 2),
            Received("Potato", 1, 3) & Reach("Potato Field", "Location", 1),
        ], 2)

        collection_state.has = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)

        collection_state.has = Mock(return_value=True)
        collection_state.can_reach = Mock(return_value=True)
        self.assertTrue(special_count(collection_state))
        self.assertEqual(2, collection_state.has.call_count)  # FIXME could be lowered to 1 by really removing short circuited rules, not just adding score
        self.assertEqual(1, collection_state.can_reach.call_count)

    def test_given_two_disconnected_received_when_evaluate_then_evaluate_received_before_reach(self):
        collection_state = Mock()
        special_count = create_optimized_count([
            Received("Carrot", 1, 2),
            Received("Carrot", 1, 2),
            Received("Carrot", 1, 2),
            Received("Potato", 1, 2),
            Received("Potato", 1, 3) & Reach("Potato Field", "Location", 1),
        ], 3)

        collection_state.has = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)  # Carrot

    def test_given_or_rules_when_evaluate_then_evaluate_or_before_reach(self):
        collection_state = Mock()
        special_count = create_optimized_count([
            Received("Carrot", 1, 2) | Reach("Carrot Field", "Location", 1),
            Received("Carrot", 1, 2) | Reach("Carrot Field", "Location", 1),
            Received("Potato", 1, 3) & Reach("Potato Field", "Location", 1),
        ], 2)

        collection_state.has = Mock(return_value=False)
        collection_state.can_reach = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)
        self.assertEqual(1, collection_state.can_reach.call_count)

        collection_state.has = Mock(side_effect=lambda x, y, z: (x, y, z) in {("Carrot", 1, 2)})
        self.assertTrue(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)

    def test_given_two_rules_with_common_part_when_evaluate_then_evaluate_common_part_first(self):
        collection_state = Mock()
        special_count = create_optimized_count([
            Received("Potato", 1, 1) & Received("Brocoli", 1, 1),
            Received("Brocoli", 1, 1) & Received("Carrot", 1, 2),
            Received("Brocoli", 1, 3),
            Received("Brocoli", 1, 2) & Received("Carrot", 1, 1),
            Received("Carrot", 1, 1)
        ], 2)

        collection_state.has = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(2, collection_state.has.call_count)

        collection_state.has = Mock(side_effect=lambda x, y, z: (x, y, z) in {("Carrot", 1, 1), ("Brocoli", 1, 1), ("Brocoli", 1, 2)})
        self.assertTrue(special_count(collection_state))
        self.assertEqual(2, collection_state.has.call_count)

    def test_given_two_rules_with_has_when_evaluate_then_has_is_broken_down(self):
        collection_state = Mock()
        rules = {
            "Potato": Received("Progressive Seed", 1, 1) & Reach("Potato field", "Location", 1),
            "Carrot": Received("Progressive Seed", 1, 2) & Reach("Carrot field", "Location", 1),
            "Brocoli": Received("Progressive Seed", 1, 3) & Reach("Brocoli field", "Location", 1),
        }
        special_count = create_optimized_count([
            Has("Potato", rules),
            Has("Carrot", rules),
            Has("Brocoli", rules),
        ], 2)

        collection_state.has = Mock(return_value=False)
        self.assertFalse(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)

        collection_state.has = Mock(side_effect=lambda x, y, z: (x, y, z) in {("Progressive Seed", 1, 1), ("Progressive Seed", 1, 2)})
        collection_state.can_reach = Mock(return_value=True)
        self.assertTrue(special_count(collection_state))
        self.assertEqual(1, collection_state.has.call_count)
        self.assertEqual(2, collection_state.can_reach.call_count)
