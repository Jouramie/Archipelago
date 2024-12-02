import unittest

from ...stardew_rule.assumption import WorldMap


class TestWorldMap(unittest.TestCase):

    def test_given_a_floor_is_available_when_find_accessible_spots_then_upper_floors_are_assumed_available(self):
        world_map = WorldMap()
        expected_available_floors = list(reversed([
            ("Region", "The Mines - Floor 5"),
            ("Region", "The Mines - Floor 10"),
            ("Region", "The Mines - Floor 15"),
            ("Region", "The Mines - Floor 20"),
        ]))

        available_floors = world_map.find_accessible_spots_given_available("Region", "The Mines - Floor 25")

        self.assertEqual(expected_available_floors, available_floors)

    def test_given_a_floor_is_unavailable_when_find_inaccessible_spots_then_lower_floors_are_assumed_unavailable(self):
        world_map = WorldMap()
        expected_available_floors = [
            ("Region", "The Mines - Floor 100"),
            ("Region", "The Mines - Floor 105"),
            ("Region", "The Mines - Floor 110"),
            ("Region", "The Mines - Floor 115"),
            ("Region", "The Mines - Floor 120"),
        ]

        available_floors = world_map.find_inaccessible_spots_given_unavailable("Region", "The Mines - Floor 95")

        self.assertEqual(expected_available_floors, available_floors)
