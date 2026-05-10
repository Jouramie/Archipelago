import logging
from itertools import chain
from typing import TYPE_CHECKING, Callable

from BaseClasses import DEFAULT_COLLECTION_RULE, CollectionState, Region, Entrance, Location
from Options import PlandoConnection
from .connections import RANDOMIZED_CONNECTIONS
from .portals import REGION_ORDER, SHOP_POINTS, CHECKPOINTS, PORTALS
from .rules import MessengerHardRules
from .transitions import TRANSITIONS

if TYPE_CHECKING:
    from . import MessengerWorld

logger = logging.getLogger(__name__)

REVERSED_RANDOMIZED_CONNECTIONS = {v: k for k, v in RANDOMIZED_CONNECTIONS.items()}
GLITCHED_ITEM = "Glitched Item"

ONE_WAY_EXITS = {
    "Glacial Peak - Left exit",
    "Artificer's Challenge",
    "Dark Cave - Left exit",
    "Elemental Skylands - Right exit"
}

ONE_WAY_ENTRANCES = {
    "Elemental Skylands - Air Shmup",
    "Tower of Time - Left",
    "Riviere Turquoise - Right",
    "Glacial Peak - Left"
}

COUPLED_EXITS = {
    one_way + " exit": other_way + " exit"
    for one_way, other_way in RANDOMIZED_CONNECTIONS.items()
    if one_way != "Tower HQ"
    if one_way != "Artificer"
    if one_way + " exit" not in ONE_WAY_EXITS
}
COUPLED_EXITS["Artificer's Portal"] = "Corrupted Future Portal"


def handle_auto_tabbing(data: str) -> int:
    match data:
        case "Level_01_NinjaVillage":
            return 1
        case "Level_02_AutumnHills":
            return 2
        case "Level_03_ForlornTemple":
            return 3
        case "Level_04_Catacombs":
            return 4
        case "Level_04_C_RiviereTurquoise":
            return 13
        case "Level_05_A_HowlingGrotto":
            return 6
        case "Level_05_B_SunkenShrine":
            return 14
        case "Level_06_A_BambooCreek":
            return 5
        case "Level_07_QuillshroomMarsh":
            return 7
        case "Level_08_SearingCrags":
            return 8
        case "Level_09_A_GlacialPeak":
            return 9
        case "Level_09_B_ElementalSkylands":
            return 15
        case "Level_10_A_TowerOfTime":
            return 10
        case "Level_11_A_CloudRuins":
            return 11
        case "Level_12_UnderWorld":
            return 12
        case _:
            return 0


TRACKER_PACK_CONFIG = {
    "external_pack_key": "ut_pack_path",
    "map_page_folder": "tracker",
    "map_page_maps": "maps/maps.json",
    "map_page_locations": [
        "locations/AutumnHills.json", "locations/BambooCreek.json", "locations/Catacombs.json", "locations/CloudRuins.json",
        "locations/CorruptedFuture.json", "locations/DarkCave.json", "locations/ElementalSkylands.json", "locations/ForlornTemple.json",
        "locations/GlacialPeak.json", "locations/HowlingGrotto.json", "locations/MusicBox.json", "locations/NinjaVillage.json",
        "locations/QuillshroomMarsh.json", "locations/RiviereTurquoise.json", "locations/SearingCrags.json", "locations/SunkenShrine.json",
        "locations/TheShop.json", "locations/TowerOfTime.json", "locations/Underworld.json"
    ],
    "map_page_setting_key": "Slot:{player}:CurrentRegion",
    "map_page_index": handle_auto_tabbing,
}


def find_spot(portal_key: int) -> str:
    """finds the spot associated with the portal key"""
    parent = REGION_ORDER[portal_key // 100]
    if portal_key % 100 == 0:
        return f"{parent} Portal"
    if portal_key % 100 // 10 == 1:
        return SHOP_POINTS[parent][portal_key % 10]
    return CHECKPOINTS[parent][portal_key % 10]


def reverse_portal_exits_into_portal_plando(portal_exits: list[int]) -> list[PlandoConnection]:
    return [
        PlandoConnection("Autumn Hills", find_spot(portal_exits[0]), "both"),
        PlandoConnection("Riviere Turquoise", find_spot(portal_exits[1]), "both"),
        PlandoConnection("Howling Grotto", find_spot(portal_exits[2]), "both"),
        PlandoConnection("Sunken Shrine", find_spot(portal_exits[3]), "both"),
        PlandoConnection("Searing Crags", find_spot(portal_exits[4]), "both"),
        PlandoConnection("Glacial Peak", find_spot(portal_exits[5]), "both"),
    ]


def reverse_transitions_into_plando_connections(transitions: list[list[int]]) -> list[PlandoConnection]:
    plando_connections = []

    for connection in [
        PlandoConnection(REVERSED_RANDOMIZED_CONNECTIONS[TRANSITIONS[transition[0]]], TRANSITIONS[transition[1]], "both")
        for transition in transitions
    ]:
        if connection.exit in {con.entrance for con in plando_connections}:
            continue
        plando_connections.append(connection)

    return plando_connections


def add_glitched_rules(world: "MessengerWorld", hard_logic: MessengerHardRules) -> None:
    multiworld = world.multiworld

    for entrance in multiworld.get_entrances(world.player):

        try:
            rule = hard_logic.connection_rules[entrance.name]
        except KeyError:
            rule = DEFAULT_COLLECTION_RULE

        if entrance.access_rule == rule:
            continue

        def glitch_aware_rule(state: CollectionState, glitched_rule=rule, previous_rule=entrance.access_rule) -> bool:
            return (state.has(GLITCHED_ITEM, world.player) and glitched_rule(state)) or previous_rule(state)

        entrance.access_rule = glitch_aware_rule

    for loc in multiworld.get_locations(world.player):
        try:
            rule = hard_logic.location_rules[loc.name]
        except KeyError:
            rule = DEFAULT_COLLECTION_RULE

        if loc.access_rule == rule:
            continue

        def glitch_aware_rule(state: CollectionState, glitched_rule=rule, previous_rule=loc.access_rule) -> bool:
            return (state.has(GLITCHED_ITEM, world.player) and glitched_rule(state)) or previous_rule(state)

        loc.access_rule = glitch_aware_rule


def _transition_region_to_exit_name(region_name: str) -> str:
    if region_name == "Artificer":
        return "HQ - Artificer's Portal"

    if region_name == "Tower HQ":
        return "HQ - Artificer's Challenge"

    # Maybe add Corrupted Future portal? Does it work decoupled?

    return f"{region_name} exit"


def _create_tracker_transition_exits() -> dict[str, str]:
    transitions = {
        source: _transition_region_to_exit_name(source) for source in
        chain(RANDOMIZED_CONNECTIONS.keys())
    }
    return transitions


def disconnect_deferred_exits(transitions: list[PlandoConnection], get_region: Callable[[str], Region]) -> dict[str, str]:
    """Disconnect the exits, but save their destinations in a map to reconnect when it is visited."""

    deferred_connections = {}

    def disconnect_exit(transition: Entrance, tracker_name_override: str):
        transition.name = tracker_name_override
        deferred_connections[tracker_name_override] = transition.connected_region.name
        transition.connected_region.entrances.remove(transition)
        transition.connected_region = None

    for region_name, transition_exit in _create_tracker_transition_exits().items():
        if region_name == "Artificer":
            tower = get_region("Tower HQ")
            artificer_portal: Entrance = next(e for e in tower.exits if e.name == "Artificer's Portal")
            disconnect_exit(artificer_portal, "HQ - Artificer's Portal")
            continue

        try:
            region = get_region(region_name)
        except KeyError:
            logger.warning(f"Unable to find region {region_name} for transition exit {transition_exit}, skipping.")
            continue

        real_connection = next((transition for transition in transitions if transition.entrance == region_name), None)
        if real_connection is not None:
            actual_exit_name = f"{real_connection.entrance} -> {real_connection.exit}"
        else:
            real_connection = next((con for con in transitions if con.exit == region_name), None)
            if real_connection is None:
                continue
            actual_exit_name = f"{real_connection.exit} -> {real_connection.entrance}"

        try:
            actual_exit: Entrance = next(e for e in region.exits if e.name == actual_exit_name)
        except StopIteration:
            logger.warning(f"Unable to find exit {actual_exit_name} in region {region_name} for transition exit {transition_exit}, skipping.")
            continue

        disconnect_exit(actual_exit, transition_exit)

    tower = get_region("Tower HQ")
    for portal in PORTALS:
        actual_exit: Entrance = next(e for e in tower.exits if e.name == f"ToTHQ {portal} Portal")
        disconnect_exit(actual_exit, "HQ - " + portal + " Portal")

        # FIXME those do not really work since they're events and not hooked on actual datastorage...
        unlock_region = get_region(portal + " - Portal")
        unlock_event: Location = next(e for e in unlock_region.locations if e.name == f"{portal} Portal")
        unlock_event.name = portal + " - Portal unlock"

    return deferred_connections


def connect_visited_entrances(connections_map: dict[str, str], get_region: Callable[[str], Region], get_entrance: Callable[[str], Entrance],
                              visited_exits: list[str], decoupled: bool = False) -> None:
    def connect_exit_to_destination(visited_exit: str) -> Region:
        transition = get_entrance(visited_exit)
        _destination = get_region(connections_map[visited_exit])
        transition.connect(_destination)
        return _destination

    for e in visited_exits:
        try:
            destination = connect_exit_to_destination(e)
            if not decoupled:
                coupled_exit = _transition_region_to_exit_name(destination.name)
                connect_exit_to_destination(coupled_exit)
        except KeyError:
            logger.warning(f"Unable to find region/entrance for visited exit {e}, skipping connection.")
            continue
