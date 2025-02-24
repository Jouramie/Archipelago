from .item_data import ItemData, Group
from .. import StardewContent
from ..content.vanilla.ginger_island import ginger_island_content_pack


def filter_deprecated_items(items: list[ItemData]) -> list[ItemData]:
    return [item for item in items if Group.DEPRECATED not in item.groups]


def filter_ginger_island_items(exclude_island: bool, items: list[ItemData]) -> list[ItemData]:
    return [item for item in items if not exclude_island or Group.GINGER_ISLAND not in item.groups]


def filter_mod_items(mods: set[str], items: list[ItemData]) -> list[ItemData]:
    return [item for item in items if item.mod_name is None or item.mod_name in mods]


def remove_excluded_items(items, content: StardewContent):
    exclude_ginger_island = ginger_island_content_pack.name not in content.registered_packs
    return remove_excluded_items_island_mods(items, exclude_ginger_island, content.registered_packs)


def remove_excluded_items_island_mods(items, exclude_ginger_island: bool, mods: set[str]):
    deprecated_filter = filter_deprecated_items(items)
    ginger_island_filter = filter_ginger_island_items(exclude_ginger_island, deprecated_filter)
    mod_filter = filter_mod_items(mods, ginger_island_filter)
    return mod_filter
