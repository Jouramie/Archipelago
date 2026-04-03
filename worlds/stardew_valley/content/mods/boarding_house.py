from ..game_content import ContentPack
from ..mod_registry import register_mod_content_pack
from ...data import villagers_data
from ...mods.mod_names import Mod

register_mod_content_pack(ContentPack(
    Mod.boarding_house,
    villagers=(
        villagers_data.gregory,
        villagers_data.sheila,
        villagers_data.joel,
    )
))
