from ..game_content import ContentPack
from ..mod_registry import register_mod_content_pack
from ..override import override
from ...data import villagers_data
from ...mods.mod_names import Mod

register_mod_content_pack(ContentPack(
    Mod.jasper,
    villagers=(
        villagers_data.jasper,
        override(villagers_data.gunther, mod_name=Mod.jasper),
        override(villagers_data.marlon, mod_name=Mod.jasper),
    )
))
