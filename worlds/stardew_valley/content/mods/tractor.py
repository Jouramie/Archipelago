from ..game_content import ContentPack
from ..mod_registry import register_mod_content_pack
from ...data.building import Building
from ...data.shop import ShopSource
from ...mods.mod_names import Mod
from ...strings.artisan_good_names import ArtisanGood
from ...strings.building_names import ModBuilding
from ...strings.metal_names import MetalBar
from ...strings.region_names import Region

register_mod_content_pack(ContentPack(
    Mod.tractor,
    farm_buildings=(
        Building(
            ModBuilding.tractor_garage,
            sources=(
                ShopSource(
                    shop_region=Region.carpenter,
                    price=150_000,
                    items_price=((20, MetalBar.iron), (5, MetalBar.iridium), (1, ArtisanGood.battery_pack)),
                ),
            ),
        ),
    ),
))
