from ..game_content import ContentPack, StardewContent
from ...data import fish_data
from ...data.game_item import GenericSource, ItemTag, Tag
from ...data.harvest import HarvestCropSource
from ...data.hats_data import Hats
from ...data.requirement import DangerousMinesRequirement, CraftedItemsRequirement
from ...data.shop import HatMouseSource, TailoringSource
from ...strings.content_pack_names import ContentPack as ContentPackNames
from ...strings.crop_names import Fruit
from ...strings.metal_names import MetalBar
from ...strings.region_names import Region
from ...strings.seed_names import Seed


class QiBoardContentPack(ContentPack):
    def harvest_source_hook(self, content: StardewContent):
        content.untag_item(Seed.qi_bean, ItemTag.CROPSANITY_SEED)


qi_board_content_pack = QiBoardContentPack(
    ContentPackNames.qi_board,
    dependencies=(
        ContentPackNames.pelican_town,
        ContentPackNames.ginger_island,
    ),
    harvest_sources={
        # This one is a bit special, because it's only available during the special order, but it can be found from like, everywhere.
        Seed.qi_bean: (GenericSource(regions=(Region.qi_walnut_room,)),),
        Fruit.qi_fruit: (HarvestCropSource(seed=Seed.qi_bean),),
    },
    fishes=(
        fish_data.ms_angler,
        fish_data.son_of_crimsonfish,
        fish_data.glacierfish_jr,
        fish_data.legend_ii,
        fish_data.radioactive_carp,
    ),
    hat_sources={
        Hats.space_helmet: (HatMouseSource(price=20000, unlock_requirements=(DangerousMinesRequirement(120),)),),
        Hats.qi_mask: (Tag(ItemTag.HAT), TailoringSource(tailoring_items=(Fruit.qi_fruit,)),),
        Hats.radioactive_goggles: (Tag(ItemTag.HAT), TailoringSource(tailoring_items=(MetalBar.radioactive,)),),
        Hats.gnomes_cap: (Tag(ItemTag.HAT), HatMouseSource(price=1000, unlock_requirements=(CraftedItemsRequirement(9999),)),),
    },
)
