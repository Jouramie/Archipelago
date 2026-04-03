from ....data.artisan import MachineSource
from ....mods.mod_names import Mod
from ....strings.artisan_good_names import ArtisanGood
from ....strings.crop_names import Fruit
from ....strings.machine_names import Machine
from ....test.content import SVContentPackTestBase


class TestArtisanEquipment(SVContentPackTestBase):
    mods = (Mod.deepwoods,)

    def test_mango_wine_exists(self):
        self.assertIn(MachineSource(item=Fruit.mango, machine=Machine.keg), self.content.game_items[ArtisanGood.specific_wine(Fruit.mango)].sources)
        self.assertIn(MachineSource(item=Fruit.mango, machine=Machine.keg), self.content.game_items[ArtisanGood.wine].sources)
