from .. import SVContentPackTestBase
from ....mods.mod_names import Mod
from ....strings.villager_names import ModNPC


class TestJasperWithoutSVE(SVContentPackTestBase):
    mods = (Mod.jasper,)

    def test_gunther_is_added(self):
        self.assertIn(ModNPC.gunther, self.content.villagers)
        self.assertEqual(self.content.villagers[ModNPC.gunther].mod_name, Mod.jasper)

    def test_marlon_is_added(self):
        self.assertIn(ModNPC.marlon, self.content.villagers)
        self.assertEqual(self.content.villagers[ModNPC.marlon].mod_name, Mod.jasper)


class TestJasperWithSVE(SVContentPackTestBase):
    mods = (Mod.jasper, Mod.sve)

    def test_gunther_is_added(self):
        self.assertIn(ModNPC.gunther, self.content.villagers)
        self.assertEqual(self.content.villagers[ModNPC.gunther].mod_name, Mod.sve)

    def test_marlon_is_added(self):
        self.assertIn(ModNPC.marlon, self.content.villagers)
        self.assertEqual(self.content.villagers[ModNPC.marlon].mod_name, Mod.sve)
