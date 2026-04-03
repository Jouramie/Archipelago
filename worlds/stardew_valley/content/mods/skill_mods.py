from ..game_content import ContentPack
from ..mod_registry import register_mod_content_pack
from ...data.skill import Skill
from ...mods.mod_names import Mod
from ...strings.skill_names import ModSkill

register_mod_content_pack(ContentPack(
    Mod.luck_skill,
    skills=(Skill(name=ModSkill.luck, has_mastery=False),)
))

register_mod_content_pack(ContentPack(
    Mod.socializing_skill,
    skills=(Skill(name=ModSkill.socializing, has_mastery=False),)
))

register_mod_content_pack(ContentPack(
    Mod.cooking_skill,
    skills=(Skill(name=ModSkill.cooking, has_mastery=False),)
))

register_mod_content_pack(ContentPack(
    Mod.binning_skill,
    skills=(Skill(name=ModSkill.binning, has_mastery=False),)
))
