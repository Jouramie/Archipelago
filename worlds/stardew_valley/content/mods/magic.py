from ..game_content import ContentPack
from ..mod_registry import register_mod_content_pack
from ...data.skill import Skill
from ...mods.mod_names import Mod
from ...strings.skill_names import ModSkill

register_mod_content_pack(ContentPack(
    Mod.magic,
    skills=(Skill(name=ModSkill.magic, has_mastery=False),)
))
