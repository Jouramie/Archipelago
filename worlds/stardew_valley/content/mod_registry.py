from .game_content import ContentPack

FAKE_CONTENT_PACK_SENTINEL = object()
by_mod = {}


def register_mod_content_pack(content_pack: ContentPack):
    by_mod[content_pack.name] = content_pack


def register_fake_content_pack(content_pack: str):
    by_mod[content_pack] = FAKE_CONTENT_PACK_SENTINEL
