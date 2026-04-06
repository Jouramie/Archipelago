all_events = set()


def event(name: str):
    all_events.add(name)
    return name


class Event:
    victory = event("Victory")

    received_walnuts = event("Received Walnuts")
    received_qi_gems = event("Received Qi Gems")
    received_progressive_weapon = event("Received Progressive Weapon")
    received_progression_item = event("Received Progression Item")
    received_progression_percent = event("Received Progression Percent")

    sleep_in_farmhouse = event("Sleep in Farmhouse")
    sleep_in_island_farmhouse = event("Sleep in Island Farmhouse")
