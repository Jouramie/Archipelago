from collections.abc import Mapping
from types import MappingProxyType

from .model import ConnectionData, RandomizationFlag, RegionData
from ..strings.entrance_names import LogicEntrance, Entrance
from ..strings.region_names import LogicRegion, Region

vanilla_regions: tuple[RegionData, ...] = (
    RegionData(Region.menu, (Entrance.to_stardew_valley,)),
    RegionData(Region.stardew_valley, (Entrance.to_farmhouse,)),
    RegionData(Region.farm_house,
               (Entrance.farmhouse_to_farm, Entrance.downstairs_to_cellar, LogicEntrance.farmhouse_cooking, LogicEntrance.watch_queen_of_sauce)),
    RegionData(Region.cellar),
    RegionData(Region.farm,
               (Entrance.farm_to_backwoods, Entrance.farm_to_bus_stop, Entrance.farm_to_forest, Entrance.farm_to_farmcave, Entrance.enter_greenhouse,
                Entrance.enter_coop, Entrance.enter_barn, Entrance.enter_shed, Entrance.enter_slime_hutch, LogicEntrance.grow_spring_crops,
                LogicEntrance.grow_summer_crops, LogicEntrance.grow_fall_crops, LogicEntrance.grow_winter_crops, LogicEntrance.shipping)),
    RegionData(Region.backwoods, (Entrance.backwoods_to_mountain,)),
    RegionData(Region.bus_stop,
               (Entrance.bus_stop_to_town, Entrance.take_bus_to_desert, Entrance.bus_stop_to_tunnel_entrance)),
    RegionData(Region.forest,
               (Entrance.forest_to_town, Entrance.enter_secret_woods, Entrance.forest_to_wizard_tower, Entrance.forest_to_marnie_ranch,
                Entrance.forest_to_leah_cottage, Entrance.forest_to_sewer, Entrance.forest_to_mastery_cave, LogicEntrance.buy_from_traveling_merchant,
                LogicEntrance.complete_raccoon_requests, LogicEntrance.fish_in_waterfall, LogicEntrance.attend_flower_dance, LogicEntrance.attend_trout_derby,
                LogicEntrance.attend_festival_of_ice)),
    RegionData(LogicRegion.forest_waterfall),
    RegionData(Region.farm_cave),
    RegionData(Region.greenhouse,
               (LogicEntrance.grow_spring_crops_in_greenhouse, LogicEntrance.grow_summer_crops_in_greenhouse, LogicEntrance.grow_fall_crops_in_greenhouse,
                LogicEntrance.grow_winter_crops_in_greenhouse, LogicEntrance.grow_indoor_crops_in_greenhouse)),
    RegionData(Region.mountain,
               (Entrance.mountain_to_railroad, Entrance.mountain_to_tent, Entrance.mountain_to_carpenter_shop,
                Entrance.mountain_to_the_mines, Entrance.enter_quarry, Entrance.mountain_to_adventurer_guild,
                Entrance.mountain_to_town, Entrance.mountain_to_maru_room)),
    RegionData(Region.maru_room),
    RegionData(Region.tunnel_entrance, (Entrance.tunnel_entrance_to_bus_tunnel,)),
    RegionData(Region.bus_tunnel),
    RegionData(Region.town,
               (Entrance.town_to_community_center, Entrance.town_to_beach, Entrance.town_to_hospital, Entrance.town_to_pierre_general_store,
                Entrance.town_to_saloon, Entrance.town_to_alex_house, Entrance.town_to_trailer, Entrance.town_to_mayor_manor, Entrance.town_to_sam_house,
                Entrance.town_to_haley_house, Entrance.town_to_sewer, Entrance.town_to_clint_blacksmith, Entrance.town_to_museum, Entrance.town_to_jojamart,
                Entrance.purchase_movie_ticket, LogicEntrance.buy_experience_books, LogicEntrance.attend_egg_festival, LogicEntrance.attend_fair,
                LogicEntrance.attend_spirit_eve, LogicEntrance.attend_winter_star)),
    RegionData(Region.beach,
               (Entrance.beach_to_willy_fish_shop, Entrance.enter_elliott_house, Entrance.enter_tide_pools, LogicEntrance.fishing, LogicEntrance.attend_luau,
                LogicEntrance.attend_moonlight_jellies, LogicEntrance.attend_night_market, LogicEntrance.attend_squidfest)),
    RegionData(Region.railroad, (Entrance.enter_bathhouse_entrance, Entrance.enter_witch_warp_cave)),
    RegionData(Region.ranch),
    RegionData(Region.leah_house),
    RegionData(Region.mastery_cave),
    RegionData(Region.sewer, (Entrance.enter_mutant_bug_lair,)),
    RegionData(Region.mutant_bug_lair),
    RegionData(Region.wizard_tower, (Entrance.enter_wizard_basement, Entrance.use_desert_obelisk)),
    RegionData(Region.wizard_basement),
    RegionData(Region.tent),
    RegionData(Region.carpenter, (Entrance.enter_sebastian_room,)),
    RegionData(Region.sebastian_room),
    RegionData(Region.adventurer_guild, (Entrance.adventurer_guild_to_bedroom,)),
    RegionData(Region.adventurer_guild_bedroom),
    RegionData(Region.community_center,
               (Entrance.access_crafts_room, Entrance.access_pantry, Entrance.access_fish_tank,
                Entrance.access_boiler_room, Entrance.access_bulletin_board, Entrance.access_vault)),
    RegionData(Region.crafts_room),
    RegionData(Region.pantry),
    RegionData(Region.fish_tank),
    RegionData(Region.boiler_room),
    RegionData(Region.bulletin_board),
    RegionData(Region.vault),
    RegionData(Region.hospital, (Entrance.enter_harvey_room,)),
    RegionData(Region.harvey_room),
    RegionData(Region.pierre_store, (Entrance.enter_sunroom,)),
    RegionData(Region.sunroom),
    RegionData(Region.saloon, (Entrance.play_journey_of_the_prairie_king, Entrance.play_junimo_kart)),
    RegionData(Region.jotpk_world_1, (Entrance.reach_jotpk_world_2,)),
    RegionData(Region.jotpk_world_2, (Entrance.reach_jotpk_world_3,)),
    RegionData(Region.jotpk_world_3),
    RegionData(Region.junimo_kart_1, (Entrance.reach_junimo_kart_2,)),
    RegionData(Region.junimo_kart_2, (Entrance.reach_junimo_kart_3,)),
    RegionData(Region.junimo_kart_3, (Entrance.reach_junimo_kart_4,)),
    RegionData(Region.junimo_kart_4),
    RegionData(Region.alex_house),
    RegionData(Region.trailer),
    RegionData(Region.mayor_house),
    RegionData(Region.sam_house),
    RegionData(Region.haley_house),
    RegionData(Region.blacksmith, (LogicEntrance.blacksmith_copper,)),
    RegionData(Region.museum),
    RegionData(Region.jojamart, (Entrance.enter_abandoned_jojamart,)),
    RegionData(Region.abandoned_jojamart, (Entrance.enter_movie_theater,)),
    RegionData(Region.movie_ticket_stand),
    RegionData(Region.movie_theater),
    RegionData(Region.fish_shop),
    RegionData(Region.elliott_house),
    RegionData(Region.tide_pools),
    RegionData(Region.bathhouse_entrance, (Entrance.enter_locker_room,)),
    RegionData(Region.locker_room, (Entrance.enter_public_bath,)),
    RegionData(Region.public_bath),
    RegionData(Region.witch_warp_cave, (Entrance.enter_witch_swamp,)),
    RegionData(Region.witch_swamp, (Entrance.enter_witch_hut,)),
    RegionData(Region.witch_hut, (Entrance.witch_warp_to_wizard_basement,)),
    RegionData(Region.quarry, (Entrance.enter_quarry_mine_entrance,)),
    RegionData(Region.quarry_mine_entrance, (Entrance.enter_quarry_mine,)),
    RegionData(Region.quarry_mine),
    RegionData(Region.secret_woods),
    RegionData(Region.desert, (Entrance.enter_skull_cavern_entrance, Entrance.enter_oasis, LogicEntrance.attend_desert_festival)),
    RegionData(Region.oasis, (Entrance.enter_casino,)),
    RegionData(Region.casino),
    RegionData(Region.skull_cavern_entrance, (Entrance.enter_skull_cavern,)),
    RegionData(Region.skull_cavern, (Entrance.mine_to_skull_cavern_floor_25,)),
    RegionData(Region.skull_cavern_25, (Entrance.mine_to_skull_cavern_floor_50,)),
    RegionData(Region.skull_cavern_50, (Entrance.mine_to_skull_cavern_floor_75,)),
    RegionData(Region.skull_cavern_75, (Entrance.mine_to_skull_cavern_floor_100,)),
    RegionData(Region.skull_cavern_100, (Entrance.mine_to_skull_cavern_floor_125,)),
    RegionData(Region.skull_cavern_125, (Entrance.mine_to_skull_cavern_floor_150,)),
    RegionData(Region.skull_cavern_150, (Entrance.mine_to_skull_cavern_floor_175,)),
    RegionData(Region.skull_cavern_175, (Entrance.mine_to_skull_cavern_floor_200,)),
    RegionData(Region.skull_cavern_200),

    RegionData(Region.coop),
    RegionData(Region.barn),
    RegionData(Region.shed),
    RegionData(Region.slime_hutch),

    RegionData(Region.mines, (LogicEntrance.talk_to_mines_dwarf, Entrance.dig_to_mines_floor_5)),
    RegionData(Region.mines_floor_5, (Entrance.dig_to_mines_floor_10,)),
    RegionData(Region.mines_floor_10, (Entrance.dig_to_mines_floor_15,)),
    RegionData(Region.mines_floor_15, (Entrance.dig_to_mines_floor_20,)),
    RegionData(Region.mines_floor_20, (Entrance.dig_to_mines_floor_25,)),
    RegionData(Region.mines_floor_25, (Entrance.dig_to_mines_floor_30,)),
    RegionData(Region.mines_floor_30, (Entrance.dig_to_mines_floor_35,)),
    RegionData(Region.mines_floor_35, (Entrance.dig_to_mines_floor_40,)),
    RegionData(Region.mines_floor_40, (Entrance.dig_to_mines_floor_45,)),
    RegionData(Region.mines_floor_45, (Entrance.dig_to_mines_floor_50,)),
    RegionData(Region.mines_floor_50, (Entrance.dig_to_mines_floor_55,)),
    RegionData(Region.mines_floor_55, (Entrance.dig_to_mines_floor_60,)),
    RegionData(Region.mines_floor_60, (Entrance.dig_to_mines_floor_65,)),
    RegionData(Region.mines_floor_65, (Entrance.dig_to_mines_floor_70,)),
    RegionData(Region.mines_floor_70, (Entrance.dig_to_mines_floor_75,)),
    RegionData(Region.mines_floor_75, (Entrance.dig_to_mines_floor_80,)),
    RegionData(Region.mines_floor_80, (Entrance.dig_to_mines_floor_85,)),
    RegionData(Region.mines_floor_85, (Entrance.dig_to_mines_floor_90,)),
    RegionData(Region.mines_floor_90, (Entrance.dig_to_mines_floor_95,)),
    RegionData(Region.mines_floor_95, (Entrance.dig_to_mines_floor_100,)),
    RegionData(Region.mines_floor_100, (Entrance.dig_to_mines_floor_105,)),
    RegionData(Region.mines_floor_105, (Entrance.dig_to_mines_floor_110,)),
    RegionData(Region.mines_floor_110, (Entrance.dig_to_mines_floor_115,)),
    RegionData(Region.mines_floor_115, (Entrance.dig_to_mines_floor_120,)),
    RegionData(Region.mines_floor_120),

    RegionData(LogicRegion.mines_dwarf_shop),
    RegionData(LogicRegion.blacksmith_copper, (LogicEntrance.blacksmith_iron,)),
    RegionData(LogicRegion.blacksmith_iron, (LogicEntrance.blacksmith_gold,)),
    RegionData(LogicRegion.blacksmith_gold, (LogicEntrance.blacksmith_iridium,)),
    RegionData(LogicRegion.blacksmith_iridium),
    RegionData(LogicRegion.kitchen),
    RegionData(LogicRegion.queen_of_sauce),
    RegionData(LogicRegion.fishing),

    RegionData(LogicRegion.spring_farming),
    RegionData(LogicRegion.summer_farming, (LogicEntrance.grow_summer_fall_crops_in_summer,)),
    RegionData(LogicRegion.fall_farming, (LogicEntrance.grow_summer_fall_crops_in_fall,)),
    RegionData(LogicRegion.winter_farming),
    RegionData(LogicRegion.summer_or_fall_farming),
    RegionData(LogicRegion.indoor_farming),

    RegionData(LogicRegion.shipping),
    RegionData(LogicRegion.traveling_cart, (LogicEntrance.buy_from_traveling_merchant_sunday,
                                            LogicEntrance.buy_from_traveling_merchant_monday,
                                            LogicEntrance.buy_from_traveling_merchant_tuesday,
                                            LogicEntrance.buy_from_traveling_merchant_wednesday,
                                            LogicEntrance.buy_from_traveling_merchant_thursday,
                                            LogicEntrance.buy_from_traveling_merchant_friday,
                                            LogicEntrance.buy_from_traveling_merchant_saturday)),
    RegionData(LogicRegion.traveling_cart_sunday),
    RegionData(LogicRegion.traveling_cart_monday),
    RegionData(LogicRegion.traveling_cart_tuesday),
    RegionData(LogicRegion.traveling_cart_wednesday),
    RegionData(LogicRegion.traveling_cart_thursday),
    RegionData(LogicRegion.traveling_cart_friday),
    RegionData(LogicRegion.traveling_cart_saturday),
    RegionData(LogicRegion.raccoon_daddy, (LogicEntrance.buy_from_raccoon,)),
    RegionData(LogicRegion.raccoon_shop),

    RegionData(LogicRegion.egg_festival),
    RegionData(LogicRegion.desert_festival),
    RegionData(LogicRegion.flower_dance),
    RegionData(LogicRegion.luau),
    RegionData(LogicRegion.trout_derby),
    RegionData(LogicRegion.moonlight_jellies),
    RegionData(LogicRegion.fair),
    RegionData(LogicRegion.spirit_eve),
    RegionData(LogicRegion.festival_of_ice),
    RegionData(LogicRegion.night_market),
    RegionData(LogicRegion.winter_star),
    RegionData(LogicRegion.squidfest),
    RegionData(LogicRegion.bookseller_1, (LogicEntrance.buy_year1_books,)),
    RegionData(LogicRegion.bookseller_2, (LogicEntrance.buy_year3_books,)),
    RegionData(LogicRegion.bookseller_3),
)
ginger_island_regions = (
    # This overrides the regions from vanilla... When regions are moved to content packs, overriding existing entrances should no longer be necessary.
    RegionData(Region.mountain,
               (Entrance.mountain_to_railroad, Entrance.mountain_to_tent, Entrance.mountain_to_carpenter_shop,
                Entrance.mountain_to_the_mines, Entrance.enter_quarry, Entrance.mountain_to_adventurer_guild,
                Entrance.mountain_to_town, Entrance.mountain_to_maru_room, Entrance.mountain_to_leo_treehouse)),
    RegionData(Region.wizard_tower, (Entrance.enter_wizard_basement, Entrance.use_desert_obelisk, Entrance.use_island_obelisk,)),
    RegionData(Region.fish_shop, (Entrance.fish_shop_to_boat_tunnel,)),
    RegionData(Region.mines_floor_120, (Entrance.dig_to_dangerous_mines_20, Entrance.dig_to_dangerous_mines_60, Entrance.dig_to_dangerous_mines_100)),
    RegionData(Region.skull_cavern_200, (Entrance.enter_dangerous_skull_cavern,)),

    RegionData(Region.leo_treehouse),
    RegionData(Region.boat_tunnel, (Entrance.boat_to_ginger_island,)),
    RegionData(Region.dangerous_skull_cavern),
    RegionData(Region.island_south,
               (Entrance.island_south_to_west, Entrance.island_south_to_north, Entrance.island_south_to_east, Entrance.island_south_to_southeast,
                Entrance.use_island_resort, Entrance.parrot_express_docks_to_volcano, Entrance.parrot_express_docks_to_dig_site,
                Entrance.parrot_express_docks_to_jungle), ),
    RegionData(Region.island_resort),
    RegionData(Region.island_west,
               (Entrance.island_west_to_islandfarmhouse, Entrance.island_west_to_gourmand_cave, Entrance.island_west_to_crystals_cave,
                Entrance.island_west_to_shipwreck, Entrance.island_west_to_qi_walnut_room, Entrance.use_farm_obelisk, Entrance.parrot_express_jungle_to_docks,
                Entrance.parrot_express_jungle_to_dig_site, Entrance.parrot_express_jungle_to_volcano, LogicEntrance.grow_spring_crops_on_island,
                LogicEntrance.grow_summer_crops_on_island, LogicEntrance.grow_fall_crops_on_island, LogicEntrance.grow_winter_crops_on_island,
                LogicEntrance.grow_indoor_crops_on_island), ),
    RegionData(Region.island_east, (Entrance.island_east_to_leo_hut, Entrance.island_east_to_island_shrine)),
    RegionData(Region.island_shrine),
    RegionData(Region.island_south_east, (Entrance.island_southeast_to_pirate_cove,)),
    RegionData(Region.island_north,
               (Entrance.talk_to_island_trader, Entrance.island_north_to_field_office, Entrance.island_north_to_dig_site, Entrance.island_north_to_volcano,
                Entrance.parrot_express_volcano_to_dig_site, Entrance.parrot_express_volcano_to_jungle, Entrance.parrot_express_volcano_to_docks), ),
    RegionData(Region.volcano, (Entrance.climb_to_volcano_5, Entrance.volcano_to_secret_beach)),
    RegionData(Region.volcano_secret_beach),
    RegionData(Region.volcano_floor_5, (Entrance.talk_to_volcano_dwarf, Entrance.climb_to_volcano_10)),
    RegionData(Region.volcano_dwarf_shop),
    RegionData(Region.volcano_floor_10),
    RegionData(Region.island_trader),
    RegionData(Region.island_farmhouse, (LogicEntrance.island_cooking,)),
    RegionData(Region.gourmand_frog_cave),
    RegionData(Region.colored_crystals_cave),
    RegionData(Region.shipwreck),
    RegionData(Region.qi_walnut_room),
    RegionData(Region.leo_hut),
    RegionData(Region.pirate_cove),
    RegionData(Region.field_office),
    RegionData(Region.dig_site,
               (Entrance.dig_site_to_professor_snail_cave, Entrance.parrot_express_dig_site_to_volcano,
                Entrance.parrot_express_dig_site_to_docks, Entrance.parrot_express_dig_site_to_jungle), ),

    RegionData(Region.professor_snail_cave),
    RegionData(Region.dangerous_mines_20),
    RegionData(Region.dangerous_mines_60),
    RegionData(Region.dangerous_mines_100),
)

# Exists and where they lead
vanilla_connections: tuple[ConnectionData, ...] = (
    ConnectionData(Entrance.to_stardew_valley, Region.stardew_valley),
    ConnectionData(Entrance.to_farmhouse, Region.farm_house),
    ConnectionData(Entrance.farmhouse_to_farm, Region.farm),
    ConnectionData(Entrance.downstairs_to_cellar, Region.cellar),
    ConnectionData(Entrance.farm_to_backwoods, Region.backwoods),
    ConnectionData(Entrance.farm_to_bus_stop, Region.bus_stop),
    ConnectionData(Entrance.farm_to_forest, Region.forest),
    ConnectionData(Entrance.farm_to_farmcave, Region.farm_cave, flag=RandomizationFlag.NON_PROGRESSION),
    ConnectionData(Entrance.enter_greenhouse, Region.greenhouse),
    ConnectionData(Entrance.enter_coop, Region.coop),
    ConnectionData(Entrance.enter_barn, Region.barn),
    ConnectionData(Entrance.enter_shed, Region.shed),
    ConnectionData(Entrance.enter_slime_hutch, Region.slime_hutch),
    ConnectionData(Entrance.use_desert_obelisk, Region.desert),
    ConnectionData(Entrance.backwoods_to_mountain, Region.mountain),
    ConnectionData(Entrance.bus_stop_to_town, Region.town),
    ConnectionData(Entrance.bus_stop_to_tunnel_entrance, Region.tunnel_entrance),
    ConnectionData(Entrance.tunnel_entrance_to_bus_tunnel, Region.bus_tunnel, flag=RandomizationFlag.NON_PROGRESSION),
    ConnectionData(Entrance.take_bus_to_desert, Region.desert),
    ConnectionData(Entrance.forest_to_town, Region.town),
    ConnectionData(Entrance.forest_to_wizard_tower, Region.wizard_tower,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_wizard_basement, Region.wizard_basement, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.forest_to_marnie_ranch, Region.ranch,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.forest_to_leah_cottage, Region.leah_house,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_secret_woods, Region.secret_woods),
    ConnectionData(Entrance.forest_to_sewer, Region.sewer, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.forest_to_mastery_cave, Region.mastery_cave, flag=RandomizationFlag.BUILDINGS ^ RandomizationFlag.EXCLUDE_MASTERIES),
    ConnectionData(Entrance.town_to_sewer, Region.sewer, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_mutant_bug_lair, Region.mutant_bug_lair, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.mountain_to_railroad, Region.railroad),
    ConnectionData(Entrance.mountain_to_tent, Region.tent,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.mountain_to_carpenter_shop, Region.carpenter,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.mountain_to_maru_room, Region.maru_room,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_sebastian_room, Region.sebastian_room, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.mountain_to_adventurer_guild, Region.adventurer_guild,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.adventurer_guild_to_bedroom, Region.adventurer_guild_bedroom),
    ConnectionData(Entrance.enter_quarry, Region.quarry),
    ConnectionData(Entrance.enter_quarry_mine_entrance, Region.quarry_mine_entrance,
                   flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_quarry_mine, Region.quarry_mine),
    ConnectionData(Entrance.mountain_to_town, Region.town),
    ConnectionData(Entrance.town_to_community_center, Region.community_center,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.access_crafts_room, Region.crafts_room),
    ConnectionData(Entrance.access_pantry, Region.pantry),
    ConnectionData(Entrance.access_fish_tank, Region.fish_tank),
    ConnectionData(Entrance.access_boiler_room, Region.boiler_room),
    ConnectionData(Entrance.access_bulletin_board, Region.bulletin_board),
    ConnectionData(Entrance.access_vault, Region.vault),
    ConnectionData(Entrance.town_to_hospital, Region.hospital,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_harvey_room, Region.harvey_room, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.town_to_pierre_general_store, Region.pierre_store,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_sunroom, Region.sunroom, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.town_to_clint_blacksmith, Region.blacksmith,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_saloon, Region.saloon,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.play_journey_of_the_prairie_king, Region.jotpk_world_1),
    ConnectionData(Entrance.reach_jotpk_world_2, Region.jotpk_world_2),
    ConnectionData(Entrance.reach_jotpk_world_3, Region.jotpk_world_3),
    ConnectionData(Entrance.play_junimo_kart, Region.junimo_kart_1),
    ConnectionData(Entrance.reach_junimo_kart_2, Region.junimo_kart_2),
    ConnectionData(Entrance.reach_junimo_kart_3, Region.junimo_kart_3),
    ConnectionData(Entrance.reach_junimo_kart_4, Region.junimo_kart_4),
    ConnectionData(Entrance.town_to_sam_house, Region.sam_house,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_haley_house, Region.haley_house,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_mayor_manor, Region.mayor_house,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_alex_house, Region.alex_house,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_trailer, Region.trailer,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_museum, Region.museum,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.town_to_jojamart, Region.jojamart,
                   flag=RandomizationFlag.PELICAN_TOWN | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.purchase_movie_ticket, Region.movie_ticket_stand),
    ConnectionData(Entrance.enter_abandoned_jojamart, Region.abandoned_jojamart),
    ConnectionData(Entrance.enter_movie_theater, Region.movie_theater),
    ConnectionData(Entrance.town_to_beach, Region.beach),
    ConnectionData(Entrance.enter_elliott_house, Region.elliott_house,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.beach_to_willy_fish_shop, Region.fish_shop,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_tide_pools, Region.tide_pools),
    ConnectionData(Entrance.mountain_to_the_mines, Region.mines,
                   flag=RandomizationFlag.NON_PROGRESSION | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.dig_to_mines_floor_5, Region.mines_floor_5),
    ConnectionData(Entrance.dig_to_mines_floor_10, Region.mines_floor_10),
    ConnectionData(Entrance.dig_to_mines_floor_15, Region.mines_floor_15),
    ConnectionData(Entrance.dig_to_mines_floor_20, Region.mines_floor_20),
    ConnectionData(Entrance.dig_to_mines_floor_25, Region.mines_floor_25),
    ConnectionData(Entrance.dig_to_mines_floor_30, Region.mines_floor_30),
    ConnectionData(Entrance.dig_to_mines_floor_35, Region.mines_floor_35),
    ConnectionData(Entrance.dig_to_mines_floor_40, Region.mines_floor_40),
    ConnectionData(Entrance.dig_to_mines_floor_45, Region.mines_floor_45),
    ConnectionData(Entrance.dig_to_mines_floor_50, Region.mines_floor_50),
    ConnectionData(Entrance.dig_to_mines_floor_55, Region.mines_floor_55),
    ConnectionData(Entrance.dig_to_mines_floor_60, Region.mines_floor_60),
    ConnectionData(Entrance.dig_to_mines_floor_65, Region.mines_floor_65),
    ConnectionData(Entrance.dig_to_mines_floor_70, Region.mines_floor_70),
    ConnectionData(Entrance.dig_to_mines_floor_75, Region.mines_floor_75),
    ConnectionData(Entrance.dig_to_mines_floor_80, Region.mines_floor_80),
    ConnectionData(Entrance.dig_to_mines_floor_85, Region.mines_floor_85),
    ConnectionData(Entrance.dig_to_mines_floor_90, Region.mines_floor_90),
    ConnectionData(Entrance.dig_to_mines_floor_95, Region.mines_floor_95),
    ConnectionData(Entrance.dig_to_mines_floor_100, Region.mines_floor_100),
    ConnectionData(Entrance.dig_to_mines_floor_105, Region.mines_floor_105),
    ConnectionData(Entrance.dig_to_mines_floor_110, Region.mines_floor_110),
    ConnectionData(Entrance.dig_to_mines_floor_115, Region.mines_floor_115),
    ConnectionData(Entrance.dig_to_mines_floor_120, Region.mines_floor_120),
    ConnectionData(Entrance.enter_skull_cavern_entrance, Region.skull_cavern_entrance,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_oasis, Region.oasis,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_casino, Region.casino, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_skull_cavern, Region.skull_cavern),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_25, Region.skull_cavern_25),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_50, Region.skull_cavern_50),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_75, Region.skull_cavern_75),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_100, Region.skull_cavern_100),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_125, Region.skull_cavern_125),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_150, Region.skull_cavern_150),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_175, Region.skull_cavern_175),
    ConnectionData(Entrance.mine_to_skull_cavern_floor_200, Region.skull_cavern_200),
    ConnectionData(Entrance.enter_witch_warp_cave, Region.witch_warp_cave, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_witch_swamp, Region.witch_swamp, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_witch_hut, Region.witch_hut, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.witch_warp_to_wizard_basement, Region.wizard_basement, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_bathhouse_entrance, Region.bathhouse_entrance,
                   flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.enter_locker_room, Region.locker_room, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.enter_public_bath, Region.public_bath, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(LogicEntrance.talk_to_mines_dwarf, LogicRegion.mines_dwarf_shop),

    ConnectionData(LogicEntrance.buy_from_traveling_merchant, LogicRegion.traveling_cart),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_sunday, LogicRegion.traveling_cart_sunday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_monday, LogicRegion.traveling_cart_monday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_tuesday, LogicRegion.traveling_cart_tuesday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_wednesday, LogicRegion.traveling_cart_wednesday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_thursday, LogicRegion.traveling_cart_thursday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_friday, LogicRegion.traveling_cart_friday),
    ConnectionData(LogicEntrance.buy_from_traveling_merchant_saturday, LogicRegion.traveling_cart_saturday),
    ConnectionData(LogicEntrance.complete_raccoon_requests, LogicRegion.raccoon_daddy),
    ConnectionData(LogicEntrance.fish_in_waterfall, LogicRegion.forest_waterfall),
    ConnectionData(LogicEntrance.buy_from_raccoon, LogicRegion.raccoon_shop),
    ConnectionData(LogicEntrance.farmhouse_cooking, LogicRegion.kitchen),
    ConnectionData(LogicEntrance.watch_queen_of_sauce, LogicRegion.queen_of_sauce),

    ConnectionData(LogicEntrance.grow_spring_crops, LogicRegion.spring_farming),
    ConnectionData(LogicEntrance.grow_summer_crops, LogicRegion.summer_farming),
    ConnectionData(LogicEntrance.grow_fall_crops, LogicRegion.fall_farming),
    ConnectionData(LogicEntrance.grow_winter_crops, LogicRegion.winter_farming),
    ConnectionData(LogicEntrance.grow_spring_crops_in_greenhouse, LogicRegion.spring_farming),
    ConnectionData(LogicEntrance.grow_summer_crops_in_greenhouse, LogicRegion.summer_farming),
    ConnectionData(LogicEntrance.grow_fall_crops_in_greenhouse, LogicRegion.fall_farming),
    ConnectionData(LogicEntrance.grow_winter_crops_in_greenhouse, LogicRegion.winter_farming),
    ConnectionData(LogicEntrance.grow_indoor_crops_in_greenhouse, LogicRegion.indoor_farming),
    ConnectionData(LogicEntrance.grow_summer_fall_crops_in_summer, LogicRegion.summer_or_fall_farming),
    ConnectionData(LogicEntrance.grow_summer_fall_crops_in_fall, LogicRegion.summer_or_fall_farming),

    ConnectionData(LogicEntrance.shipping, LogicRegion.shipping),
    ConnectionData(LogicEntrance.blacksmith_copper, LogicRegion.blacksmith_copper),
    ConnectionData(LogicEntrance.blacksmith_iron, LogicRegion.blacksmith_iron),
    ConnectionData(LogicEntrance.blacksmith_gold, LogicRegion.blacksmith_gold),
    ConnectionData(LogicEntrance.blacksmith_iridium, LogicRegion.blacksmith_iridium),
    ConnectionData(LogicEntrance.fishing, LogicRegion.fishing),
    ConnectionData(LogicEntrance.attend_egg_festival, LogicRegion.egg_festival),
    ConnectionData(LogicEntrance.attend_desert_festival, LogicRegion.desert_festival),
    ConnectionData(LogicEntrance.attend_flower_dance, LogicRegion.flower_dance),
    ConnectionData(LogicEntrance.attend_luau, LogicRegion.luau),
    ConnectionData(LogicEntrance.attend_trout_derby, LogicRegion.trout_derby),
    ConnectionData(LogicEntrance.attend_moonlight_jellies, LogicRegion.moonlight_jellies),
    ConnectionData(LogicEntrance.attend_fair, LogicRegion.fair),
    ConnectionData(LogicEntrance.attend_spirit_eve, LogicRegion.spirit_eve),
    ConnectionData(LogicEntrance.attend_festival_of_ice, LogicRegion.festival_of_ice),
    ConnectionData(LogicEntrance.attend_night_market, LogicRegion.night_market),
    ConnectionData(LogicEntrance.attend_winter_star, LogicRegion.winter_star),
    ConnectionData(LogicEntrance.attend_squidfest, LogicRegion.squidfest),
    ConnectionData(LogicEntrance.buy_experience_books, LogicRegion.bookseller_1),
    ConnectionData(LogicEntrance.buy_year1_books, LogicRegion.bookseller_2),
    ConnectionData(LogicEntrance.buy_year3_books, LogicRegion.bookseller_3),
)

ginger_island_connections = (
    ConnectionData(Entrance.use_island_obelisk, Region.island_south),
    ConnectionData(Entrance.use_farm_obelisk, Region.farm),
    ConnectionData(Entrance.mountain_to_leo_treehouse, Region.leo_treehouse, flag=RandomizationFlag.BUILDINGS | RandomizationFlag.LEAD_TO_OPEN_AREA),
    ConnectionData(Entrance.fish_shop_to_boat_tunnel, Region.boat_tunnel, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.boat_to_ginger_island, Region.island_south),
    ConnectionData(Entrance.enter_dangerous_skull_cavern, Region.dangerous_skull_cavern),
    ConnectionData(Entrance.dig_to_dangerous_mines_20, Region.dangerous_mines_20),
    ConnectionData(Entrance.dig_to_dangerous_mines_60, Region.dangerous_mines_60),
    ConnectionData(Entrance.dig_to_dangerous_mines_100, Region.dangerous_mines_100),
    ConnectionData(Entrance.island_south_to_west, Region.island_west),
    ConnectionData(Entrance.island_south_to_north, Region.island_north),
    ConnectionData(Entrance.island_south_to_east, Region.island_east),
    ConnectionData(Entrance.island_south_to_southeast, Region.island_south_east),
    ConnectionData(Entrance.use_island_resort, Region.island_resort),
    ConnectionData(Entrance.island_west_to_islandfarmhouse, Region.island_farmhouse, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_west_to_gourmand_cave, Region.gourmand_frog_cave, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_west_to_crystals_cave, Region.colored_crystals_cave, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_west_to_shipwreck, Region.shipwreck, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_west_to_qi_walnut_room, Region.qi_walnut_room, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_east_to_leo_hut, Region.leo_hut, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_east_to_island_shrine, Region.island_shrine, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_southeast_to_pirate_cove, Region.pirate_cove, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_north_to_field_office, Region.field_office, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_north_to_dig_site, Region.dig_site),
    ConnectionData(Entrance.dig_site_to_professor_snail_cave, Region.professor_snail_cave, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.island_north_to_volcano, Region.volcano, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.volcano_to_secret_beach, Region.volcano_secret_beach, flag=RandomizationFlag.BUILDINGS),
    ConnectionData(Entrance.talk_to_island_trader, Region.island_trader),
    ConnectionData(Entrance.climb_to_volcano_5, Region.volcano_floor_5),
    ConnectionData(Entrance.talk_to_volcano_dwarf, Region.volcano_dwarf_shop),
    ConnectionData(Entrance.climb_to_volcano_10, Region.volcano_floor_10),
    ConnectionData(Entrance.parrot_express_jungle_to_docks, Region.island_south),
    ConnectionData(Entrance.parrot_express_dig_site_to_docks, Region.island_south),
    ConnectionData(Entrance.parrot_express_volcano_to_docks, Region.island_south),
    ConnectionData(Entrance.parrot_express_volcano_to_jungle, Region.island_west),
    ConnectionData(Entrance.parrot_express_docks_to_jungle, Region.island_west),
    ConnectionData(Entrance.parrot_express_dig_site_to_jungle, Region.island_west),
    ConnectionData(Entrance.parrot_express_docks_to_dig_site, Region.dig_site),
    ConnectionData(Entrance.parrot_express_volcano_to_dig_site, Region.dig_site),
    ConnectionData(Entrance.parrot_express_jungle_to_dig_site, Region.dig_site),
    ConnectionData(Entrance.parrot_express_dig_site_to_volcano, Region.island_north),
    ConnectionData(Entrance.parrot_express_docks_to_volcano, Region.island_north),
    ConnectionData(Entrance.parrot_express_jungle_to_volcano, Region.island_north),
    ConnectionData(LogicEntrance.grow_spring_crops_on_island, LogicRegion.spring_farming),
    ConnectionData(LogicEntrance.grow_summer_crops_on_island, LogicRegion.summer_farming),
    ConnectionData(LogicEntrance.grow_fall_crops_on_island, LogicRegion.fall_farming),
    ConnectionData(LogicEntrance.grow_winter_crops_on_island, LogicRegion.winter_farming),
    ConnectionData(LogicEntrance.grow_indoor_crops_on_island, LogicRegion.indoor_farming),
    ConnectionData(LogicEntrance.island_cooking, LogicRegion.kitchen),
)

connections_without_ginger_island_by_name: Mapping[str, ConnectionData] = MappingProxyType({
    connection.name: connection
    for connection in vanilla_connections
})
regions_without_ginger_island_by_name: Mapping[str, RegionData] = MappingProxyType({
    region.name: region
    for region in vanilla_regions
})

connections_with_ginger_island_by_name: Mapping[str, ConnectionData] = MappingProxyType({
    connection.name: connection
    for connection in vanilla_connections + ginger_island_connections
})
regions_with_ginger_island_by_name: Mapping[str, RegionData] = MappingProxyType({
    region.name: region
    for region in vanilla_regions + ginger_island_regions
})