from .remixed_bundles import all_remixed_bundles

all_bundle_items_except_money = []

for bundle in all_remixed_bundles:
    all_bundle_items_except_money.extend(bundle.items)

all_bundle_items_by_name = {item.item_name: item for item in all_bundle_items_except_money}
