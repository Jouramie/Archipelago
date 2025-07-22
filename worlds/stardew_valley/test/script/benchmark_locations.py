"""
Copy of the script in test/benchmark, adapted to Stardew Valley.

Run with `python -m worlds.stardew_valley.test.script.benchmark_locations --options minimal_locations_maximal_items`
"""

import argparse
import collections
import gc
import logging
import time
import typing
from random import Random
from typing import Protocol

from BaseClasses import CollectionState, Location, Item, get_seed
from Utils import init_logging
from ..bases import setup_solo_multiworld
from ..options import presets
from ...stardew_rule import to_optimized_v2, to_optimized_v1
from ...stardew_rule.rule_explain import explain

assert to_optimized_v2
assert to_optimized_v1


def run_locations_benchmark():
    init_logging("Benchmark Runner")
    logger = logging.getLogger("Benchmark")

    class StateGenerator(Protocol):
        def __next__(self) -> CollectionState:
            ...

    class CollectingStateGenerator:
        def __init__(self, state: CollectionState, items: list[Item], random: Random):
            self.state = state
            self.collecting_queue = collections.deque(items)
            self.removing_queue = collections.deque()
            self.random = random

            self.is_filling = True

            self.random.shuffle(self.collecting_queue)

        def __next__(self) -> CollectionState:
            if self.is_filling:
                item = self.collecting_queue.pop()
                self.state.collect(item)
                self.removing_queue.append(item)

                if not self.collecting_queue:
                    self.is_filling = False
                    self.random.shuffle(self.removing_queue)

            else:
                item = self.removing_queue.pop()
                self.state.remove(item)
                self.collecting_queue.append(item)

                if not self.removing_queue:
                    self.is_filling = True
                    self.random.shuffle(self.collecting_queue)

            return self.state

    class NoopStateGenerator:
        def __init__(self, state: CollectionState):
            self.state = state

        def __next__(self) -> CollectionState:
            return self.state

    class BenchmarkRunner:
        gen_steps: typing.Tuple[str, ...] = (
            "generate_early", "create_regions", "create_items", "set_rules", "generate_basic", "pre_fill")
        rule_iterations: int = 1_000

        @staticmethod
        def format_times_from_counter(counter: collections.Counter[str], top: int = 5) -> str:
            return "\n".join(f"  {time:.4f} in {name}" for name, time in counter.most_common(top))

        def location_test(self, test_location: Location, state: StateGenerator, state_name: str) -> float:
            rule = test_location.access_rule
            rule = to_optimized_v2(rule)
            # logger.info(str(rule))
            # logger.info(str(rule.evaluation_tree))
            # logger.info(f"average depth = {rule.evaluation_tree.average_leaf_depth}")

            with TimeIt(f"{test_location.game} {self.rule_iterations} "
                        f"runs of {test_location}.access_rule({state_name})", logger) as t:
                for _ in range(self.rule_iterations):
                    rule(next(state))
                # if time is taken to disentangle complex ref chains,
                # this time should be attributed to the rule.
                gc.collect()
            return t.dif

        def main(self):
            game = "Stardew Valley"
            summary_data: typing.Dict[str, collections.Counter[str]] = {
                "empty_state": collections.Counter(),
                "filling_state": collections.Counter(),
                "all_state": collections.Counter(),
            }
            try:
                parser = argparse.ArgumentParser()
                parser.add_argument('--options', help="Define the option set to use, from the preset in test/__init__.py .", type=str, required=True)
                parser.add_argument('--seed', help="Define the seed to use.", type=int, required=True)
                parser.add_argument('--filling_seed', help="Define the seed to use for the filling_state.", type=int, default=None)
                parser.add_argument('--location', help="Define the specific location to benchmark.", type=str, default=None)
                parser.add_argument('--state', help="Define the state in which the location will be benchmarked.", type=str, default=None)
                args = parser.parse_args()
                options_set = args.options
                options = getattr(presets, options_set)()
                seed = args.seed
                filling_seed = args.filling_seed
                location = args.location
                state = args.state

                if filling_seed is None:
                    filling_seed = get_seed()

                multiworld = setup_solo_multiworld(options, seed)
                gc.collect()

                if location:
                    locations = [multiworld.get_location(location, 1)]
                else:
                    locations = sorted(multiworld.get_locations(1))

                for location in locations:
                    if state == "empty_state" or not state:
                        time_taken = self.location_test(location, NoopStateGenerator(multiworld.state), "empty_state")
                        summary_data["empty_state"][location.name] = time_taken

                    if state == "filling_state" or not state:
                        logger.info(f"Using seed {filling_seed} for filling_state.")
                        generator = CollectingStateGenerator(multiworld.state, multiworld.itempool, Random(filling_seed))
                        time_taken = self.location_test(location, generator, "filling_state")
                        summary_data["filling_state"][location.name] = time_taken

                    if state == "all_state" or not state:
                        all_state = multiworld.get_all_state(False)
                        time_taken = self.location_test(location, NoopStateGenerator(all_state), "all_state")
                        summary_data["all_state"][location.name] = time_taken

                total_empty_state = sum(summary_data["empty_state"].values())
                total_all_state = sum(summary_data["all_state"].values())

                logger.info(f"{game} took {total_empty_state / len(locations):.4f} "
                            f"seconds per location in empty_state and {total_all_state / len(locations):.4f} "
                            f"in all_state. (all times summed for {self.rule_iterations} runs.)")
                logger.info(f"Top times in empty_state:\n"
                            f"{self.format_times_from_counter(summary_data['empty_state'])}")
                logger.info(f"Top times in all_state:\n"
                            f"{self.format_times_from_counter(summary_data['all_state'])}")

                if len(locations) == 1:
                    logger.info(str(explain(locations[0].access_rule, all_state, False)))

            except Exception as e:
                logger.exception(e)

    runner = BenchmarkRunner()
    runner.main()


class TimeIt:
    def __init__(self, name: str, time_logger=None):
        self.name = name
        self.logger = time_logger
        self.timer = None
        self.end_timer = None

    def __enter__(self):
        self.timer = time.perf_counter()
        return self

    @property
    def dif(self):
        return self.end_timer - self.timer

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.end_timer:
            self.end_timer = time.perf_counter()
        # if self.logger:
        #     self.logger.info(f"{self.dif:.4f} seconds in {self.name}.")

if __name__ == "__main__":
    run_locations_benchmark()
