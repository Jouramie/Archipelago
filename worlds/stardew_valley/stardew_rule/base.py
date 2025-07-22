from __future__ import annotations

import enum
import sys
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from threading import Lock
from typing import Iterable, Dict, Union, Sized, Hashable, Callable, Tuple, Set, Optional, cast, ClassVar, Protocol, cast

from BaseClasses import CollectionState
from .literal import true_, false_, LiteralStardewRule
from .protocol import StardewRule

MISSING_ITEM = "THIS ITEM IS MISSING"


# TODO maybe this would be easier to understand in term of lower/upper bounds
class ShortCircuitPropagation(enum.Enum):
    NONE = enum.auto()
    """None mean the two rules are not related and can't short-circuit one another."""
    POSITIVE = enum.auto()
    """Positive mean that a True result from evaluating other will simplify self. Typically, other is more restrictive than self."""
    NEGATIVE = enum.auto()
    """Negative mean that a False result from evaluating other will simplify self. Typically, other is less restrictive than self."""
    EQUAL = enum.auto()
    """Equal mean that True or False results from evaluating other will simplify self. Typically, other and self are as restrictive."""

    @property
    def reverse(self):
        if self is ShortCircuitPropagation.POSITIVE:
            return ShortCircuitPropagation.NEGATIVE
        elif self is ShortCircuitPropagation.NEGATIVE:
            return ShortCircuitPropagation.POSITIVE
        return self


@dataclass(frozen=True)
class AssumptionState:
    """Lower bound is inclusive, upper bound is exclusive.
    """
    combinable_values: Dict[Hashable, Tuple[int, int]] = field(default_factory=dict)

    def add_lower_bounds(self, rule: CanShortCircuit):
        new_bounds = {}

        for key, value in rule.lower_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert upper_bound >= value

            if lower_bound >= value:
                continue

            new_bounds[key] = (value, upper_bound)

        return AssumptionState(self.combinable_values | new_bounds)

    def add_upper_bounds(self, rule: CanShortCircuit):
        new_bounds = {}

        for key, value in rule.upper_bounds:
            lower_bound, upper_bound = self.combinable_values.get(key, (0, sys.maxsize))
            assert lower_bound <= value

            if upper_bound <= value:
                continue

            new_bounds[key] = (lower_bound, value)

        return AssumptionState(self.combinable_values | new_bounds)

    def __str__(self):
        return f"{{{', '.join(f'{key}: {self.str_bound(*bound)}' for key, bound in self.combinable_values.items())}}}"

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def str_bound(lower_bound: int, upper_bound: int) -> str:
        if lower_bound == 0:
            return f"[0, {upper_bound})"
        if upper_bound == sys.maxsize:
            return f"[{lower_bound}, ~)"
        if lower_bound + 1 == upper_bound:
            return f"{lower_bound}"
        return f"[{lower_bound}, {upper_bound})"


class CanShortCircuit(Protocol):

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        ...

    def simplify(self) -> CanShortCircuit:
        ...

    def simplify_knowing(self, assumption_state: AssumptionState) -> CanShortCircuit:
        """Simplify the rule knowing the state of other rules."""
        ...

    @property
    def short_circuit_able_component(self) -> Optional[CanShortCircuit]:
        """Return the combinable part of the rule."""
        ...

    def calculate_short_circuit_propagation(self, other: CanShortCircuit) -> ShortCircuitPropagation:
        """Return the link between two rules.
        NONE if there is no possible short-circuit propagation;
        POSITIVE if a True result from evaluating self short-circuit other;
        NEGATIVE if a False result from evaluating self short-circuit other;
        EQUAL if both resul;ts from evaluating self short-circuit other. In other words, both rules have the same combinable part.

        A POSITIVE or a NEGATIVE implies that other will short-circuit self in the opposite way.

        And/Or rules will always have a NONE link one another.
        """
        ...

    @property
    def lower_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        ...

    @property
    def upper_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        ...


# TODO split in two abstract class, one for BaseStardewRule, the other for Simplifiable / NonSimplifiable
class BaseStardewRule(StardewRule, CanShortCircuit, ABC):

    def __or__(self, other) -> StardewRule:
        if other is true_ or other is false_ or type(other) is Or:
            return other | self

        return Or(self, other)

    def __and__(self, other) -> StardewRule:
        if other is true_ or other is false_ or type(other) is And:
            return other & self

        return And(self, other)

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[BaseStardewRule, bool]:
        return self, self(state)

    def simplify(self) -> BaseStardewRule:
        return self

    def simplify_knowing(self, assumption_state: AssumptionState) -> BaseStardewRule:
        return self

    @property
    def short_circuit_able_component(self) -> Optional[BaseStardewRule]:
        return None

    def calculate_short_circuit_propagation(self, other: CanShortCircuit) -> ShortCircuitPropagation:
        return ShortCircuitPropagation.NONE

    @property
    def lower_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return ()

    @property
    def upper_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return ()


class CombinableStardewRule(BaseStardewRule, ABC):

    @property
    @abstractmethod
    def combination_key(self) -> Hashable:
        raise NotImplementedError

    @property
    @abstractmethod
    def value(self):
        raise NotImplementedError

    def is_same_rule(self, other: CombinableStardewRule):
        return self.combination_key == other.combination_key

    def add_into(self, rules: Dict[Hashable, CombinableStardewRule], reducer: Callable[[CombinableStardewRule, CombinableStardewRule], CombinableStardewRule]) \
            -> Dict[Hashable, CombinableStardewRule]:
        rules = dict(rules)

        if self.combination_key in rules:
            rules[self.combination_key] = reducer(self, rules[self.combination_key])
        else:
            rules[self.combination_key] = self

        return rules

    def __and__(self, other):
        if isinstance(other, CombinableStardewRule) and self.is_same_rule(other):
            return And.combine(self, other)
        return super().__and__(other)

    def __or__(self, other):
        if isinstance(other, CombinableStardewRule) and self.is_same_rule(other):
            return Or.combine(self, other)
        return super().__or__(other)

    @property
    def short_circuit_able_component(self) -> Optional[CanShortCircuit]:
        return self

    def calculate_short_circuit_propagation(self, other: CanShortCircuit) -> ShortCircuitPropagation:
        if not isinstance(other, CombinableStardewRule):
            return other.calculate_short_circuit_propagation(self)

        # Different key means nothing in common, so no short-circuit propagation.
        if self.combination_key != other.combination_key:
            return ShortCircuitPropagation.NONE

        # Both have same value, so rule is in fact the same.
        if self.value == other.value:
            return ShortCircuitPropagation.EQUAL

        # Self has a higher value, so self evaluating to True mean that other will be True was well.
        if self.value > other.value:
            return ShortCircuitPropagation.POSITIVE

        # Self has a lower value, so self evaluating to False mean that other will be False was well.
        return ShortCircuitPropagation.NEGATIVE

    def simplify_knowing(self, assumption_state: AssumptionState) -> StardewRule:
        try:
            (lower_bound, upper_bound) = assumption_state.combinable_values[self.combination_key]
            if self.value <= lower_bound:
                return true_
            if upper_bound <= self.value:
                return false_
            return self
        except KeyError:
            return self

    @cached_property
    def lower_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return ((self.combination_key, self.value),)

    @cached_property
    def upper_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return ((self.combination_key, self.value),)


class _SimplificationState:
    original_simplifiable_rules: Tuple[StardewRule, ...]

    rules_to_simplify: deque[StardewRule]
    simplified_rules: Set[StardewRule]
    lock: Lock

    def __init__(self, simplifiable_rules: Tuple[StardewRule, ...], rules_to_simplify: Optional[deque[StardewRule]] = None,
                 simplified_rules: Optional[Set[StardewRule]] = None):
        if simplified_rules is None:
            simplified_rules = set()

        self.original_simplifiable_rules = simplifiable_rules
        self.rules_to_simplify = rules_to_simplify
        self.simplified_rules = simplified_rules
        self.locked = False

    @property
    def is_simplified(self):
        return self.rules_to_simplify is not None and not self.rules_to_simplify

    def short_circuit(self, complement: LiteralStardewRule):
        self.rules_to_simplify = deque()
        self.simplified_rules = {complement}

    def try_popleft(self):
        try:
            self.rules_to_simplify.popleft()
        except IndexError:
            pass

    def acquire_copy(self):
        state = _SimplificationState(self.original_simplifiable_rules, self.rules_to_simplify.copy(), self.simplified_rules.copy())
        state.acquire()
        return state

    def merge(self, other: _SimplificationState):
        return _SimplificationState(self.original_simplifiable_rules + other.original_simplifiable_rules)

    def add(self, rule: StardewRule):
        return _SimplificationState(self.original_simplifiable_rules + (rule,))

    def acquire(self):
        """
        This just set a boolean to True and is absolutely not thread safe. It just works because AP is single-threaded.
        """
        if self.locked is True:
            return False

        self.locked = True
        return True

    def release(self):
        assert self.locked
        self.locked = False


class AggregatingStardewRule(BaseStardewRule, ABC):
    """
    Logic for both "And" and "Or" rules.
    """
    identity: ClassVar[LiteralStardewRule]
    complement: ClassVar[LiteralStardewRule]
    symbol: ClassVar[str]

    combinable_rules: Dict[Hashable, CombinableStardewRule]
    simplification_state: _SimplificationState
    _last_short_circuiting_rule: Optional[StardewRule] = None

    def __init__(self, *rules: StardewRule, _combinable_rules=None, _simplification_state=None):
        if _combinable_rules is None:
            assert rules, f"Can't create an aggregating condition without rules"
            rules, _combinable_rules = self.split_rules(rules)
            _simplification_state = _SimplificationState(rules)

        self.combinable_rules = _combinable_rules
        self.simplification_state = _simplification_state

    @property
    def original_rules(self):
        return RepeatableChain(self.combinable_rules.values(), self.simplification_state.original_simplifiable_rules)

    @property
    def current_rules(self):
        if self.simplification_state.rules_to_simplify is None:
            return self.original_rules

        return RepeatableChain(self.combinable_rules.values(), self.simplification_state.simplified_rules, self.simplification_state.rules_to_simplify)

    @classmethod
    def split_rules(cls, rules: Union[Iterable[StardewRule]]) -> Tuple[Tuple[StardewRule, ...], Dict[Hashable, CombinableStardewRule]]:
        other_rules = []
        reduced_rules = {}
        for rule in rules:
            if isinstance(rule, CombinableStardewRule):
                key = rule.combination_key
                if key not in reduced_rules:
                    reduced_rules[key] = rule
                    continue

                reduced_rules[key] = cls.combine(reduced_rules[key], rule)
                continue

            if type(rule) is cls:
                other_rules.extend(rule.simplification_state.original_simplifiable_rules)  # noqa
                reduced_rules = cls.merge(reduced_rules, rule.combinable_rules)  # noqa
                continue

            other_rules.append(rule)

        return tuple(other_rules), reduced_rules

    @classmethod
    def merge(cls, left: Dict[Hashable, CombinableStardewRule], right: Dict[Hashable, CombinableStardewRule]) -> Dict[Hashable, CombinableStardewRule]:
        reduced_rules = dict(left)
        for key, rule in right.items():
            if key not in reduced_rules:
                reduced_rules[key] = rule
                continue

            reduced_rules[key] = cls.combine(reduced_rules[key], rule)

        return reduced_rules

    @staticmethod
    @abstractmethod
    def combine(left: CombinableStardewRule, right: CombinableStardewRule) -> CombinableStardewRule:
        raise NotImplementedError

    def short_circuit_simplification(self):
        self.simplification_state.short_circuit(self.complement)
        self.combinable_rules = {}
        return self.complement, self.complement.value

    def short_circuit_evaluation(self, rule):
        self._last_short_circuiting_rule = rule
        return self, self.complement.value

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        """
        The global idea here is the same as short-circuiting operators, applied to evaluation and rule simplification.
        """

        # Directly checking last rule that short-circuited, in case state has not changed.
        if self._last_short_circuiting_rule:
            if self._last_short_circuiting_rule(state) is self.complement.value:
                return self.short_circuit_evaluation(self._last_short_circuiting_rule)
            self._last_short_circuiting_rule = None

        # Combinable rules are considered already simplified, so we evaluate them right away to go faster.
        for rule in self.combinable_rules.values():
            if rule(state) is self.complement.value:
                return self.short_circuit_evaluation(rule)

        if self.simplification_state.is_simplified:
            # The rule is fully simplified, so now we can only evaluate.
            for rule in self.simplification_state.simplified_rules:
                if rule(state) is self.complement.value:
                    return self.short_circuit_evaluation(rule)
            return self, self.identity.value

        return self.evaluate_while_simplifying_stateful(state)

    def evaluate_while_simplifying_stateful(self, state):
        local_state = self.simplification_state
        try:
            # Creating a new copy, so we don't modify the rules while we're already evaluating it. This can happen if a rule is used for an entrance and a
            # location. When evaluating a given rule what requires access to a region, the region cache can get an update. If it does, we could enter this rule
            # again. Since the simplification is stateful, the set of simplified rules can be modified while it's being iterated on, and cause a crash.
            #
            # After investigation, for millions of call to this method, copy were acquired 425 times.
            # Merging simplification state in parent call was deemed useless.
            if not local_state.acquire():
                local_state = local_state.acquire_copy()
                self.simplification_state = local_state

            # Evaluating what has already been simplified. First it will be faster than simplifying "new" rules, but we also assume that if we reach this point
            # and there are already are simplified rule, one of these rules has short-circuited, and might again, so we can leave early.
            for rule in local_state.simplified_rules:
                if rule(state) is self.complement.value:
                    return self.short_circuit_evaluation(rule)

            # If the queue is None, it means we have not start simplifying. Otherwise, we will continue simplification where we left.
            if local_state.rules_to_simplify is None:
                rules_to_simplify = frozenset(local_state.original_simplifiable_rules)
                if self.complement in rules_to_simplify:
                    return self.short_circuit_simplification()
                local_state.rules_to_simplify = deque(rules_to_simplify)

            # Start simplification where we left.
            while local_state.rules_to_simplify:
                result = self.evaluate_rule_while_simplifying_stateful(local_state, state)
                local_state.try_popleft()
                if result is not None:
                    return result

            # The whole rule has been simplified and evaluated without short-circuit.
            return self, self.identity.value
        finally:
            local_state.release()

    def evaluate_rule_while_simplifying_stateful(self, local_state, state):
        simplified, value = local_state.rules_to_simplify[0].evaluate_while_simplifying(state)

        # Identity is removed from the resulting simplification since it does not affect the result.
        if simplified is self.identity:
            return

        # If we find a complement here, we know the rule will always short-circuit, what ever the state.
        if simplified is self.complement:
            return self.short_circuit_simplification()
        # Keep the simplified rule to be reevaluated later.
        local_state.simplified_rules.add(simplified)

        # Now we use the value to short-circuit if it is the complement.
        if value is self.complement.value:
            return self.short_circuit_evaluation(simplified)

    # TODO str should be original rules, repr should be current simplified rules.
    def __str__(self):
        return f"({self.symbol.join(str(rule) for rule in self.original_rules)})"

    def __repr__(self):
        return f"({self.symbol.join(repr(rule) for rule in self.original_rules)})"

    def __eq__(self, other):
        return (isinstance(other, type(self)) and self.combinable_rules == other.combinable_rules and
                self.simplification_state.original_simplifiable_rules == other.simplification_state.original_simplifiable_rules)

    def __hash__(self):
        if len(self.combinable_rules) + len(self.simplification_state.original_simplifiable_rules) > 5:
            return id(self)

        return hash((*self.combinable_rules.values(), self.simplification_state.original_simplifiable_rules))

    def simplify(self) -> StardewRule:
        simplified_rules = []
        assumption_state = AssumptionState()
        for rule in self.current_rules:
            if rule is self.complement:
                return self.complement

            # Assuming all identity have already been removed when the rule was created originally.

            # TODO use simplify knowing, add the assumption that other rules are short-circuited.
            simplified_rule = rule.simplify()

            if simplified_rule is self.identity:
                continue

            if simplified_rule is self.complement:
                return self.complement

            simplified_rules.append(simplified_rule)

        # The process of creating a new rule will handle merging of aggregating and combinable rules.
        return type(self)(*simplified_rules)

    def simplify_knowing(self, assumption_state: AssumptionState) -> StardewRule:
        combinable_rules = {}
        for key, rule in self.combinable_rules.items():
            simplified_rule = rule.simplify_knowing(assumption_state)

            if simplified_rule is self.complement:
                return self.complement

            if simplified_rule is self.identity:
                continue

            combinable_rules[key] = rule.simplify_knowing(assumption_state)

        if not combinable_rules:
            assert self.simplification_state.original_simplifiable_rules
            if len(self.simplification_state.original_simplifiable_rules) == 1:
                return next(iter(self.simplification_state.original_simplifiable_rules))

        return type(self)(_combinable_rules=combinable_rules, _simplification_state=self.simplification_state)


class Or(AggregatingStardewRule):
    identity = false_
    complement = true_
    symbol = " | "

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate_while_simplifying(state)[1]

    def __or__(self, other):
        if other is true_ or other is false_:
            return other | self

        if isinstance(other, CombinableStardewRule):
            return Or(_combinable_rules=other.add_into(self.combinable_rules, self.combine), _simplification_state=self.simplification_state)

        if type(other) is Or:
            other = cast(Or, other)
            return Or(_combinable_rules=self.merge(self.combinable_rules, other.combinable_rules),
                      _simplification_state=self.simplification_state.merge(other.simplification_state))

        return Or(_combinable_rules=self.combinable_rules, _simplification_state=self.simplification_state.add(other))

    @staticmethod
    def combine(left: CombinableStardewRule, right: CombinableStardewRule) -> CombinableStardewRule:
        return min(left, right, key=lambda x: x.value)

    @property
    def short_circuit_able_component(self) -> Optional[CanShortCircuit]:
        if not self.combinable_rules:
            return None

        if len(self.combinable_rules) == 1:
            return next(iter(self.combinable_rules.values()))

        return Or(_combinable_rules=self.combinable_rules, _simplification_state=_SimplificationState(()))

    def calculate_short_circuit_propagation(self, other: CanShortCircuit) -> ShortCircuitPropagation:
        # TODO see that later
        raise NotImplementedError("hey maybe you should implement or short circuit propagation...")

    @cached_property
    def upper_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return RepeatableChain(rule.upper_bounds for rule in self.combinable_rules.values())


class And(AggregatingStardewRule):
    identity = true_
    complement = false_
    symbol = " & "

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate_while_simplifying(state)[1]

    def __and__(self, other):
        if other is true_ or other is false_:
            return other & self

        if isinstance(other, CombinableStardewRule):
            return And(_combinable_rules=other.add_into(self.combinable_rules, self.combine), _simplification_state=self.simplification_state)

        if type(other) is And:
            other = cast(And, other)
            return And(_combinable_rules=self.merge(self.combinable_rules, other.combinable_rules),
                       _simplification_state=self.simplification_state.merge(other.simplification_state))

        return And(_combinable_rules=self.combinable_rules, _simplification_state=self.simplification_state.add(other))

    @staticmethod
    def combine(left: CombinableStardewRule, right: CombinableStardewRule) -> CombinableStardewRule:
        return max(left, right, key=lambda x: x.value)

    @property
    def short_circuit_able_component(self) -> Optional[CanShortCircuit]:
        if not self.combinable_rules:
            return None

        if len(self.combinable_rules) == 1:
            return next(iter(self.combinable_rules.values()))

        return And(_combinable_rules=self.combinable_rules, _simplification_state=_SimplificationState(()))

    def calculate_short_circuit_propagation(self, other: CanShortCircuit) -> ShortCircuitPropagation:
        if isinstance(other, CombinableStardewRule):
            return self.__calculate_short_circuit_propagation_combinable(other)
        elif isinstance(other, And):
            return self.__calculate_short_circuit_propagation_and(other)
        return ShortCircuitPropagation.NONE

    def __calculate_short_circuit_propagation_combinable(self, other: CombinableStardewRule):
        # Different key means not enough in common, so no short-circuit propagation.
        if other.combination_key not in self.combinable_rules:
            return ShortCircuitPropagation.NONE

        # Self has a higher value, meaning it is more restrictive than the other rule.
        value = self.combinable_rules[other.combination_key].value
        if value == other.value:
            return ShortCircuitPropagation.EQUAL

        if value > other.value:
            return ShortCircuitPropagation.POSITIVE

        # Self has a lower value, so self evaluating to False mean that other will be False was well.
        if len(self.combinable_rules) == 1 and value < other.value:
            return ShortCircuitPropagation.NEGATIVE

        # Values are diverging, so no short-circuit propagation.
        return ShortCircuitPropagation.NONE

    def __calculate_short_circuit_propagation_and(self, other: And):
        # No combinable rules, so no short-circuit propagation.
        if not self.combinable_rules or not other.combinable_rules:
            return ShortCircuitPropagation.NONE

        if self.combinable_rules == other.combinable_rules:
            return ShortCircuitPropagation.EQUAL

        # No intersection means rules are diverging, so no short-circuit propagation.
        intersection = self.combinable_rules.keys() & other.combinable_rules.keys()
        if not intersection:
            return ShortCircuitPropagation.NONE

        if len(intersection) == len(self.combinable_rules):
            smaller = self
            larger = other
        elif len(intersection) == len(other.combinable_rules):
            smaller = other
            larger = self
        else:
            # Both have different keys, which means rules are diverging. No short-circuit propagation.
            return ShortCircuitPropagation.NONE

        # larger has a higher value, meaning it is more restrictive than the other rule.
        more_restrictive = all(larger.combinable_rules[key].value >= smaller.combinable_rules[key].value for key in smaller.combinable_rules)
        if more_restrictive:
            if self is larger:
                return ShortCircuitPropagation.POSITIVE
            return ShortCircuitPropagation.NEGATIVE

        # self has a lower value, so it is less restrictive that other rule.
        less_restrictive = all(larger.combinable_rules[key].value <= smaller.combinable_rules[key].value for key in self.combinable_rules)
        if len(self.combinable_rules) == len(other.combinable_rules) and less_restrictive:
            if self is larger:
                return ShortCircuitPropagation.NEGATIVE
            return ShortCircuitPropagation.POSITIVE

        print("man I never thought this would happen...")
        # Self has a lower or diverging values, so self evaluating to False mean that other will be False was well.
        return ShortCircuitPropagation.NONE

    @cached_property
    def lower_bounds(self) -> Iterable[Tuple[Hashable, int]]:
        return RepeatableChain(rule.lower_bounds for rule in self.combinable_rules.values())


@dataclass(frozen=True)
class Has(BaseStardewRule):
    item: str
    # For sure there is a better way than just passing all the rules everytime
    other_rules: Dict[str, StardewRule] = field(repr=False, hash=False, compare=False)
    group: str = "item"

    def __call__(self, state: CollectionState) -> bool:
        return self.evaluate_while_simplifying(state)[1]

    def evaluate_while_simplifying(self, state: CollectionState) -> Tuple[StardewRule, bool]:
        return self.other_rules[self.item].evaluate_while_simplifying(state)

    def simplify(self) -> StardewRule:
        sub_rule = self.other_rules[self.item]
        if isinstance(sub_rule, LiteralStardewRule):
            return sub_rule
        return cast(BaseStardewRule, sub_rule).simplify()

    def __str__(self):
        if self.item not in self.other_rules:
            return f"Has {self.item} ({self.group}) -> {MISSING_ITEM}"
        return f"Has {self.item} ({self.group})"

    def __repr__(self):
        if self.item not in self.other_rules:
            return f"Has {self.item} ({self.group}) -> {MISSING_ITEM}"
        return f"Has {self.item} ({self.group}) -> {repr(self.other_rules[self.item])}"


class RepeatableChain(Iterable, Sized):
    """
    Essentially a copy of what's in the core, with proper type hinting
    """

    def __init__(self, *iterable: Union[Iterable, Sized]):
        self.iterables = iterable

    def __iter__(self):
        return chain.from_iterable(self.iterables)

    def __bool__(self):
        return any(sub_iterable for sub_iterable in self.iterables)

    def __len__(self):
        return sum(len(iterable) for iterable in self.iterables)

    def __contains__(self, item):
        return any(item in it for it in self.iterables)

    def __repr__(self):
        return f"RepeatableChain({', '.join(repr(i) for i in self.iterables)})"
