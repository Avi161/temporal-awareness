"""ContrastivePrefReq: requirements for filtering ContrastivePreferences pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...common.base_schema import BaseSchema

log = logging.getLogger(__name__)
from .contrastive_preferences import ContrastivePreferences
from .preference_types import PreferenceSample

if TYPE_CHECKING:
    from .contrastive_preferences import ContrastivePreferences
    from ..preference import PreferenceDataset


@dataclass
class ContrastivePrefReq(BaseSchema):
    """Requirements for filtering ContrastivePreferences pairs.

    All fields default to False (no requirement). Set to True to require
    the corresponding property on ContrastivePreferences.
    """

    # Label requirements (both False = no requirement, allows multilabel pairing)
    same_labels: bool = False
    different_labels: bool = False

    # Reward requirements
    same_rewards: bool = False
    different_rewards: bool = False

    # Time requirements
    same_times: bool = False
    different_times: bool = False

    # Horizon requirements
    same_horizon: bool = False
    different_horizon: bool = False
    neither_horizon: bool = False
    both_horizon: bool = False
    only_short_horizon: bool = False
    only_long_horizon: bool = False
    only_one_horizon: bool = False

    # Rational choice requirements
    both_rational: bool = False
    neither_rational: bool = False
    only_short_rational: bool = False
    only_long_rational: bool = False
    only_one_rational: bool = False

    # Associated choice requirements
    both_associated: bool = False
    neither_associated: bool = False
    only_short_associated: bool = False
    only_long_associated: bool = False
    only_one_associated: bool = False

    def verify(self) -> None:
        """Verify requirements are logically consistent.

        Raises:
            ValueError: If requirements are contradictory.
        """
        errors = []

        # Check mutually exclusive pairs
        exclusive_pairs = [
            ("same_labels", "different_labels"),
            ("same_rewards", "different_rewards"),
            ("same_times", "different_times"),
            ("same_horizon", "different_horizon"),
            ("both_rational", "neither_rational"),
            ("both_associated", "neither_associated"),
        ]
        for a, b in exclusive_pairs:
            if getattr(self, a) and getattr(self, b):
                errors.append(f"{a} and {b} are mutually exclusive")

        # Horizon: mutually exclusive groups
        horizon_exclusive = [
            "neither_horizon",
            "both_horizon",
            "only_short_horizon",
            "only_long_horizon",
        ]
        active_horizon = [h for h in horizon_exclusive if getattr(self, h)]
        if len(active_horizon) > 1:
            errors.append(
                f"Horizon requirements are mutually exclusive: {active_horizon}"
            )

        # same_horizon requires both_horizon (can't be same if one is missing)
        if self.same_horizon and self.neither_horizon:
            errors.append("same_horizon requires both samples to have horizons")
        if self.same_horizon and self.only_one_horizon:
            errors.append("same_horizon is incompatible with only_one_horizon")
        if self.same_horizon and (self.only_short_horizon or self.only_long_horizon):
            errors.append("same_horizon is incompatible with only_short/long_horizon")

        # different_horizon with neither_horizon is impossible
        if self.different_horizon and self.neither_horizon:
            errors.append(
                "different_horizon requires at least one sample to have horizon"
            )

        # Rational: mutually exclusive groups
        rational_exclusive = [
            "both_rational",
            "neither_rational",
            "only_short_rational",
            "only_long_rational",
        ]
        active_rational = [r for r in rational_exclusive if getattr(self, r)]
        if len(active_rational) > 1:
            errors.append(
                f"Rational requirements are mutually exclusive: {active_rational}"
            )

        # Associated: mutually exclusive groups
        associated_exclusive = [
            "both_associated",
            "neither_associated",
            "only_short_associated",
            "only_long_associated",
        ]
        active_associated = [a for a in associated_exclusive if getattr(self, a)]
        if len(active_associated) > 1:
            errors.append(
                f"Associated requirements are mutually exclusive: {active_associated}"
            )

        # only_one_X with specific only_short/long is redundant but not invalid
        # (only_one_horizon with only_short_horizon is just more specific)

        if errors:
            raise ValueError(f"Invalid ContrastivePrefReq: {'; '.join(errors)}")

    def passes(self, pair: ContrastivePreferences) -> bool:
        """Check if a ContrastivePreferences pair passes all requirements."""
        self.verify()

        def fail(req_name: str, req_val: bool, pair_val: bool) -> bool:
            print(
                f"\n\nPair {pair.short_term.sample_idx}/{pair.long_term.sample_idx} "
                f"\n\nfailed: {req_name}={req_val} but pair.{req_name}={pair_val}"
            )
            return False

        # Label checks
        if self.same_labels and not pair.same_labels:
            return fail("same_labels", self.same_labels, pair.same_labels)
        if self.different_labels and pair.same_labels:
            return fail("different_labels", self.different_labels, not pair.same_labels)

        # Reward checks
        if self.same_rewards and not pair.same_rewards:
            return fail("same_rewards", self.same_rewards, pair.same_rewards)
        if self.different_rewards and pair.same_rewards:
            return fail(
                "different_rewards", self.different_rewards, not pair.same_rewards
            )

        # Time checks
        if self.same_times and not pair.same_times:
            return fail("same_times", self.same_times, pair.same_times)
        if self.different_times and pair.same_times:
            return fail("different_times", self.different_times, not pair.same_times)

        # Horizon checks
        if self.same_horizon and not pair.same_horizon:
            return fail("same_horizon", self.same_horizon, pair.same_horizon)
        if self.different_horizon and pair.same_horizon:
            return fail(
                "different_horizon", self.different_horizon, not pair.same_horizon
            )
        if self.neither_horizon and not pair.neither_horizon:
            return fail("neither_horizon", self.neither_horizon, pair.neither_horizon)
        if self.both_horizon and not pair.both_horizon:
            return fail("both_horizon", self.both_horizon, pair.both_horizon)
        if self.only_short_horizon and not pair.only_short_horizon:
            return fail(
                "only_short_horizon", self.only_short_horizon, pair.only_short_horizon
            )
        if self.only_long_horizon and not pair.only_long_horizon:
            return fail(
                "only_long_horizon", self.only_long_horizon, pair.only_long_horizon
            )
        if self.only_one_horizon and not pair.only_one_horizon:
            return fail(
                "only_one_horizon", self.only_one_horizon, pair.only_one_horizon
            )

        # Rational checks
        if self.both_rational and not pair.both_rational:
            return fail("both_rational", self.both_rational, pair.both_rational)
        if self.neither_rational and not pair.neither_rational:
            return fail(
                "neither_rational", self.neither_rational, pair.neither_rational
            )
        if self.only_short_rational and not pair.only_short_rational:
            return fail(
                "only_short_rational",
                self.only_short_rational,
                pair.only_short_rational,
            )
        if self.only_long_rational and not pair.only_long_rational:
            return fail(
                "only_long_rational", self.only_long_rational, pair.only_long_rational
            )
        if self.only_one_rational and not pair.only_one_rational:
            return fail(
                "only_one_rational", self.only_one_rational, pair.only_one_rational
            )

        # Associated checks
        if self.both_associated and not pair.both_associated:
            return fail("both_associated", self.both_associated, pair.both_associated)
        if self.neither_associated and not pair.neither_associated:
            return fail(
                "neither_associated", self.neither_associated, pair.neither_associated
            )
        if self.only_short_associated and not pair.only_short_associated:
            return fail(
                "only_short_associated",
                self.only_short_associated,
                pair.only_short_associated,
            )
        if self.only_long_associated and not pair.only_long_associated:
            return fail(
                "only_long_associated",
                self.only_long_associated,
                pair.only_long_associated,
            )
        if self.only_one_associated and not pair.only_one_associated:
            return fail(
                "only_one_associated",
                self.only_one_associated,
                pair.only_one_associated,
            )

        log.debug(
            f"Pair {pair.short_term.sample_idx}/{pair.long_term.sample_idx} passed all requirements"
        )
        return True


def get_contrastive_preferences(
    dataset: PreferenceDataset,
    req: ContrastivePrefReq | None = None,
) -> list[ContrastivePreferences]:
    """Find pairs of samples that differ primarily by time_horizon with different choices.

    This function groups samples by their content (formatting_id and reward/time values)
    and finds pairs where:
    - One sample chose short_term and one chose long_term
    - The primary difference is the time_horizon (which affects rational choice)

    Args:
        dataset: PreferenceDataset containing samples to search
        req: Optional ContrastivePrefReq specifying filtering requirements

    Returns:
        List of ContrastivePreferences pairs
    """

    if req is None:
        req = ContrastivePrefReq()

    # Verify requirements are valid
    req.verify()

    # Group samples by content (same rewards, times, but potentially different time_horizon/labels)
    # Key: (short_reward, long_reward, short_time, long_time)
    # Note: formatting_id is NOT included so different label formats can pair for multilabel choice
    content_groups: dict[tuple, list[PreferenceSample]] = {}

    for pref in dataset.preferences:
        # Skip samples with unknown choice
        if pref.choice_term not in ("short_term", "long_term"):
            continue

        # Build content key - excludes formatting_id to allow multilabel pairing
        key = (
            pref.short_term_reward,
            pref.long_term_reward,
            pref.short_term_time,
            pref.long_term_time,
        )
        if key not in content_groups:
            content_groups[key] = []
        content_groups[key].append(pref)

    # Find pairs with different choices within each group
    pairs: list[ContrastivePreferences] = []
    total_short = 0
    total_long = 0
    total_candidates = 0
    total_passed = 0

    for key, samples in content_groups.items():
        # Separate by choice
        short_choosers = [s for s in samples if s.choice_term == "short_term"]
        long_choosers = [s for s in samples if s.choice_term == "long_term"]
        total_short += len(short_choosers)
        total_long += len(long_choosers)

        print(
            f"Content group {key}: {len(short_choosers)} short, {len(long_choosers)} long"
        )

        # Create pairs
        for short_sample in short_choosers:
            for long_sample in long_choosers:
                total_candidates += 1
                candidate_pair = ContrastivePreferences(
                    short_term=short_sample,
                    long_term=long_sample,
                )
                if req.passes(candidate_pair):
                    total_passed += 1
                    pairs.append(candidate_pair)

    print(
        f"Contrastive pairs: {total_short} short choosers, {total_long} long choosers, "
        f"{total_candidates} candidates, {total_passed} passed"
    )

    # Sort by minimum choice probability (highest confidence pairs first)
    pairs.sort(key=lambda p: p.min_choice_prob, reverse=True)

    assert len(pairs) != 0, "Bad config! No contrastive pairs pass requirements."

    return pairs


def get_factorial_doe() -> dict[str, ContrastivePrefReq]:
    """Get factorial design of experiments configurations.

    Returns a dictionary of named ContrastivePrefReq configurations for
    systematic exploration of contrastive pair characteristics.

    Returns:
        Dict mapping descriptive names to ContrastivePrefReq instances.
    """
    return {
        # Baseline: no special requirements
        "baseline": ContrastivePrefReq(),
        # Horizon variations
        "only_horizon_different": ContrastivePrefReq(
            same_rewards=True,
            same_times=True,
            different_horizon=True,
        ),
        "same_horizon": ContrastivePrefReq(
            same_horizon=True,
        ),
        "neither_horizon": ContrastivePrefReq(
            neither_horizon=True,
        ),
        "both_horizon": ContrastivePrefReq(
            both_horizon=True,
        ),
        "only_short_horizon": ContrastivePrefReq(
            only_short_horizon=True,
        ),
        "only_long_horizon": ContrastivePrefReq(
            only_long_horizon=True,
        ),
        # Reward variations
        "only_reward_different": ContrastivePrefReq(
            different_rewards=True,
            same_times=True,
            same_horizon=True,
        ),
        "same_rewards": ContrastivePrefReq(
            same_rewards=True,
        ),
        # Time variations
        "only_time_different": ContrastivePrefReq(
            same_rewards=True,
            different_times=True,
            same_horizon=True,
        ),
        "same_times": ContrastivePrefReq(
            same_times=True,
        ),
        # Rational choice variations
        "both_rational": ContrastivePrefReq(
            both_rational=True,
        ),
        "neither_rational": ContrastivePrefReq(
            neither_rational=True,
        ),
        "only_short_rational": ContrastivePrefReq(
            only_short_rational=True,
        ),
        "only_long_rational": ContrastivePrefReq(
            only_long_rational=True,
        ),
        # Associated choice variations
        "both_associated": ContrastivePrefReq(
            both_associated=True,
        ),
        "neither_associated": ContrastivePrefReq(
            neither_associated=True,
        ),
        "only_short_associated": ContrastivePrefReq(
            only_short_associated=True,
        ),
        "only_long_associated": ContrastivePrefReq(
            only_long_associated=True,
        ),
        # Rational + Associated combinations
        "both_rational_neither_associated": ContrastivePrefReq(
            both_rational=True,
            neither_associated=True,
        ),
        "neither_rational_both_associated": ContrastivePrefReq(
            neither_rational=True,
            both_associated=True,
        ),
        "both_rational_both_associated": ContrastivePrefReq(
            both_rational=True,
            both_associated=True,
        ),
        "neither_rational_neither_associated": ContrastivePrefReq(
            neither_rational=True,
            neither_associated=True,
        ),
        # Strict controlled experiments
        "strict_only_horizon_varies": ContrastivePrefReq(
            same_labels=True,
            same_rewards=True,
            same_times=True,
            different_horizon=True,
        ),
        "strict_only_reward_varies": ContrastivePrefReq(
            same_labels=True,
            different_rewards=True,
            same_times=True,
            same_horizon=True,
        ),
        "strict_only_time_varies": ContrastivePrefReq(
            same_labels=True,
            same_rewards=True,
            different_times=True,
            same_horizon=True,
        ),
    }
