#!/usr/bin/env python
"""Check flip coherence across models with full variation grid.

This script generates preference data for multiple models using
MINIMAL_PROMPT_DATASET_CONFIG with do_full_variation_grid=True,
then analyzes flip coherence to verify proper flip pair generation.

Usage:
  # Run all models and save results
  python check_flip_coherence.py --save results/

  # Re-run analysis on saved results
  python check_flip_coherence.py --load results/

  # Generate visualization only
  python check_flip_coherence.py --load results/ --viz
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal, Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.intertemporal.data.default_configs import MINIMAL_PROMPT_DATASET_CONFIG
from src.intertemporal.prompt import PromptDatasetConfig, PromptDatasetGenerator
from src.intertemporal.preference.preference_dataset import PreferenceDataset
from src.intertemporal.preference.preference_querier import (
    PreferenceQuerier,
    PreferenceQueryConfig,
)
from src.intertemporal.preference.preference_analysis import (
    analyze_preferences,
    get_simple_label_styles,
)
from src.intertemporal.formatting.formatting_variation import FormattingVariation


# =============================================================================
# Ad-hoc formatting metadata computation
# =============================================================================


@dataclass
class FormattingMetadata:
    """Formatting metadata computed ad-hoc for coherence analysis."""

    content_key: str
    is_flipped: bool
    has_time_unit_variation: bool
    has_spell_numbers: bool
    label_style: str


def compute_content_key(
    short_reward: float,
    short_time: float,
    long_reward: float,
    long_time: float,
    time_horizon: float | None,
) -> str:
    """Compute content key from reward/time values."""
    horizon_str = f"{time_horizon}" if time_horizon is not None else "none"
    return (
        f"sr{round(short_reward)}_st{short_time}_"
        f"lr{round(long_reward)}_lt{long_time}_"
        f"h{horizon_str}"
    )


def generate_variation_metadata(config: PromptDatasetConfig) -> dict[int, FormattingMetadata]:
    """Generate formatting metadata for each sample_idx using the same grid logic as the generator.

    Returns a dict mapping sample_idx -> FormattingMetadata.
    """
    # Recreate the grid logic from PromptDatasetGenerator
    gen = PromptDatasetGenerator(config)

    short_term_grid = gen.generate_option_grid("short_term")
    long_term_grid = gen.generate_option_grid("long_term")
    time_horizons_grid = config.time_horizons
    var_grid = gen.generate_formatting_variation_grid()

    full_grid = list(product(short_term_grid, long_term_grid, time_horizons_grid, var_grid))

    metadata = {}
    for i, (short_data, long_data, time_horizon, variation) in enumerate(full_grid):
        short_reward, short_time = short_data[0], short_data[1].to_years()
        long_reward, long_time = long_data[0], long_data[1].to_years()
        horizon_years = time_horizon.to_years() if time_horizon else None

        labels = variation.labels
        label_style = f"{labels[0]}_{labels[1]}"

        metadata[i] = FormattingMetadata(
            content_key=compute_content_key(
                short_reward, short_time, long_reward, long_time, horizon_years
            ),
            is_flipped=variation.flip_order,
            has_time_unit_variation=variation.time_unit_variation,
            has_spell_numbers=variation.spell_numbers,
            label_style=label_style,
        )

    return metadata


def get_pref_metadata(pref: Any, metadata_map: dict[int, FormattingMetadata]) -> FormattingMetadata | None:
    """Get formatting metadata for a preference sample."""
    return metadata_map.get(pref.sample_idx)


# Models to test (use backend:model format for API models)
MODELS = [
    "anthropic:claude-opus-4-20250514",
    "openai:gpt-4o",
    "Qwen/Qwen3.5-9B",
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-0.6B",
]


def log(msg: str = "") -> None:
    """Print with immediate flush."""
    print(msg, flush=True)


@dataclass
class FlipPairStats:
    """Statistics about flip pairs in a dataset."""

    total_content_variations: int = 0
    pairs_with_both_orderings: int = 0
    pairs_missing_flipped: int = 0
    pairs_missing_non_flipped: int = 0

    # By condition
    by_condition: dict[str, tuple[int, int, int]] = field(default_factory=dict)


@dataclass
class FlipCoherenceResult:
    """Flip coherence result for a model."""

    model_name: str
    n_samples: int
    n_flipped: int
    n_non_flipped: int

    # Overall coherence
    flip_coherent: int = 0
    flip_total: int = 0

    # By condition
    by_condition: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def flip_pct(self) -> float | None:
        if self.flip_total == 0:
            return None
        return 100 * self.flip_coherent / self.flip_total

    @property
    def short_name(self) -> str:
        name = self.model_name
        if ":" in name:
            name = name.split(":")[1]
        if "/" in name:
            name = name.split("/")[1]
        return name


def check_flip_pair_coverage(
    prompt_dataset, metadata_map: dict[int, FormattingMetadata]
) -> FlipPairStats:
    """Check how many content variations have both flip orderings."""
    stats = FlipPairStats()

    # Group by content (everything except is_flipped)
    groups = defaultdict(list)
    for sample in prompt_dataset.samples:
        meta = metadata_map.get(sample.sample_idx)
        if meta is None:
            continue
        content_key = (
            meta.content_key,
            meta.has_spell_numbers,
            meta.has_time_unit_variation,
            meta.label_style,
        )
        groups[content_key].append((sample, meta))

    stats.total_content_variations = len(groups)

    for content_key, sample_metas in groups.items():
        has_flipped = any(m.is_flipped for _, m in sample_metas)
        has_non_flipped = any(not m.is_flipped for _, m in sample_metas)

        if has_flipped and has_non_flipped:
            stats.pairs_with_both_orderings += 1
        elif has_flipped and not has_non_flipped:
            stats.pairs_missing_non_flipped += 1
        elif has_non_flipped and not has_flipped:
            stats.pairs_missing_flipped += 1

    # Check by condition
    conditions = [
        ("Simple, No Spell, No Unit", True, False, False),
        ("All, No Spell, No Unit", None, False, False),
        ("Simple, No Spell, Unit", True, False, True),
        ("All, No Spell, Unit", None, False, True),
        ("Simple, Spell, No Unit", True, True, False),
        ("All, Spell, No Unit", None, True, False),
        ("Simple, Spell, Unit", True, True, True),
        ("All, Spell, Unit", None, True, True),
    ]

    for name, simple_only, spell, unit in conditions:
        filtered_groups = defaultdict(list)
        for sample in prompt_dataset.samples:
            meta = metadata_map.get(sample.sample_idx)
            if meta is None:
                continue
            if simple_only is not None:
                is_simple = meta.label_style in get_simple_label_styles()
                if is_simple != simple_only:
                    continue
            if meta.has_spell_numbers != spell:
                continue
            if meta.has_time_unit_variation != unit:
                continue

            content_key = (meta.content_key, meta.label_style)
            filtered_groups[content_key].append((sample, meta))

        total = len(filtered_groups)
        both = 0
        missing = 0
        for key, sample_metas in filtered_groups.items():
            has_flipped = any(m.is_flipped for _, m in sample_metas)
            has_non_flipped = any(not m.is_flipped for _, m in sample_metas)
            if has_flipped and has_non_flipped:
                both += 1
            elif not has_flipped:
                missing += 1

        stats.by_condition[name] = (both, missing, total)

    return stats


def compute_flip_coherence(
    pref_dataset: PreferenceDataset, metadata_map: dict[int, FormattingMetadata]
) -> FlipCoherenceResult:
    """Compute flip coherence for a preference dataset."""
    prefs = pref_dataset.preferences

    # Build list of (pref, meta) tuples
    pref_metas = []
    for p in prefs:
        meta = metadata_map.get(p.sample_idx)
        if meta is not None:
            pref_metas.append((p, meta))

    n_flipped = sum(1 for _, m in pref_metas if m.is_flipped)
    n_non_flipped = sum(1 for _, m in pref_metas if not m.is_flipped)

    result = FlipCoherenceResult(
        model_name=pref_dataset.model,
        n_samples=len(prefs),
        n_flipped=n_flipped,
        n_non_flipped=n_non_flipped,
    )

    # Group by content (for flip coherence, we match on everything except is_flipped)
    groups = defaultdict(list)
    for p, meta in pref_metas:
        content_key = (
            meta.content_key,
            meta.has_spell_numbers,
            meta.has_time_unit_variation,
            meta.label_style,
        )
        groups[content_key].append((p, meta))

    # Compute overall flip coherence
    for key, samples in groups.items():
        flipped = [(p, m) for p, m in samples if m.is_flipped]
        non_flipped = [(p, m) for p, m in samples if not m.is_flipped]

        if flipped and non_flipped:
            for f_p, _ in flipped:
                for nf_p, _ in non_flipped:
                    result.flip_total += 1
                    # Coherent if both made same choice (both rational or both irrational)
                    if f_p.matches_rational == nf_p.matches_rational:
                        result.flip_coherent += 1

    # By condition
    conditions = [
        ("Simple, No Spell, No Unit", True, False, False),
        ("All, No Spell, No Unit", None, False, False),
        ("Simple, No Spell, Unit", True, False, True),
        ("All, No Spell, Unit", None, False, True),
        ("Simple, Spell, No Unit", True, True, False),
        ("All, Spell, No Unit", None, True, False),
        ("Simple, Spell, Unit", True, True, True),
        ("All, Spell, Unit", None, True, True),
    ]

    for name, simple_only, spell, unit in conditions:
        filtered = [
            (p, m)
            for p, m in pref_metas
            if m.has_spell_numbers == spell and m.has_time_unit_variation == unit
        ]
        if simple_only is not None:
            filtered = [
                (p, m)
                for p, m in filtered
                if (m.label_style in get_simple_label_styles()) == simple_only
            ]

        cond_groups = defaultdict(list)
        for p, m in filtered:
            key = (m.content_key, m.label_style)
            cond_groups[key].append((p, m))

        coherent = 0
        total = 0
        for key, samples in cond_groups.items():
            flipped = [(p, m) for p, m in samples if m.is_flipped]
            non_flipped = [(p, m) for p, m in samples if not m.is_flipped]
            if flipped and non_flipped:
                for f_p, _ in flipped:
                    for nf_p, _ in non_flipped:
                        total += 1
                        if f_p.matches_rational == nf_p.matches_rational:
                            coherent += 1

        result.by_condition[name] = (coherent, total)

    return result


def generate_preferences(model: str, prompt_dataset) -> PreferenceDataset:
    """Generate preferences for a model (local or API)."""
    log(f"  Loading model {model}...")

    config = PreferenceQueryConfig(
        max_new_tokens=512,
        temperature=0.0,
        subsample=1.0,
    )

    querier = PreferenceQuerier(config)
    log(f"  Querying {len(prompt_dataset.samples)} samples...")
    pref_data = querier.query_dataset(prompt_dataset, model)
    log(f"  Got {len(pref_data.preferences)} preferences")
    pref_data.pop_heavy()

    return pref_data


def get_model_filename(model: str) -> str:
    """Convert model name to safe filename."""
    return model.replace("/", "_").replace(":", "_") + ".json"


def save_preference_dataset(pref_dataset: PreferenceDataset, save_dir: Path) -> None:
    """Save a preference dataset to disk."""
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = get_model_filename(pref_dataset.model)
    filepath = save_dir / filename
    with open(filepath, "w") as f:
        json.dump(pref_dataset.to_dict(), f, indent=2)
    log(f"  Saved to {filepath}")


def load_preference_datasets(load_dir: Path) -> list[PreferenceDataset]:
    """Load all preference datasets from a directory."""
    datasets = []
    for filepath in sorted(load_dir.glob("*.json")):
        log(f"Loading {filepath.name}...")
        dataset = PreferenceDataset.from_json(filepath)
        datasets.append(dataset)
        log(f"  Loaded {len(dataset.preferences)} preferences for {dataset.model}")
    return datasets


def print_pair_coverage(stats: FlipPairStats) -> None:
    """Print flip pair coverage statistics."""
    log("\n" + "=" * 80)
    log("FLIP PAIR COVERAGE IN PROMPT DATASET")
    log("=" * 80)

    log(f"\nTotal content variations: {stats.total_content_variations}")
    log(f"Pairs with BOTH orderings: {stats.pairs_with_both_orderings}")
    log(f"Pairs missing flipped: {stats.pairs_missing_flipped}")
    log(f"Pairs missing non-flipped: {stats.pairs_missing_non_flipped}")

    if stats.pairs_missing_flipped > 0:
        pct_missing = 100 * stats.pairs_missing_flipped / stats.total_content_variations
        log(f"\nWARNING: {pct_missing:.1f}% of variations are missing flipped version!")
    else:
        log("\nALL variations have both flipped and non-flipped versions!")

    log("\nBy Condition:")
    log("-" * 60)
    log(f"{'Condition':<30} | {'Pairs':<8} | {'Missing':<8} | {'Total':<8}")
    log("-" * 60)
    for name, (both, missing, total) in stats.by_condition.items():
        log(f"{name:<30} | {both:<8} | {missing:<8} | {total:<8}")


def visualize_coherence_results(
    results: list[FlipCoherenceResult],
    output_dir: Path,
    save_format: Literal["png", "pdf"] = "png",
) -> None:
    """Generate visualizations for flip coherence results."""
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition mapping for cleaner column names
    condition_to_col = {
        "Simple, No Spell, No Unit": "Base-S",
        "All, No Spell, No Unit": "Base-A",
        "Simple, No Spell, Unit": "Unit-S",
        "All, No Spell, Unit": "Unit-A",
        "Simple, Spell, No Unit": "Spell-S",
        "All, Spell, No Unit": "Spell-A",
        "Simple, Spell, Unit": "Full-S",
        "All, Spell, Unit": "Full-A",
    }
    col_order = [
        "Base-S",
        "Base-A",
        "Unit-S",
        "Unit-A",
        "Spell-S",
        "Spell-A",
        "Full-S",
        "Full-A",
    ]

    # Build data matrix
    model_names = [r.short_name for r in results]
    data = []
    for r in results:
        row = []
        for cond_name in condition_to_col.keys():
            coherent, total = r.by_condition.get(cond_name, (0, 0))
            pct = 100 * coherent / total if total > 0 else np.nan
            row.append(pct)
        data.append(row)

    data = np.array(data)

    # Heatmap
    fig, ax = plt.subplots(figsize=(12, max(4, len(model_names) * 0.6)))

    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    # Labels
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right")
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(col_order)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if val < 50 or val > 80 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Flip Coherence (%)")

    ax.set_title("Flip Coherence by Model and Condition")

    # Legend box
    legend_text = (
        "Conditions:\n"
        "  Base: No variations\n"
        "  Unit: Time unit variation\n"
        "  Spell: Number spelling\n"
        "  Full: All variations\n"
        "\n"
        "Suffixes:\n"
        "  -S: Simple labels only (a/b, x/y, [i]/[ii], etc.)\n"
        "  -A: All label styles"
    )
    ax.text(
        1.02,
        0.02,
        legend_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    heatmap_path = output_dir / f"flip_coherence_heatmap.{save_format}"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved heatmap to: {heatmap_path}")

    # Bar chart by model
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    condition_groups = [
        ("Base (no variations)", ["Base-S", "Base-A"]),
        ("Unit (time unit variation)", ["Unit-S", "Unit-A"]),
        ("Spell (number spelling)", ["Spell-S", "Spell-A"]),
        ("Full (all variations)", ["Full-S", "Full-A"]),
    ]

    x = np.arange(len(model_names))
    width = 0.35

    for ax, (title, cols) in zip(axes, condition_groups):
        col_indices = [col_order.index(c) for c in cols]
        vals_s = data[:, col_indices[0]]
        vals_a = data[:, col_indices[1]]

        bars1 = ax.bar(
            x - width / 2, vals_s, width, label="Simple labels", color="#5DA5DA"
        )
        bars2 = ax.bar(
            x + width / 2, vals_a, width, label="All labels", color="#FAA43A"
        )

        ax.set_ylabel("Flip Coherence (%)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Chance")
        ax.axhline(y=90, color="green", linestyle="--", alpha=0.5, label="Good")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    bars_path = output_dir / f"flip_coherence_bars.{save_format}"
    plt.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Saved bar charts to: {bars_path}")


def print_legend() -> None:
    """Print the condition legend to stdout."""
    log("\n" + "=" * 60)
    log("LEGEND")
    log("=" * 60)
    log("\nConditions:")
    log("  Base   : No variations (baseline)")
    log("  Unit   : Time unit variation (e.g., weeks vs days)")
    log("  Spell  : Number spelling variation (e.g., 50 vs fifty)")
    log("  Full   : All variations combined")
    log("\nLabel Suffixes:")
    log("  -S     : Simple labels only (4 styles: a/b, x/y, [i]/[ii], etc.)")
    log("  -A     : All label styles (9 styles including complex)")
    log()


def print_coherence_results(results: list[FlipCoherenceResult]) -> None:
    """Print flip coherence results for all models."""
    log("\n" + "=" * 80)
    log("FLIP COHERENCE BY MODEL (SUMMARY)")
    log("=" * 80)

    # Sort by overall flip coherence
    results = sorted(
        results,
        key=lambda r: r.flip_pct if r.flip_pct is not None else -1,
        reverse=True,
    )

    log(f"\n{'Model':<25} | {'Flipped':<8} | {'Non-Flip':<8} | {'Flip Coherence':<15}")
    log("-" * 70)
    for r in results:
        flip_str = f"{r.flip_coherent}/{r.flip_total}" if r.flip_total > 0 else "N/A"
        pct_str = f"({r.flip_pct:.1f}%)" if r.flip_pct is not None else ""
        log(
            f"{r.short_name:<25} | {r.n_flipped:<8} | {r.n_non_flipped:<8} | {flip_str} {pct_str}"
        )

    # By condition
    log("\n" + "=" * 80)
    log("FLIP COHERENCE BY CONDITION")
    log("=" * 80)

    conditions = list(results[0].by_condition.keys()) if results else []

    for cond in conditions:
        log(f"\n{cond}:")
        log("-" * 50)
        for r in results:
            coherent, total = r.by_condition.get(cond, (0, 0))
            if total > 0:
                pct = 100 * coherent / total
                log(f"  {r.short_name:<25}: {coherent}/{total} ({pct:.1f}%)")
            else:
                log(f"  {r.short_name:<25}: N/A")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check flip coherence across models")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to test (default: all)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check prompt dataset coverage, don't run models",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="results/flip_coherence",
        help="Directory to save preference datasets (default: results/flip_coherence)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving preference datasets",
    )
    parser.add_argument(
        "--load",
        type=str,
        default=None,
        help="Directory to load preference datasets from (skip model querying, runs viz)",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization generation when using --load",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Print the legend for condition abbreviations",
    )
    args = parser.parse_args()

    # Print legend if requested
    if args.legend:
        print_legend()
        if not args.load and not args.models:
            return

    # Generate prompt dataset
    log("=" * 80)
    log("GENERATING PROMPT DATASET")
    log("=" * 80)
    log("Config: MINIMAL_PROMPT_DATASET_CONFIG with do_full_variation_grid=True")

    config_dict = MINIMAL_PROMPT_DATASET_CONFIG.copy()
    config_dict["do_full_variation_grid"] = True

    config = PromptDatasetConfig.from_dict(config_dict)
    prompt_dataset = PromptDatasetGenerator(config).generate()
    log(f"Generated {len(prompt_dataset.samples)} prompts")

    # Generate formatting metadata (computed ad-hoc for coherence analysis)
    metadata_map = generate_variation_metadata(config)

    # Show sample distribution
    n_flipped = sum(1 for m in metadata_map.values() if m.is_flipped)
    n_non_flipped = len(metadata_map) - n_flipped
    log(f"  Flipped: {n_flipped}, Non-flipped: {n_non_flipped}")

    # Check flip pair coverage
    log("\nChecking flip pair coverage...")
    stats = check_flip_pair_coverage(prompt_dataset, metadata_map)
    print_pair_coverage(stats)

    # Handle --load mode: load saved datasets and run analysis only
    if args.load:
        load_dir = Path(args.load)
        if not load_dir.exists():
            log(f"ERROR: Load directory does not exist: {load_dir}")
            return

        log(f"\n{'=' * 80}")
        log(f"LOADING SAVED PREFERENCE DATASETS FROM: {load_dir}")
        log("=" * 80)

        pref_datasets = load_preference_datasets(load_dir)
        if not pref_datasets:
            log("ERROR: No preference datasets found in directory")
            return

        # Run analysis on loaded datasets
        results = []
        for pref_dataset in pref_datasets:
            log(f"\n{'=' * 80}")
            log(f"ANALYZING: {pref_dataset.model}")
            log("=" * 80)

            # Compute flip coherence
            log("  Computing flip coherence...")
            coherence = compute_flip_coherence(pref_dataset, metadata_map)
            results.append(coherence)

            # Print quick summary
            if coherence.flip_pct is not None:
                log(
                    f"  -> Flip coherence: {coherence.flip_coherent}/{coherence.flip_total} ({coherence.flip_pct:.1f}%)"
                )
            else:
                log("  -> Flip coherence: N/A")

            # Print individual analysis
            log("\n  Detailed analysis:")
            analysis = analyze_preferences(pref_dataset)
            analysis.print_summary()

        # Print combined results
        if results:
            print_coherence_results(results)
            print_legend()

            # Generate visualization by default for --load
            if not args.no_viz:
                log("\n" + "=" * 80)
                log("GENERATING VISUALIZATIONS")
                log("=" * 80)
                viz_dir = load_dir / "viz"
                visualize_coherence_results(results, viz_dir)

        log("\n" + "=" * 80)
        log("DONE!")
        log("=" * 80)
        return

    # Check only mode - don't run models
    if args.check_only:
        log("\n--check-only specified, skipping model runs.")
        return

    # Filter models
    models_to_run = list(MODELS)
    if args.models:
        models_to_run = [
            m for m in models_to_run if any(arg in m for arg in args.models)
        ]

    log(f"\nWill run {len(models_to_run)} models:")
    for model in models_to_run:
        log(f"  - {model}")

    # Setup save directory (default enabled)
    save_dir = None if args.no_save else Path(args.save)
    if save_dir:
        log(f"\nResults will be saved to: {save_dir}")

    # Generate preferences for each model
    results = []
    for i, model in enumerate(models_to_run):
        log(f"\n{'=' * 80}")
        log(f"MODEL {i + 1}/{len(models_to_run)}: {model}")
        log("=" * 80)

        try:
            pref_dataset = generate_preferences(model, prompt_dataset)

            # Save if requested
            if save_dir:
                save_preference_dataset(pref_dataset, save_dir)

            # Compute flip coherence
            log("  Computing flip coherence...")
            coherence = compute_flip_coherence(pref_dataset, metadata_map)
            results.append(coherence)

            # Print quick summary
            if coherence.flip_pct is not None:
                log(
                    f"  -> Flip coherence: {coherence.flip_coherent}/{coherence.flip_total} ({coherence.flip_pct:.1f}%)"
                )
            else:
                log("  -> Flip coherence: N/A")

            # Print individual analysis
            log("\n  Detailed analysis:")
            analysis = analyze_preferences(pref_dataset)
            analysis.print_summary()

        except Exception as e:
            log(f"ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Print combined results
    if results:
        print_coherence_results(results)
        print_legend()

        # Generate visualization if save_dir exists
        if save_dir and not args.no_viz:
            log("\n" + "=" * 80)
            log("GENERATING VISUALIZATIONS")
            log("=" * 80)
            viz_dir = save_dir / "viz"
            visualize_coherence_results(results, viz_dir)

    log("\n" + "=" * 80)
    log("DONE!")
    log("=" * 80)


if __name__ == "__main__":
    main()
