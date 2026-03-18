#!/usr/bin/env python
"""
Load a preference dataset and print analysis.

Usage:
    # Analyze a specific experiment
    uv run python scripts/intertemporal/analyze_choice.py deb6

    # Analyze with full path
    uv run python scripts/intertemporal/analyze_choice.py out/experiments/deb6

    # List available experiments
    uv run python scripts/intertemporal/analyze_choice.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.intertemporal.common import get_experiment_dir, get_pref_dataset_dir
from src.intertemporal.preference import PreferenceDataset
from src.intertemporal.preference.preference_analysis import analyze_preferences
from src.common.logging import log, log_banner


def find_preference_data(experiment_path: Path) -> Path | None:
    """Find preference data JSON in an experiment directory or pref_datasets."""
    # Look for any json files in the experiment directory
    json_files = list(experiment_path.glob("*.json"))
    pref_files = [f for f in json_files if "pref" in f.name.lower() or "preference" in f.name.lower()]
    if pref_files:
        return pref_files[0]

    # Check in preference_datasets directory
    pref_dir = get_pref_dataset_dir()
    if pref_dir.exists():
        # Get the most recent file
        pref_files = [f for f in pref_dir.glob("*.json") if f.is_file()]
        if pref_files:
            return max(pref_files, key=lambda p: p.stat().st_mtime)

    return None


def list_preference_datasets() -> list[Path]:
    """List all preference datasets."""
    pref_dir = get_pref_dataset_dir()
    if not pref_dir.exists():
        return []
    return sorted(pref_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def list_experiments() -> None:
    """List available experiments and preference datasets."""
    # List preference datasets
    log_banner("Preference Datasets")
    log()

    pref_files = list_preference_datasets()
    if pref_files:
        for i, pf in enumerate(pref_files[:10]):  # Show most recent 10
            size_mb = pf.stat().st_size / (1024 * 1024)
            log(f"  [{i+1}] {pf.name} ({size_mb:.1f} MB)")
        if len(pref_files) > 10:
            log(f"  ... and {len(pref_files) - 10} more")
    else:
        log("  No preference datasets found")
    log()

    # List experiments
    exp_dir = get_experiment_dir()
    log_banner("Experiments")
    log()

    if not exp_dir.exists():
        log("  No experiments directory found")
        return

    experiments = sorted(exp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not experiments:
        log("  No experiments found")
        return

    for exp in experiments[:10]:  # Show most recent 10
        if exp.is_dir():
            has_log = (exp / "log.txt").exists()
            status = "has log" if has_log else "no log"
            log(f"  {exp.name}: {status}")

    log()


def load_and_analyze(identifier: str, detailed: bool = True) -> None:
    """Load preference dataset and print analysis.

    Args:
        identifier: Can be:
            - A number (1-10) to select from recent preference datasets
            - An experiment name (e.g., 'deb6')
            - A full path to a JSON file
    """
    pref_file: Path | None = None

    # Check if it's a number (index into recent datasets)
    if identifier.isdigit():
        idx = int(identifier) - 1
        pref_files = list_preference_datasets()
        if 0 <= idx < len(pref_files):
            pref_file = pref_files[idx]
        else:
            log(f"Error: Invalid index {identifier}. Use --list to see available datasets.")
            sys.exit(1)

    # Check if it's a direct path to a JSON file
    elif identifier.endswith(".json"):
        pref_file = Path(identifier)
        if not pref_file.exists():
            log(f"Error: File not found: {pref_file}")
            sys.exit(1)

    # Otherwise, treat as experiment name
    else:
        if "/" in identifier or identifier.startswith("."):
            exp_path = Path(identifier)
        else:
            exp_path = get_experiment_dir() / identifier

        if exp_path.exists() and exp_path.is_dir():
            pref_file = find_preference_data(exp_path)
            if not pref_file:
                # Use most recent from preference_datasets
                pref_files = list_preference_datasets()
                if pref_files:
                    pref_file = pref_files[0]
                    log(f"Note: Using most recent preference dataset")

        if not pref_file:
            # Maybe it's a partial match on dataset filename
            pref_dir = get_pref_dataset_dir()
            if pref_dir.exists():
                matches = [f for f in pref_dir.glob("*.json") if identifier in f.name]
                if matches:
                    pref_file = matches[0]

    if not pref_file or not pref_file.exists():
        log(f"Error: Could not find preference data for '{identifier}'")
        log("Use --list to see available datasets")
        sys.exit(1)

    log(f"Loading: {pref_file.name}")
    size_mb = pref_file.stat().st_size / (1024 * 1024)
    log(f"Size: {size_mb:.1f} MB")
    log()

    # Load dataset (lightweight for analysis - much faster)
    dataset = PreferenceDataset.from_json_lightweight(pref_file)

    # Analyze and print
    analysis = analyze_preferences(dataset)
    if detailed:
        analysis.print_all()
    else:
        analysis.print_summary()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze preference dataset from an experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "experiment",
        nargs="?",
        help="Experiment name or path (e.g., 'deb6' or 'out/experiments/deb6')",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available experiments",
    )
    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="Print summary only (skip detailed breakdowns)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        list_experiments()
        return 0

    if not args.experiment:
        print("Error: Please specify an experiment name or use --list")
        print("Usage: uv run python scripts/intertemporal/analyze_choice.py <experiment>")
        return 1

    load_and_analyze(args.experiment, detailed=not args.summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
