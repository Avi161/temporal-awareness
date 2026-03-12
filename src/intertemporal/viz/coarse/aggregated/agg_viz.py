"""Main entry point for aggregated coarse patching visualization.

Creates structured output with per-metric-column plots organized by:
- Sweep type (layer, position)
- Pair grouping (same_labels, future: flipped_labels, group_labels)
- Perspective (short/long clean)
- Mode (denoising/noising)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .data_extraction import extract_column_data
from .metric_plots import plot_column
from .style import COLUMN_METRICS


def plot_aggregated_structured(
    result: CoarseActPatchAggregatedResults,
    output_dir: Path,
) -> None:
    """Create structured aggregated visualization.

    Directory structure:
        agg_layer_sweep/
          same_labels/
            short/                    # clean=short perspective
              denoising/
                core.png, probs.png, logits.png, fork.png, vocab.png, trajectory.png
              noising/
                (same 6 files)
            long/                     # clean=long perspective
              denoising/
              noising/
        agg_pos_sweep/
          (same structure)

    Args:
        result: Aggregated coarse patching results
        output_dir: Base output directory
    """
    output_dir = Path(output_dir)

    sweep_types: list[Literal["layer", "position"]] = ["layer", "position"]
    pair_groupings = ["same_labels"]  # Future: flipped_labels, group_labels
    perspectives: list[Literal["short", "long"]] = ["short", "long"]
    modes: list[Literal["denoising", "noising"]] = ["denoising", "noising"]
    columns = list(COLUMN_METRICS.keys())

    for sweep_type in sweep_types:
        sweep_dir_name = f"agg_{sweep_type}_sweep"

        for grouping in pair_groupings:
            for perspective in perspectives:
                for mode in modes:
                    # Build directory path
                    dir_path = (
                        output_dir
                        / sweep_dir_name
                        / grouping
                        / perspective
                        / mode
                    )
                    dir_path.mkdir(parents=True, exist_ok=True)

                    # Build title prefix (compact format)
                    sweep_label = "Layer" if sweep_type == "layer" else "Pos"
                    mode_label = "Denoise" if mode == "denoising" else "Noise"
                    title_prefix = (
                        f"{sweep_label} | {perspective.upper()}=clean | "
                        f"{mode_label} | n={result.n_samples}"
                    )

                    for column in columns:
                        # Extract data
                        column_data = extract_column_data(
                            result,
                            column,
                            sweep_type,
                            perspective,
                            mode,
                        )

                        if not column_data.metrics:
                            continue

                        # Plot
                        output_path = dir_path / f"{column}.png"
                        plot_column(
                            column_data,
                            output_path,
                            title_prefix,
                        )

    print(f"[viz] Aggregated structured plots saved to {output_dir}")
