"""Data extraction for aggregated visualization.

Extracts metrics across pairs and step sizes for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .....activation_patching.coarse import CoarseActPatchAggregatedResults
from .....activation_patching.act_patch_metrics import IntervenedChoiceMetrics
from .style import COLUMN_METRICS


@dataclass
class MetricSeries:
    """Single metric aggregated across pairs.

    Stores per-pair series aligned to a common x-axis grid.
    """

    metric_name: str
    x_values: list[int] = field(default_factory=list)
    per_pair_series: list[list[float | None]] = field(default_factory=list)
    mean_series: list[float] = field(default_factory=list)

    def compute_mean(self) -> None:
        """Compute mean_series from per_pair_series."""
        n_points = len(self.x_values)
        self.mean_series = []
        for x_idx in range(n_points):
            values = [
                series[x_idx]
                for series in self.per_pair_series
                if series[x_idx] is not None
            ]
            self.mean_series.append(sum(values) / len(values) if values else 0.0)


@dataclass
class ColumnData:
    """All metrics for one column (e.g., vocab)."""

    column_name: str
    metrics: list[MetricSeries] = field(default_factory=list)


def extract_column_data(
    agg_results: CoarseActPatchAggregatedResults,
    column_name: str,
    sweep_type: Literal["layer", "position"],
    clean_traj: Literal["short", "long"],
    mode: Literal["denoising", "noising"],
) -> ColumnData:
    """Extract metric data for a column, aggregated across pairs and step sizes.

    Args:
        agg_results: Aggregated coarse patching results
        column_name: Column name (core, probs, logits, fork, vocab, trajectory)
        sweep_type: "layer" or "position"
        clean_traj: "short" or "long" - which is treated as clean
        mode: "denoising" or "noising"

    Returns:
        ColumnData with metrics aligned to common x-axis
    """
    # Get metric names for this column
    metric_names = list(COLUMN_METRICS.get(column_name, []))

    # For core column, use disruption instead of recovery for noising mode
    if column_name == "core" and mode == "noising":
        metric_names = [
            "disruption" if m == "recovery" else m for m in metric_names
        ]

    # Collect all x-values across all samples and step sizes
    all_x_values: set[int] = set()
    step_sizes = (
        agg_results.layer_step_sizes
        if sweep_type == "layer"
        else agg_results.position_step_sizes
    )

    for sample_id, result in agg_results.by_sample.items():
        for step_size in step_sizes:
            sweep_results = (
                result.get_layer_results_for_step(step_size)
                if sweep_type == "layer"
                else result.get_position_results_for_step(step_size)
            )
            all_x_values.update(sweep_results.keys())

    x_values = sorted(all_x_values)
    if not x_values:
        return ColumnData(column_name=column_name)

    # Initialize metric series
    metric_series_map: dict[str, MetricSeries] = {
        name: MetricSeries(metric_name=name, x_values=x_values)
        for name in metric_names
    }

    # Extract data for each pair + step_size combination (each becomes one series)
    for sample_id, result in agg_results.by_sample.items():
        for step_size in step_sizes:
            sweep_results = (
                result.get_layer_results_for_step(step_size)
                if sweep_type == "layer"
                else result.get_position_results_for_step(step_size)
            )

            if not sweep_results:
                continue

            # Create one series for this pair+step combination
            for metric_name in metric_names:
                series: list[float | None] = []
                for x in x_values:
                    target_result = sweep_results.get(x)
                    if target_result is None:
                        series.append(None)
                        continue

                    # Get metrics for the appropriate perspective
                    if clean_traj == "short":
                        # short=clean: denoising recovers short, noising disrupts short
                        if mode == "denoising":
                            metrics = target_result.get_denoising_metrics()
                        else:
                            metrics = target_result.get_noising_metrics()
                    else:
                        # long=clean: need to switch perspective
                        # What was denoising (recover short) becomes noising (disrupt long)
                        switched = target_result.switch()
                        if mode == "denoising":
                            metrics = switched.get_denoising_metrics()
                        else:
                            metrics = switched.get_noising_metrics()

                    # Extract the metric value
                    value = getattr(metrics, metric_name, None)
                    series.append(value)

                metric_series_map[metric_name].per_pair_series.append(series)

    # Compute means
    for metric_series in metric_series_map.values():
        metric_series.compute_mean()

    return ColumnData(
        column_name=column_name,
        metrics=list(metric_series_map.values()),
    )
