"""Main entry point for component comparison visualizations."""

from __future__ import annotations

from pathlib import Path

from .....activation_patching.coarse import CoarseActPatchResults
from .constants import (
    COMPONENTS,
    SUBDIR_DECOMP,
    SUBDIR_OVERVIEW,
    SUBDIR_REDUNDANCY,
    SUBDIR_SANITY,
    SUBDIR_SYNTHESIS,
)
from .decomposition import plot_decomposition
from .overview import plot_overview
from .redundancy import plot_redundancy
from .sanity import plot_sanity_checks
from .synthesis import plot_synthesis


def plot_all_component_comparisons(
    results_by_component: dict[str, CoarseActPatchResults],
    output_dir: Path,
    step_size: int = 1,
) -> None:
    """Generate all multi-component comparison plots organized by category.

    Directory structure:
        01_sanity_checks/     - Validation plots (check these first)
        02_overview/          - Big picture heatmaps
        03_component_decomp/  - Attention vs MLP analysis
        04_redundancy/        - Noising vs denoising comparison
        05_circuit_synthesis/ - Information flow summary

    Args:
        results_by_component: Dict mapping component name to its results
        output_dir: Directory to save plots
        step_size: Step size to use for extracting data
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    sanity_dir = output_dir / SUBDIR_SANITY
    overview_dir = output_dir / SUBDIR_OVERVIEW
    decomp_dir = output_dir / SUBDIR_DECOMP
    redundancy_dir = output_dir / SUBDIR_REDUNDANCY
    synthesis_dir = output_dir / SUBDIR_SYNTHESIS

    for d in [sanity_dir, overview_dir, decomp_dir, redundancy_dir, synthesis_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Extract layer and position data for each component
    layer_data = {}
    pos_data = {}
    for comp in COMPONENTS:
        if comp in results_by_component:
            result = results_by_component[comp]
            layer_data[comp] = result.get_layer_results_for_step(step_size)
            pos_data[comp] = result.get_position_results_for_step(step_size)

    if not layer_data:
        print("[viz] No component data available for comparison plots")
        return

    # Generate plots by category
    plot_sanity_checks(layer_data, sanity_dir)
    plot_overview(layer_data, pos_data, results_by_component, overview_dir)
    plot_decomposition(layer_data, pos_data, decomp_dir)
    plot_redundancy(layer_data, pos_data, redundancy_dir)
    plot_synthesis(layer_data, pos_data, synthesis_dir)

    # Generate README
    _generate_readme(output_dir)

    print(f"[viz] Component comparison plots saved to {output_dir}")


def _generate_readme(output_dir: Path) -> None:
    """Generate README.md documenting all plots."""
    readme_content = """# Component Comparison Plots

## Directory Structure

### 01_sanity_checks/
*Check these first. If they fail badly, everything else is suspect.*

| File | Description |
|------|-------------|
| resid_delta_sanity.png | Compares resid_pre[L+1] vs resid_post[L] (should match) |
| resid_delta_difference.png | Difference plot with +/-0.02 tolerance band |

### 02_overview/
*The big picture: "Where does the computation happen?"*

| File | Description |
|------|-------------|
| heatmap_layers_denoising.png | Layer x Component attribution (denoising recovery) |
| heatmap_layers_noising.png | Layer x Component attribution (noising disruption) |
| heatmap_layers_colnorm_denoising.png | Column-normalized heatmap (reveals attn/mlp fine structure) |
| heatmap_layers_colnorm_noising.png | Column-normalized heatmap (noising) |
| heatmap_positions_denoising.png | Position x Component attribution (denoising) |
| heatmap_positions_noising.png | Position x Component attribution (noising) |
| layer_position_heatmap.png | 2D localization map (layer x position interaction) |

### 03_component_decomp/
*Drilling in: "Is it attention or MLP? Which specific components?"*

| File | Description |
|------|-------------|
| attn_vs_mlp_layer.png | Attention vs MLP scatter by layer |
| attn_vs_mlp_position.png | Attention vs MLP scatter by position |
| attn_vs_mlp_paired.png | Paired scatter with arrows showing denoising->noising movement |
| component_importance_ranked.png | Top components ranked (denoising + noising bars) |
| cumulative_recovery.png | Cumulative recovery build-up through layers |
| marginal_contribution.png | Per-layer marginal contribution with secondary y-axis |
| position_component_interaction.png | Effect vs position for each component |
| position_interaction_zoomed.png | Zoomed panels for hub regions |

### 04_redundancy/
*Methodology comparison: "Can I trust the scores? What has backup pathways?"*

| File | Description |
|------|-------------|
| noise_vs_denoise_per_component_layer.png | Noising vs denoising scatter (4 components x layers) |
| noise_vs_denoise_per_component_position.png | Noising vs denoising scatter (4 components x positions) |
| redundancy_gap.png | Redundancy gap (disruption - recovery) per layer |
| redundancy_gap_sorted.png | Redundancy gap sorted by magnitude |
| difference_heatmap_layer.png | Redundancy gap heatmap (layers) |
| difference_heatmap_position.png | Redundancy gap heatmap (positions) |

### 05_circuit_synthesis/
*The punchline: synthesized information flow.*

| File | Description |
|------|-------------|
| information_flow_diagram.png | Circuit summary showing key positions, attention, and MLP layers |

## Interpretation Guide

- **AND region** (scatter plots): High in both denoising and noising -> necessary AND sufficient
- **OR region**: High denoising, low noising -> sufficient but has backup pathways
- **Necessity** (positive redundancy gap): Required for correct answer
- **Sufficiency** (negative redundancy gap): Helpful but replaceable

## Plot Style

- Sqrt color mapping for better resolution at later layers
- adjustText library used for label placement where available
- Uniform position spacing in heatmaps (no misleading compression)
- Column-normalized heatmaps reveal fine structure in attn/mlp
"""
    (output_dir / "README.md").write_text(readme_content)
    print(f"Saved: {output_dir / 'README.md'}")
