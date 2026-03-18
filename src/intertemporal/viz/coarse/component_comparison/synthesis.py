"""Circuit synthesis plots: information flow summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .....activation_patching.coarse import SweepStepResults
from .utils import save_plot


def plot_synthesis(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Generate circuit synthesis plots."""
    _plot_information_flow_diagram(layer_data, pos_data, output_dir)


def _plot_information_flow_diagram(
    layer_data: dict[str, SweepStepResults | None],
    pos_data: dict[str, SweepStepResults | None],
    output_dir: Path,
) -> None:
    """Create information flow summary diagram synthesizing all findings."""
    # Extract key findings
    input_positions = []
    output_positions = []

    resid_post_pos = pos_data.get("resid_post", {})
    if resid_post_pos:
        positions = sorted(resid_post_pos.keys())
        pos_scores = [(p, resid_post_pos[p].recovery or 0) for p in positions]
        pos_scores.sort(key=lambda x: x[1], reverse=True)

        mid_pos = positions[len(positions) // 2] if positions else 0
        input_positions = [(p, s) for p, s in pos_scores[:10] if p < mid_pos][:3]
        output_positions = [(p, s) for p, s in pos_scores[:10] if p >= mid_pos][:3]

    # Find top layers
    attn_data = layer_data.get("attn_out", {})
    mlp_data = layer_data.get("mlp_out", {})

    top_attn_layers = []
    top_mlp_layers = []

    if attn_data:
        attn_scores = [(lyr, attn_data[lyr].recovery or 0) for lyr in attn_data.keys()]
        attn_scores.sort(key=lambda x: x[1], reverse=True)
        top_attn_layers = attn_scores[:5]

    if mlp_data:
        mlp_scores = [(lyr, mlp_data[lyr].recovery or 0) for lyr in mlp_data.keys()]
        mlp_scores.sort(key=lambda x: x[1], reverse=True)
        top_mlp_layers = mlp_scores[:5]

    # Create diagram
    fig, ax = plt.subplots(figsize=(14, 10), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(5, 7.5, "Information Flow Summary", fontsize=18, fontweight="bold", ha="center", va="center")

    # Input positions box
    rect1 = Rectangle((0.5, 5), 2.5, 2, facecolor="#E3F2FD", edgecolor="black", linewidth=2)
    ax.add_patch(rect1)
    ax.text(1.75, 6.7, "Input Positions", fontsize=12, fontweight="bold", ha="center")
    if input_positions:
        for i, (pos, score) in enumerate(input_positions[:3]):
            ax.text(1.75, 6.2 - i * 0.4, f"P{pos}: {score:.2f}", fontsize=10, ha="center")
    else:
        ax.text(1.75, 6.0, "(analyzing...)", fontsize=10, ha="center", style="italic")

    # Attention layers box
    rect2 = Rectangle((3.5, 5), 2.5, 2, facecolor="#FFF3E0", edgecolor="black", linewidth=2)
    ax.add_patch(rect2)
    ax.text(4.75, 6.7, "Attention Layers", fontsize=12, fontweight="bold", ha="center")
    if top_attn_layers:
        for i, (layer, score) in enumerate(top_attn_layers[:3]):
            ax.text(4.75, 6.2 - i * 0.4, f"L{layer}: {score:.2f}", fontsize=10, ha="center")
    else:
        ax.text(4.75, 6.0, "(no data)", fontsize=10, ha="center", style="italic")

    # MLP layers box
    rect3 = Rectangle((3.5, 2), 2.5, 2, facecolor="#E8F5E9", edgecolor="black", linewidth=2)
    ax.add_patch(rect3)
    ax.text(4.75, 3.7, "MLP Layers", fontsize=12, fontweight="bold", ha="center")
    if top_mlp_layers:
        for i, (layer, score) in enumerate(top_mlp_layers[:3]):
            ax.text(4.75, 3.2 - i * 0.4, f"L{layer}: {score:.2f}", fontsize=10, ha="center")
    else:
        ax.text(4.75, 3.0, "(no data)", fontsize=10, ha="center", style="italic")

    # Output positions box
    rect4 = Rectangle((7, 3.5), 2.5, 2, facecolor="#FCE4EC", edgecolor="black", linewidth=2)
    ax.add_patch(rect4)
    ax.text(8.25, 5.2, "Output Positions", fontsize=12, fontweight="bold", ha="center")
    if output_positions:
        for i, (pos, score) in enumerate(output_positions[:3]):
            ax.text(8.25, 4.7 - i * 0.4, f"P{pos}: {score:.2f}", fontsize=10, ha="center")
    else:
        ax.text(8.25, 4.5, "(analyzing...)", fontsize=10, ha="center", style="italic")

    # Arrows
    arrow_style = dict(arrowstyle="->", color="black", lw=2)

    ax.annotate("", xy=(3.5, 6), xytext=(3.0, 6), arrowprops=arrow_style)
    ax.annotate("", xy=(4.75, 4), xytext=(4.75, 5), arrowprops=arrow_style)
    ax.annotate("", xy=(7, 4.5), xytext=(6, 3), arrowprops=arrow_style)
    ax.annotate("", xy=(7, 5), xytext=(6, 6), arrowprops=dict(arrowstyle="->", color="gray", lw=1.5, ls="--"))

    # Legend
    ax.text(5, 0.8, "Solid arrows: primary information flow", fontsize=9, ha="center", color="black")
    ax.text(5, 0.4, "Dashed arrows: potential bypass paths", fontsize=9, ha="center", color="gray")

    # Summary stats
    if top_attn_layers and top_mlp_layers:
        attn_total = sum(s for _, s in top_attn_layers[:3])
        mlp_total = sum(s for _, s in top_mlp_layers[:3])
        ax.text(0.5, 1.5, f"Top-3 Attn Total: {attn_total:.2f}", fontsize=10, ha="left")
        ax.text(0.5, 1.1, f"Top-3 MLP Total: {mlp_total:.2f}", fontsize=10, ha="left")

    plt.tight_layout()
    save_plot(fig, output_dir, "information_flow_diagram.png")
