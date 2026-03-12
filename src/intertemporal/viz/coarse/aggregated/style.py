"""Plot styling constants for aggregated visualization."""

from __future__ import annotations

from ..colors import METRIC_COLORS

# Individual pair line style - use original colors
PAIR_LINE_ALPHA = 0.15
PAIR_LINE_WIDTH = 0.8

# Mean line style
MEAN_LINE_WIDTH = 2.0
MEAN_LINE_ALPHA = 1.0

# Spread fill (showing distribution)
SPREAD_ALPHA = 0.25

# Data point markers on mean line
MEAN_MARKER = "o"
MEAN_MARKER_SIZE = 5

# Grid style
GRID_ALPHA = 0.5
GRID_LINE_WIDTH = 0.5

# Figure settings
SUBPLOT_WIDTH = 4
SUBPLOT_HEIGHT = 3
DPI = 150

# Title styles
TITLE_FONTSIZE = 11
TITLE_FONTWEIGHT = "bold"
AXIS_LABEL_FONTSIZE = 9
TICK_LABEL_FONTSIZE = 8

# Column definitions: column_name -> list of metric field names
COLUMN_METRICS = {
    "core": [
        "recovery",
        "logit_diff",
        "norm_logit_diff",
        "reciprocal_rank_short",
    ],
    "probs": [
        "prob_short",
        "prob_long",
        "logprob_short",
        "logprob_long",
    ],
    "logits": [
        "logit_short",
        "logit_long",
        "norm_logit_short",
        "norm_logit_long",
        "rel_logit_delta",
    ],
    "fork": [
        "fork_entropy",
        "fork_diversity",
        "fork_simpson",
    ],
    "vocab": [
        "vocab_entropy",
        "vocab_diversity",
        "vocab_simpson",
    ],
    "trajectory": [
        "traj_inv_perplexity_short",
        "traj_inv_perplexity_long",
        "vocab_tcb",
    ],
}

# Display names for metrics
METRIC_DISPLAY_NAMES = {
    "recovery": "Recovery",
    "disruption": "Disruption",
    "logit_diff": "Logit Diff",
    "norm_logit_diff": "Norm Logit Diff",
    "reciprocal_rank_short": "RR (Short)",
    "prob_short": "P(Short)",
    "prob_long": "P(Long)",
    "logprob_short": "LogP(Short)",
    "logprob_long": "LogP(Long)",
    "logit_short": "Logit Short",
    "logit_long": "Logit Long",
    "norm_logit_short": "Norm Logit Short",
    "norm_logit_long": "Norm Logit Long",
    "rel_logit_delta": "Rel Logit Delta",
    "fork_entropy": "Fork Entropy",
    "fork_diversity": "Fork Diversity",
    "fork_simpson": "Fork Simpson",
    "vocab_entropy": "Vocab Entropy",
    "vocab_diversity": "Vocab Diversity",
    "vocab_simpson": "Vocab Simpson",
    "traj_inv_perplexity_short": "Inv Perp (Short)",
    "traj_inv_perplexity_long": "Inv Perp (Long)",
    "vocab_tcb": "Vocab TCB",
}

# Map metric field names to color keys
METRIC_TO_COLOR_KEY = {
    "reciprocal_rank_short": "rr_short",
    "reciprocal_rank_long": "rr_long",
    "traj_inv_perplexity_short": "inv_perplexity_short",
    "traj_inv_perplexity_long": "inv_perplexity_long",
}


def get_metric_color(metric_name: str) -> str:
    """Get color for metric from original color palette."""
    # Map field name to color key if needed
    color_key = METRIC_TO_COLOR_KEY.get(metric_name, metric_name)
    return METRIC_COLORS.get(color_key, "#1E90FF")
