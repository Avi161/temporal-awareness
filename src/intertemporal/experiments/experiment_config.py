"""Experiment configuration for intertemporal experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

from ...common import BaseSchema
from ..preference import PreferenceDataset
from ..prompt import PromptDatasetConfig


@dataclass
class ExperimentConfig(BaseSchema):
    """Experiment configuration."""

    model: str
    dataset_config: dict
    max_samples: int | None = None
    n_pairs: int | None = None
    coarse_patch_layer_step_sizes: list[int] = field(default_factory=lambda: [1])
    coarse_patch_pos_step_sizes: list[int] = field(default_factory=lambda: [1])

    @property
    def name(self) -> str:
        return self.dataset_config.get("name", "default")

    def get_prefix(self) -> str:
        cfg = PromptDatasetConfig.from_dict(self.dataset_config)
        return PreferenceDataset.make_prefix(cfg.get_id(), self.model)
