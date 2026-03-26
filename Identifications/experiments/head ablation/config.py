"""Config for the head-ablation experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AblationConfig:
    model_name: str = "model_local"
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    hidden_size: int = 768
    num_frames: int = 8
    image_size: int = 224
    patch_size: int = 16
    num_patches_per_frame: int = 196
    num_classes: int = 174

    ssv2_root_dir: Optional[str] = None
    data_split: str = "val"
    num_eval_videos: int = 50
    batch_size: int = 4
    num_workers: int = 2

    ablation_method: str = "zero"

    grad_num_batches: int = 10

    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    device: str = "cuda"

    @property
    def total_heads(self) -> int:
        return self.num_layers * 2 * self.num_heads

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
