"""config for the head ablation experiments on TimeSformer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AblationConfig:
    # model params - dont change these unless you know what ur doing
    model_name: str = "model_local"
    num_layers: int = 12
    num_heads: int = 12
    head_dim: int = 64
    hidden_size: int = 768
    num_frames: int = 8
    image_size: int = 224
    patch_size: int = 16
    num_patches_per_frame: int = 196  # (224/16)^2
    num_classes: int = 174  # SSv2 classes

    # data - set ssv2_root_dir to the evaluation_frames/ path on the cluster
    ssv2_root_dir: Optional[str] = None
    data_split: str = "val"  # "train", "val", "test"
    num_eval_videos: int = 50
    batch_size: int = 4
    num_workers: int = 2

    # ablation method
    ablation_method: str = "zero"  # "zero" or "mean"

    # gradient importance - how many batches to average over
    grad_num_batches: int = 10

    # output
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    device: str = "cuda"

    @property
    def total_heads(self) -> int:
        return self.num_layers * 2 * self.num_heads  # 288 (temporal + spatial)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
