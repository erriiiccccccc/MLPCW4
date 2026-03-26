import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple


class SSv2Dataset(Dataset):
    def __init__(self, root_dir, split="val", num_frames=8, processor=None):
        self.frames_dir = os.path.join(root_dir, "frames")
        self.num_frames = num_frames
        self.processor = processor

        csv_path = os.path.join(root_dir, "frame_lists", f"{split}.csv")
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"Frame list not found: {csv_path}\n"
                f"Expected CSV files in {root_dir}/frame_lists/"
            )

        self.samples = []
        skipped = 0
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    skipped += 1
                    continue
                video_id, n_frames, label = parts[0], int(parts[1]), int(parts[2])
                video_dir = os.path.join(self.frames_dir, video_id)
                self.samples.append((video_dir, n_frames, label))

        print(
            f"SSv2 [{split}]: {len(self.samples)} videos loaded"
            + (f" ({skipped} skipped)" if skipped else "")
        )

    def _load_frames(self, video_dir, n_frames_on_disk):
        frames = []
        for i in range(1, n_frames_on_disk + 1):
            fpath = os.path.join(video_dir, f"{i:05d}.jpg")
            try:
                img = Image.open(fpath).convert("RGB")
                frames.append(np.array(img))
            except (FileNotFoundError, OSError):
                break

        # fallback: try listing directory
        if not frames:
            if os.path.isdir(video_dir):
                frame_files = sorted(
                    f for f in os.listdir(video_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
                for fname in frame_files:
                    img = Image.open(os.path.join(video_dir, fname)).convert("RGB")
                    frames.append(np.array(img))

        return frames

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_dir, n_frames_on_disk, label = self.samples[idx]
        frames = self._load_frames(video_dir, n_frames_on_disk)

        if not frames:
            dummy = np.zeros((self.num_frames, 224, 224, 3), dtype=np.uint8)
            frames = [dummy[i] for i in range(self.num_frames)]

        total = len(frames)
        if total >= self.num_frames:
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int).tolist()
        else:
            indices = list(range(total))
            indices += [total - 1] * (self.num_frames - total)

        sampled = [frames[i] for i in indices]

        if self.processor is not None:
            inputs = self.processor(images=sampled, return_tensors="pt")
            pixel_values = inputs["pixel_values"].squeeze(0)
        else:
            stacked = np.stack(sampled).astype(np.float32) / 255.0
            pixel_values = torch.from_numpy(stacked).permute(3, 0, 1, 2)

        return pixel_values, label


def create_ssv2_dataloader(root_dir, split="val", num_frames=8,
                           batch_size=2, num_workers=0, processor=None):
    dataset = SSv2Dataset(
        root_dir=root_dir, split=split,
        num_frames=num_frames, processor=processor,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )


def create_dataloader_from_config(config, processor=None):
    if not config.ssv2_root_dir:
        raise ValueError(
            "ssv2_root_dir must be set in AblationConfig "
            "(path to evaluation_frames/ directory)"
        )
    dataset = SSv2Dataset(
        root_dir=config.ssv2_root_dir,
        split=config.data_split,
        num_frames=config.num_frames,
        processor=processor,
    )
    full_size = len(dataset)
    if config.num_eval_videos < full_size:
        dataset = Subset(dataset, list(range(config.num_eval_videos)))
        print(f"Subset to {config.num_eval_videos} videos (from {full_size})")
    return DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
