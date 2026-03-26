import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict
from transformers import TimesformerForVideoClassification, AutoImageProcessor

from config import AblationConfig
from real_video_loader import create_dataloader_from_config


def load_model_and_processor(config):
    print(f"Loading model: {config.model_name}")
    processor = AutoImageProcessor.from_pretrained(config.model_name)
    model = TimesformerForVideoClassification.from_pretrained(config.model_name)
    model = model.to(config.device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, processor


@torch.no_grad()
def evaluate(model, dataloader, device="cuda", desc="Evaluating"):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    for pixel_values, labels in tqdm(dataloader, desc=desc, leave=False):
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

        logits = model(pixel_values=pixel_values).logits

        correct_top1 += (logits.argmax(dim=-1) == labels).sum().item()

        _, top5_preds = logits.topk(5, dim=-1)
        correct_top5 += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

        total += labels.size(0)

    top1 = correct_top1 / total if total > 0 else 0.0
    top5 = correct_top5 / total if total > 0 else 0.0
    return {"top1": top1, "top5": top5, "total": total}


def run_baseline(config):
    model, processor = load_model_and_processor(config)
    dataloader = create_dataloader_from_config(config, processor=processor)

    print(f"\nRunning baseline on {config.num_eval_videos} videos...")
    results = evaluate(model, dataloader, device=config.device, desc="Baseline")

    print(f"Baseline Results:")
    print(f"  Top-1: {results['top1']:.4f} ({results['top1']*100:.2f}%)")
    print(f"  Top-5: {results['top5']:.4f} ({results['top5']*100:.2f}%)")
    return results


if __name__ == "__main__":
    import torch
    config = AblationConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    run_baseline(config)
