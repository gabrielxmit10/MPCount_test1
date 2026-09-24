"""Fast MDC adapter + MPCount forward-pass check for the Colab workflow."""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.mdc_dataset import MDCDenClsDataset
from inference import load_model
from models.models import DGModel_final
from trainers.dgtrainer import DGTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crop-size", type=int, default=320)
    parser.add_argument("--tiny-eval-samples", type=int, default=0)
    parser.add_argument("--eval-output-dir")
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--save-checkpoint")
    args = parser.parse_args()

    root = Path(args.data_root)
    train_dataset = MDCDenClsDataset(
        root=root,
        crop_size=args.crop_size,
        downsample=1,
        method="train",
        unit_size=16,
    )
    sample = train_dataset[0]
    image1, image2, points, density, occupancy = sample
    assert image1.shape == image2.shape == (3, args.crop_size, args.crop_size)
    assert density.shape == (1, args.crop_size, args.crop_size)
    assert occupancy.shape == (1, args.crop_size // 16, args.crop_size // 16)
    print(
        f"Dataset OK: {len(train_dataset)} training frames; sample crop has "
        f"{len(points)} points and density sum {density.sum().item():.3f}"
    )
    train_batch = MDCDenClsDataset.collate([sample, train_dataset[1]])
    assert train_batch[0].shape == (2, 3, args.crop_size, args.crop_size)
    assert train_batch[2][1].shape == (2, 1, args.crop_size, args.crop_size)

    validation_dataset = MDCDenClsDataset(
        root=root,
        crop_size=args.crop_size,
        downsample=1,
        method="val",
        unit_size=16,
    )
    validation_batch = next(iter(DataLoader(validation_dataset, batch_size=1, num_workers=0)))
    assert validation_batch[0].shape[0] == 1
    assert validation_batch[2].ndim == 3
    print(f"Loader OK: batch collation and {len(validation_dataset)} validation frames")

    device = torch.device(args.device)
    if args.checkpoint:
        model = load_model(args.checkpoint, "final", device, deterministic=True)
        print(f"Checkpoint OK: {args.checkpoint}")
    else:
        model = DGModel_final(pretrained=args.pretrained, deterministic=True).to(device).eval()
        initialization = "ImageNet VGG16-BN" if args.pretrained else "random"
        print(f"No MPCount checkpoint supplied; using {initialization} initialization")
    with torch.inference_mode():
        output = model(image1.unsqueeze(0).to(device))[0]
    assert output.shape == (1, 1, args.crop_size, args.crop_size), output.shape
    assert torch.isfinite(output).all()
    print(f"Model forward OK: output {tuple(output.shape)} on {device}")

    trainer = DGTrainer(
        seed=0,
        version="mpcount_smoke",
        device=str(device),
        log_para=1000,
        patch_size=512,
        mode="final",
        output_dir=tempfile.gettempdir(),
    )
    if args.tiny_eval_samples:
        model.eval()
        rows = []
        for index, batch in enumerate(DataLoader(validation_dataset, batch_size=1, num_workers=0)):
            with torch.inference_mode():
                result = trainer.test_step(model, batch)
            detail = result["_detail"]
            print(
                f"Tiny evaluation {index + 1}: {detail['name']} pred={detail['predicted_count']:.3f} "
                f"gt={detail['ground_truth_count']} abs_error={detail['absolute_error']:.3f}"
            )
            rows.append(detail)
            if index + 1 >= args.tiny_eval_samples:
                break
        if args.eval_output_dir:
            output_dir = Path(args.eval_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["name", "predicted_count", "ground_truth_count", "absolute_error"],
                )
                writer.writeheader()
                writer.writerows(rows)
            mae = sum(row["absolute_error"] for row in rows) / len(rows)
            mse = sum(
                (row["predicted_count"] - row["ground_truth_count"]) ** 2 for row in rows
            ) / len(rows)
            metrics = {"samples": len(rows), "mae": float(mae), "rmse": float(math.sqrt(mse))}
            (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            print(f"Tiny evaluation outputs: {output_dir}")

    if args.train_step:
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        loss_value = trainer.train_step(model, nn.MSELoss(), optimizer, train_batch, epoch=0)
        assert torch.isfinite(torch.tensor(loss_value))
        print(f"Tiny training step OK: loss={loss_value:.6f}")
        if args.save_checkpoint:
            checkpoint_path = Path(args.save_checkpoint)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Tiny training checkpoint created successfully: {checkpoint_path}")


if __name__ == "__main__":
    main()
