"""Run MPCount on one image or a directory without loading all images at once."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

from models.models import (
    DGModel_base,
    DGModel_cls,
    DGModel_final,
    DGModel_mem,
    DGModel_memadd,
    DGModel_memcls,
)
from utils.misc import get_padding


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
MODEL_TYPES = {
    "base": DGModel_base,
    "mem": DGModel_mem,
    "memadd": DGModel_memadd,
    "cls": DGModel_cls,
    "memcls": DGModel_memcls,
    "final": DGModel_final,
}


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu or a GPU runtime")
    return device


def image_paths(source, recursive=False):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension: {source}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"Image source does not exist: {source}")
    iterator = source.rglob("*") if recursive else source.glob("*")
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not paths:
        raise RuntimeError(f"No JPG/JPEG/PNG images found under {source}")
    return paths


def load_image(path, unit_size, device):
    image = Image.open(path).convert("RGB")
    original = np.asarray(image)
    original_width, original_height = image.size
    if unit_size > 0:
        padded_width = ((original_width + unit_size - 1) // unit_size) * unit_size
        padded_height = ((original_height + unit_size - 1) // unit_size) * unit_size
        padding, _, _ = get_padding(original_height, original_width, padded_height, padded_width)
        image = F.pad(image, padding)
    else:
        padding = (0, 0, 0, 0)
    tensor = F.normalize(F.to_tensor(image), [0.5] * 3, [0.5] * 3).unsqueeze(0).to(device)
    return tensor, original, (original_height, original_width), padding


def density_output(model, tensor):
    output = model(tensor)
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.inference_mode()
def predict(model, image, original_size, padding, patch_size=512, log_para=1000):
    height, width = image.shape[-2:]
    density = np.zeros((height, width), dtype=np.float32)
    for top in range(0, height, patch_size):
        for left in range(0, width, patch_size):
            patch = image[..., top : min(top + patch_size, height), left : min(left + patch_size, width)]
            predicted = density_output(model, patch)[0, 0].detach().float().cpu().numpy()
            patch_height, patch_width = predicted.shape
            density[top : top + patch_height, left : left + patch_width] = predicted

    left, top, _, _ = padding
    original_height, original_width = original_size
    density = density[top : top + original_height, left : left + original_width]
    count = float(density.sum() / log_para)
    return density, count


def _extract_state(payload):
    if isinstance(payload, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            if key in payload:
                return payload[key]
    return payload


def load_model(model_path, model_name, device, deterministic=True, allow_partial=False):
    model = MODEL_TYPES[model_name](pretrained=False, deterministic=deterministic).to(device)
    try:
        payload = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(model_path, map_location=device)
    except pickle.UnpicklingError:
        # Full checkpoints contain optimizer/RNG objects. Only load checkpoints you trust.
        payload = torch.load(model_path, map_location=device, weights_only=False)
    state = _extract_state(payload)
    if isinstance(state, dict) and state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    incompatible = model.load_state_dict(state, strict=not allow_partial)
    if allow_partial and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(
            f"WARNING: partial checkpoint load: {len(incompatible.missing_keys)} missing, "
            f"{len(incompatible.unexpected_keys)} unexpected keys"
        )
    model.eval()
    return model


def output_stem(path, source):
    source = Path(source)
    if source.is_dir():
        relative = path.relative_to(source).with_suffix("")
        return "__".join(relative.parts)
    return path.stem


def save_visualization(original, density, count, name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{name}_pred_dmap.npy", density)
    figure = plt.figure(figsize=(12, 5))
    image_axis = figure.add_subplot(121)
    image_axis.imshow(original)
    image_axis.set_title(name)
    image_axis.axis("off")
    density_axis = figure.add_subplot(122)
    density_axis.imshow(density)
    density_axis.set_title(f"Predicted count: {count:.3f}")
    density_axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_dir / f"{name}.png", dpi=140)
    plt.close(figure)


def main(args):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    device = resolve_device(args.device)
    paths = image_paths(args.img_path, args.recursive)
    model = load_model(
        args.model_path,
        args.model_name,
        device,
        deterministic=args.deterministic,
        allow_partial=args.allow_partial_checkpoint,
    )
    print(f"Device: {device}; images: {len(paths)}; model: {args.model_name}")

    save_handle = None
    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        save_handle = open(args.save_path, "w", encoding="utf-8")
    start_time = time()
    try:
        for path in paths:
            image, original, original_size, padding = load_image(path, args.unit_size, device)
            density, count = predict(
                model, image, original_size, padding, patch_size=args.patch_size, log_para=args.log_para
            )
            print(f"{path}: {count:.6f}")
            if save_handle:
                save_handle.write(f"{path}\t{count:.6f}\n")
                save_handle.flush()
            if args.vis_dir:
                save_visualization(original, density, count, output_stem(path, args.img_path), args.vis_dir)
            del image
    finally:
        if save_handle:
            save_handle.close()
    print(f"Total time: {time() - start_time:.2f}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-path", "--img_path", required=True, help="Image or directory")
    parser.add_argument("--model-path", "--model_path", required=True, help="MPCount .pth weights")
    parser.add_argument("--model-name", choices=sorted(MODEL_TYPES), default="final")
    parser.add_argument("--save-path", "--save_path", help="Text output containing one count per image")
    parser.add_argument("--vis-dir", "--vis_dir", help="Directory for PNG visualizations and density NPY files")
    parser.add_argument("--unit-size", "--unit_size", type=int, default=16)
    parser.add_argument("--patch-size", "--patch_size", type=int, default=512)
    parser.add_argument("--log-para", "--log_para", type=float, default=1000)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-partial-checkpoint", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
