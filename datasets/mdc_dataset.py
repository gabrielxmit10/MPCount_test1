"""MovingDroneCrowd++ adapter for MPCount.

The original dataset remains untouched. This loader reads the official
scene/clip split files and converts each head box into its centre point. For
training it creates only the cropped density target needed for the current
batch, avoiding full-resolution density-map files for 720p--4K frames.
"""

from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F

from utils.misc import get_padding, random_crop


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _numeric_name(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric frame filename, got: {path.name}") from exc


class MDCDenClsDataset(Dataset):
    """Expose MovingDroneCrowd++ frames in MPCount's den_cls format."""

    @staticmethod
    def collate(batch):
        transposed = list(zip(*batch))
        images1 = torch.stack(transposed[0], 0)
        images2 = torch.stack(transposed[1], 0)
        points = transposed[2]
        dmaps = torch.stack(transposed[3], 0)
        bmaps = torch.stack(transposed[4], 0)
        return images1, images2, (points, dmaps, bmaps)

    def __init__(
        self,
        root,
        crop_size,
        downsample,
        method,
        is_grey=False,
        unit_size=16,
        pre_resize=1.0,
        split_file=None,
        sigma=4.0,
        density_radius=7.0,
        clip_points=True,
    ):
        self.root = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
        self.method = method
        self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        self.downsample = int(downsample)
        self.is_grey = bool(is_grey)
        self.unit_size = int(unit_size) if unit_size is not None else 0
        self.pre_resize = float(pre_resize)
        self.sigma = float(sigma)
        self.density_radius = float(density_radius)
        self.clip_points = bool(clip_points)

        if method not in {"train", "val", "test"}:
            raise ValueError("method must be train, val or test")
        if self.downsample < 1:
            raise ValueError("downsample must be at least 1")

        split_name = split_file or f"{method}.txt"
        self.split_path = self.root / split_name
        if not self.split_path.is_file():
            raise FileNotFoundError(f"MDC split file not found: {self.split_path}")

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.more_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.samples = self._build_samples()
        if not self.samples:
            raise RuntimeError(f"No MDC frames were found for {self.split_path}")
        self.img_fns = [str(sample["image_path"]) for sample in self.samples]

    def _split_entries(self):
        entries = [line.strip().replace("\\", "/") for line in self.split_path.read_text(encoding="utf-8").splitlines()]
        return [entry for entry in entries if entry and not entry.startswith("#")]

    def _expand_entry(self, entry):
        parts = entry.split("/")
        if len(parts) == 2:
            return [(parts[0], parts[1])]
        if len(parts) != 1:
            raise ValueError(f"Invalid MDC split entry: {entry}")

        scene_dir = self.root / "frames" / parts[0]
        if not scene_dir.is_dir():
            raise FileNotFoundError(f"MDC scene directory not found: {scene_dir}")
        clips = sorted((path for path in scene_dir.iterdir() if path.is_dir()), key=lambda path: int(path.name))
        return [(parts[0], clip.name) for clip in clips]

    def _read_clip_points(self, annotation_path):
        points_by_frame = defaultdict(list)
        with annotation_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle), start=1):
                if len(row) != 10:
                    raise ValueError(f"{annotation_path}:{line_number}: expected 10 columns, got {len(row)}")
                frame_id = int(float(row[0]))
                x, y, width, height = map(float, row[2:6])
                if width <= 0 or height <= 0:
                    raise ValueError(f"{annotation_path}:{line_number}: non-positive head box")
                points_by_frame[frame_id].append((x + width / 2.0, y + height / 2.0))
        return points_by_frame

    def _build_samples(self):
        samples = []
        for entry in self._split_entries():
            for scene, clip in self._expand_entry(entry):
                frame_dir = self.root / "frames" / scene / clip
                annotation_path = self.root / "annotations" / scene / f"{clip}.csv"
                if not frame_dir.is_dir():
                    raise FileNotFoundError(f"MDC clip directory not found: {frame_dir}")
                if not annotation_path.is_file():
                    raise FileNotFoundError(f"MDC annotation file not found: {annotation_path}")

                points_by_frame = self._read_clip_points(annotation_path)
                image_paths = sorted(
                    (path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
                    key=_numeric_name,
                )
                for image_path in image_paths:
                    image_number = _numeric_name(image_path)
                    frame_id = image_number - 1
                    points = np.asarray(points_by_frame.get(frame_id, []), dtype=np.float32).reshape(-1, 2)
                    samples.append({
                        "image_path": image_path,
                        "points": points,
                        "name": f"{scene}__{clip}__{image_number:06d}",
                        "scene": scene,
                        "clip": clip,
                        "frame_id": frame_id,
                    })
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_sample(self, index):
        sample = self.samples[index]
        image = Image.open(sample["image_path"]).convert("RGB")
        points = sample["points"].copy()
        width, height = image.size

        if len(points):
            if self.clip_points:
                points[:, 0] = np.clip(points[:, 0], 0, max(width - 1, 0))
                points[:, 1] = np.clip(points[:, 1], 0, max(height - 1, 0))
            else:
                keep = (
                    (points[:, 0] >= 0) & (points[:, 0] < width)
                    & (points[:, 1] >= 0) & (points[:, 1] < height)
                )
                points = points[keep]

        if self.pre_resize != 1.0:
            new_width = max(1, round(width * self.pre_resize))
            new_height = max(1, round(height * self.pre_resize))
            image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
            if len(points):
                points *= np.asarray([new_width / width, new_height / height], dtype=np.float32)

        return image, points, sample["name"]

    def _make_density(self, points, height, width):
        density = np.zeros((height, width), dtype=np.float32)
        if len(points):
            x = np.clip(points[:, 0].astype(np.int64), 0, width - 1)
            y = np.clip(points[:, 1].astype(np.int64), 0, height - 1)
            np.add.at(density, (y, x), 1.0)
            density = gaussian_filter(
                density,
                self.sigma,
                mode="constant",
                truncate=self.density_radius / self.sigma,
            ).astype(np.float32, copy=False)
            density_sum = float(density.sum())
            if density_sum > 0:
                density *= len(points) / density_sum
        return density

    def _train_transform(self, image, points):
        width, height = image.size
        crop_height, crop_width = self.crop_size

        if self.is_grey or random.random() > 0.88:
            image = image.convert("L").convert("RGB")

        if height < crop_height or width < crop_width:
            padding, height, width = get_padding(height, width, crop_height, crop_width)
            left, top, _, _ = padding
            image = F.pad(image, padding)
            if len(points):
                points += np.asarray([left, top], dtype=np.float32)

        top, left = random_crop(height, width, crop_height, crop_width)
        image = F.crop(image, top, left, crop_height, crop_width)
        if len(points):
            points -= np.asarray([left, top], dtype=np.float32)
            keep = (
                (points[:, 0] >= 0) & (points[:, 0] < crop_width)
                & (points[:, 1] >= 0) & (points[:, 1] < crop_height)
            )
            points = points[keep]

        if random.random() > 0.5:
            image = F.hflip(image)
            if len(points):
                points[:, 0] = (crop_width - 1) - points[:, 0]

        density = self._make_density(points, crop_height, crop_width)
        density = torch.from_numpy(density).unsqueeze(0)

        if self.downsample > 1:
            if crop_height % self.downsample or crop_width % self.downsample:
                raise ValueError("crop size must be divisible by downsample")
            down_height = crop_height // self.downsample
            down_width = crop_width // self.downsample
            density = density.reshape(
                1, down_height, self.downsample, down_width, self.downsample
            ).sum(dim=(2, 4))
            points = points / self.downsample

        map_height, map_width = density.shape[-2:]
        if map_height % 16 or map_width % 16:
            raise ValueError("density-map size must be divisible by 16 for MPCount's occupancy target")
        occupancy = density.reshape(1, map_height // 16, 16, map_width // 16, 16).sum(dim=(2, 4))
        occupancy = (occupancy > 0).float()

        image1 = self.transform(image)
        image2 = self.more_transform(image)
        point_tensor = torch.from_numpy(points.copy()).float()
        return image1, image2, point_tensor, density.float(), occupancy

    def _eval_transform(self, image, points, name):
        width, height = image.size
        if self.unit_size > 0:
            new_width = ((width + self.unit_size - 1) // self.unit_size) * self.unit_size
            new_height = ((height + self.unit_size - 1) // self.unit_size) * self.unit_size
            padding, _, _ = get_padding(height, width, new_height, new_width)
            left, top, _, _ = padding
            image = F.pad(image, padding)
            if len(points):
                points += np.asarray([left, top], dtype=np.float32)
        else:
            padding = (0, 0, 0, 0)

        points = points / self.downsample
        image_tensor = self.transform(image)
        point_tensor = torch.from_numpy(points.copy()).float()
        return image_tensor, image_tensor.clone(), point_tensor, name, padding

    def __getitem__(self, index):
        image, points, name = self._load_sample(index)
        if self.method == "train":
            return self._train_transform(image, points)
        return self._eval_transform(image, points, name)
