"""Validate a MovingDroneCrowd++ folder without importing PyTorch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DEFAULT_SPLITS = ("train.txt", "val.txt", "test.txt")


def numeric_stem(path):
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Non-numeric frame name: {path}") from exc


def read_split(root, split_file):
    path = root / split_file
    if not path.is_file():
        raise FileNotFoundError(f"Missing split file: {path}")
    return [
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def expand_entry(root, entry):
    parts = entry.split("/")
    if len(parts) == 2:
        return [entry]
    if len(parts) != 1:
        raise ValueError(f"Invalid split entry: {entry}")
    scene_dir = root / "frames" / entry
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Missing scene directory: {scene_dir}")
    clips = sorted((item for item in scene_dir.iterdir() if item.is_dir()), key=lambda item: int(item.name))
    return [f"{entry}/{clip.name}" for clip in clips]


def split_clips(root, split_file):
    clips = []
    for entry in read_split(root, split_file):
        clips.extend(expand_entry(root, entry))
    return clips


def validate_clip(root, key, max_examples=10):
    scene, clip = key.split("/")
    frame_dir = root / "frames" / scene / clip
    annotation_path = root / "annotations" / scene / f"{clip}.csv"
    errors = []
    warnings = []

    if not frame_dir.is_dir():
        return {"clip": key, "errors": [f"Missing frame directory: {frame_dir}"], "warnings": []}
    if not annotation_path.is_file():
        return {"clip": key, "errors": [f"Missing annotation file: {annotation_path}"], "warnings": []}

    try:
        images = sorted(
            (path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=numeric_stem,
        )
    except ValueError as exc:
        return {"clip": key, "errors": [str(exc)], "warnings": []}
    if not images:
        return {"clip": key, "errors": [f"No images in {frame_dir}"], "warnings": []}

    with Image.open(images[0]) as image:
        width, height = image.size
    with Image.open(images[-1]) as image:
        if image.size != (width, height):
            errors.append(f"First/last frame dimensions differ in {key}")

    image_numbers = [numeric_stem(path) for path in images]
    expected_numbers = list(range(1, len(images) + 1))
    if image_numbers != expected_numbers:
        errors.append(f"Frame filenames are not contiguous from 1 in {key}")

    annotated_frames = set()
    identities = set()
    boxes = 0
    malformed = 0
    nonpositive = 0
    bbox_outside = 0
    center_outside = 0
    examples = []

    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if len(row) != 10:
                malformed += 1
                if len(examples) < max_examples:
                    examples.append(f"line {line_number}: expected 10 columns, got {len(row)}")
                continue
            try:
                frame_id = int(float(row[0]))
                person_id = int(float(row[1]))
                x, y, box_width, box_height = map(float, row[2:6])
            except ValueError:
                malformed += 1
                if len(examples) < max_examples:
                    examples.append(f"line {line_number}: non-numeric required value")
                continue

            boxes += 1
            annotated_frames.add(frame_id)
            identities.add(person_id)
            if box_width <= 0 or box_height <= 0:
                nonpositive += 1
            if x < 0 or y < 0 or x + box_width > width or y + box_height > height:
                bbox_outside += 1
            center_x = x + box_width / 2.0
            center_y = y + box_height / 2.0
            if center_x < 0 or center_y < 0 or center_x >= width or center_y >= height:
                center_outside += 1
                if len(examples) < max_examples:
                    examples.append(
                        f"line {line_number}: centre ({center_x:.2f}, {center_y:.2f}) outside {width}x{height}"
                    )

    expected_frame_ids = set(range(len(images)))
    missing_annotation_frames = sorted(expected_frame_ids - annotated_frames)
    extra_annotation_frames = sorted(annotated_frames - expected_frame_ids)
    if malformed:
        errors.append(f"{malformed} malformed annotation rows")
    if nonpositive:
        errors.append(f"{nonpositive} non-positive boxes")
    if missing_annotation_frames:
        warnings.append(
            f"{len(missing_annotation_frames)} frames have no annotation rows and will be treated as empty"
        )
    if extra_annotation_frames:
        errors.append(f"{len(extra_annotation_frames)} annotation frame IDs have no image")
    if bbox_outside:
        warnings.append(f"{bbox_outside} boxes extend outside the image")
    if center_outside:
        warnings.append(f"{center_outside} derived centres require clipping")

    return {
        "clip": key,
        "frames": len(images),
        "boxes": boxes,
        "identities": len(identities),
        "resolution": [width, height],
        "bbox_outside": bbox_outside,
        "center_outside": center_outside,
        "errors": errors,
        "warnings": warnings,
        "examples": examples,
    }


def validate_dataset(root, split_files=DEFAULT_SPLITS):
    root = Path(os.path.expandvars(os.path.expanduser(str(root)))).resolve()
    if not (root / "frames").is_dir() or not (root / "annotations").is_dir():
        raise FileNotFoundError(f"Expected frames/ and annotations/ under {root}")

    report = {
        "root": str(root),
        "splits": {},
        "totals": defaultdict(int),
        "errors": [],
        "warnings": [],
    }
    seen = {}
    all_clips = []
    for split_file in split_files:
        clips = split_clips(root, split_file)
        report["splits"][split_file] = {"clips": len(clips), "clip_names": clips}
        for clip in clips:
            if clip in seen:
                report["errors"].append(f"Clip {clip} occurs in both {seen[clip]} and {split_file}")
            else:
                seen[clip] = split_file
            all_clips.append((split_file, clip))

    for split_file, clip in all_clips:
        result = validate_clip(root, clip)
        split = report["splits"][split_file]
        split.setdefault("frames", 0)
        split.setdefault("boxes", 0)
        split["frames"] += result.get("frames", 0)
        split["boxes"] += result.get("boxes", 0)
        report["totals"]["clips"] += 1
        report["totals"]["frames"] += result.get("frames", 0)
        report["totals"]["boxes"] += result.get("boxes", 0)
        report["totals"]["bbox_outside"] += result.get("bbox_outside", 0)
        report["totals"]["center_outside"] += result.get("center_outside", 0)
        report["errors"].extend(f"{clip}: {message}" for message in result["errors"])
        report["warnings"].extend(f"{clip}: {message}" for message in result["warnings"])
        if result.get("examples"):
            report.setdefault("examples", {})[clip] = result["examples"]

    available = set()
    for scene_dir in (root / "frames").iterdir():
        if scene_dir.is_dir():
            for clip_dir in scene_dir.iterdir():
                if clip_dir.is_dir():
                    available.add(f"{scene_dir.name}/{clip_dir.name}")
    unlisted = sorted(available - set(seen))
    missing = sorted(set(seen) - available)
    if unlisted:
        report["warnings"].append(f"{len(unlisted)} available clips are not in the selected split files")
        report["unlisted_clips"] = unlisted
    if missing:
        report["errors"].append(f"{len(missing)} split clips are missing from frames/")
        report["missing_clips"] = missing

    report["totals"] = dict(report["totals"])
    return report


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description="Validate MovingDroneCrowd++ for MPCount")
    parser.add_argument("--root", required=True, help="MovingDroneCrowd++ root directory")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Split files to validate")
    parser.add_argument("--json-output", help="Optional path for the complete JSON report")
    args = parser.parse_args()

    report = validate_dataset(args.root, tuple(args.splits))
    print(f"Dataset: {report['root']}")
    for name, split in report["splits"].items():
        print(f"  {name}: {split['clips']} clips, {split.get('frames', 0)} frames, {split.get('boxes', 0)} boxes")
    totals = report["totals"]
    print(
        f"Total: {totals.get('clips', 0)} clips, {totals.get('frames', 0)} frames, "
        f"{totals.get('boxes', 0)} boxes"
    )
    print(
        f"Boundary notes: {totals.get('bbox_outside', 0)} boxes cross an edge; "
        f"{totals.get('center_outside', 0)} centres will be clipped"
    )
    print(f"Errors: {len(report['errors'])}; warnings: {len(report['warnings'])}")
    for message in report["errors"][:20]:
        print(f"ERROR: {message}")
    for message in report["warnings"][:20]:
        print(f"WARNING: {message}")

    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"JSON report: {output}")

    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
