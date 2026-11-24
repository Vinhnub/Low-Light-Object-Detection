"""
Deterministic Train / Val / Test Split Script for the DarkFace Dataset.

Splits `dataset/DarkFace` (6,000 annotated low-light face images) into three
strictly disjoint partitions using a fixed random seed (default: 42) and an
8:1:1 ratio (80% train / 10% val / 10% test -> 4,800 / 600 / 600 images).

Reproducibility guarantees:
  1. Source filenames are sorted with a cross-platform natural sort key before
     shuffling, eliminating OS/filesystem ordering differences (Windows vs Linux).
  2. Shuffling uses an isolated `random.Random(seed)` instance.
  3. Exact split membership, seed, ratios, and a SHA-256 fingerprint are saved
     to `split_manifest.json`, which can also be replayed via `--from-manifest`.
  4. Refuses to mix into existing split folders unless `--overwrite` is passed,
     and cleans stale `.cache.pt` / `.gt_cache_*.json` caches on overwrite.

Directory layout generated (matching `tools/split_exdark.py` & `DarkFaceDataset`):
dataset/DarkFace/
  ├── images/
  │   ├── train/            (4,800 images - 80%)
  │   ├── val/              (  600 images - 10%)
  │   └── test/             (  600 images - 10%)
  ├── labels/               (YOLO format: `0 <xc> <yc> <w> <h>` normalized)
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels_raw/           (Original DarkFace format: `<N>\\n<x1> <y1> <x2> <y2>...`)
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── train.txt             (Sorted relative image paths for training)
  ├── val.txt               (Sorted relative image paths for validation)
  ├── test.txt              (Sorted relative image paths for testing)
  ├── split_manifest.json   (Seed, ratios, stats, SHA-256 hash, exact file lists)
  └── darkface.yaml         (Dataset configuration for YOLO / Ultralytics)

Examples:
  # Preview split stats without modifying disk
  python tools/split_darkface.py --dry-run

  # Standard 8:1:1 split with fixed seed 42 (copies images, keeps originals)
  python tools/split_darkface.py --seed 42

  # Zero extra disk space on NTFS/ext4 using hardlinks (keeps originals intact)
  python tools/split_darkface.py --seed 42 --mode hardlink

  # Only generate train.txt / val.txt / test.txt & split_manifest.json
  python tools/split_darkface.py --seed 42 --mode list-only

  # Re-run cleanly over an existing split
  python tools/split_darkface.py --seed 42 --overwrite
"""

import argparse
import hashlib
import json
import os
import random
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
CLASSES = ["Face"]
SPLITS = ("train", "val", "test")
DEFAULT_SEED = 42
DEFAULT_RATIOS = (8.0, 1.0, 1.0)


def natural_sort_key(filename: str) -> Tuple:
    """Cross-platform deterministic natural sort key (e.g., 1.png, 2.png, ..., 10.png)."""
    parts = re.split(r"(\d+)", filename)
    return tuple((0, int(p)) if p.isdigit() else (1, p.lower()) for p in parts) + (filename,)


def get_image_size(img_path: Path) -> Tuple[int, int]:
    """Read (width, height) from image header without decoding full pixels when PIL is available."""
    if Image is not None:
        with Image.open(img_path) as im:
            return im.size  # (width, height)
    if cv2 is not None:
        im = cv2.imread(str(img_path))
        if im is None:
            raise ValueError(f"Failed to read image with OpenCV: {img_path}")
        h, w = im.shape[:2]
        return w, h
    raise RuntimeError("Either Pillow (PIL) or OpenCV (cv2) is required to read image dimensions.")


def find_source_dirs(darkface_dir: Path) -> Tuple[Path, Path]:
    """Locate original DarkFace image/ and label/ directories."""
    img_dir = None
    for candidate in ("image", "Image"):
        p = darkface_dir / candidate
        if p.is_dir():
            img_dir = p
            break
    if img_dir is None:
        raise FileNotFoundError(
            f"Could not find source image directory ('image/' or 'Image/') inside: {darkface_dir}"
        )

    lbl_dir = None
    for candidate in ("label", "Label"):
        p = darkface_dir / candidate
        if p.is_dir():
            lbl_dir = p
            break
    if lbl_dir is None:
        raise FileNotFoundError(
            f"Could not find source label directory ('label/' or 'Label/') inside: {darkface_dir}"
        )

    return img_dir, lbl_dir


def collect_paired_samples(img_dir: Path, lbl_dir: Path) -> Tuple[List[str], List[str]]:
    """
    Collect deterministically sorted image filenames from `img_dir` and check
    whether each has a corresponding `.txt` label in `lbl_dir`.
    """
    image_names = sorted(
        (
            p.name
            for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not image_names:
        raise FileNotFoundError(f"No image files found in: {img_dir}")

    missing_labels = [
        name for name in image_names if not (lbl_dir / f"{Path(name).stem}.txt").is_file()
    ]
    return image_names, missing_labels


def compute_split_counts(n_total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    """Compute exact integer split counts (n_train, n_val, n_test) summing to `n_total`."""
    if len(ratios) != 3:
        raise ValueError(f"Expected 3 ratios (train, val, test), got: {ratios}")
    r_train, r_val, r_test = [float(r) for r in ratios]
    if r_train <= 0 or r_val < 0 or r_test < 0:
        raise ValueError(f"Ratios must be positive for train and non-negative for val/test: {ratios}")
    total_r = r_train + r_val + r_test
    if total_r <= 0:
        raise ValueError("Sum of split ratios must be positive.")

    n_train = int(round(n_total * (r_train / total_r)))
    n_val = int(round(n_total * (r_val / total_r)))
    n_test = n_total - n_train - n_val

    if n_test < 0:
        n_val += n_test
        n_test = 0
    if n_train < 0 or n_val < 0 or n_test < 0 or (n_train + n_val + n_test != n_total):
        raise ValueError(f"Invalid split counts ({n_train}, {n_val}, {n_test}) for total={n_total}")

    return n_train, n_val, n_test


def build_deterministic_splits(
    image_names: List[str],
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
) -> Dict[str, List[str]]:
    """
    Shuffle canonically sorted `image_names` with a fixed seed and slice into
    strictly disjoint train / val / test lists (each sorted for clean output).
    """
    canonical = sorted(image_names, key=natural_sort_key)
    rng = random.Random(seed)
    shuffled = list(canonical)
    rng.shuffle(shuffled)

    n_train, n_val, n_test = compute_split_counts(len(shuffled), ratios)

    splits = {
        "train": sorted(shuffled[:n_train], key=natural_sort_key),
        "val": sorted(shuffled[n_train : n_train + n_val], key=natural_sort_key),
        "test": sorted(shuffled[n_train + n_val :], key=natural_sort_key),
    }

    # Sanity invariants: strictly disjoint and complete
    set_tr, set_va, set_te = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert not (set_tr & set_va), "Overlap detected between train and val!"
    assert not (set_tr & set_te), "Overlap detected between train and test!"
    assert not (set_va & set_te), "Overlap detected between val and test!"
    assert len(set_tr) + len(set_va) + len(set_te) == len(canonical), "Split count mismatch!"
    assert len(splits["test"]) == n_test

    return splits


def load_splits_from_manifest(
    manifest_path: Path, available_images: Sequence[str]
) -> Tuple[Dict[str, List[str]], Optional[int], Optional[List[float]]]:
    """Replay an exact saved split from `split_manifest.json`."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "splits" not in data:
        raise ValueError(f"Invalid manifest (missing 'splits' key): {manifest_path}")

    available_set = set(available_images)
    splits: Dict[str, List[str]] = {}
    seen = set()
    for s in SPLITS:
        names = list(data["splits"].get(s, []))
        missing = [n for n in names if n not in available_set]
        if missing:
            raise FileNotFoundError(
                f"Manifest split '{s}' references {len(missing)} missing images (e.g. {missing[:5]})"
            )
        overlap = seen.intersection(names)
        if overlap:
            raise ValueError(f"Manifest has duplicate images across splits: {list(overlap)[:5]}")
        seen.update(names)
        splits[s] = sorted(names, key=natural_sort_key)

    manifest_seed = data.get("seed")
    manifest_ratios = data.get("ratios_raw") or data.get("ratios")
    return splits, manifest_seed, manifest_ratios


def compute_split_fingerprint(splits: Dict[str, List[str]]) -> str:
    """Compute a deterministic SHA-256 fingerprint over the exact split assignments."""
    payload = json.dumps(
        {s: splits[s] for s in SPLITS},
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_darkface_raw_lines(
    raw_lines: List[str], img_w: int, img_h: int
) -> Tuple[List[str], int, int]:
    """
    Convert raw DarkFace pixel label lines (`<count>\\n<x1> <y1> <x2> <y2>...`)
    to normalized YOLO lines (`0 <xc> <yc> <w> <h>\\n`).

    Returns:
        (yolo_lines, valid_box_count, skipped_box_count)
    """
    non_empty = [ln.strip() for ln in raw_lines if ln.strip()]
    if not non_empty:
        return [], 0, 0

    # Standard DarkFace format has box count N on the first line
    start_idx = 1 if (len(non_empty[0].split()) == 1 and non_empty[0].isdigit()) else 0

    yolo_lines: List[str] = []
    valid_boxes = 0
    skipped_boxes = 0

    for line in non_empty[start_idx:]:
        parts = line.split()
        if len(parts) < 4:
            skipped_boxes += 1
            continue
        try:
            x1, y1, x2, y2 = (
                float(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            )
        except ValueError:
            skipped_boxes += 1
            continue

        # Clamp coordinates to image boundaries
        x1 = max(0.0, min(x1, float(img_w)))
        y1 = max(0.0, min(y1, float(img_h)))
        x2 = max(0.0, min(x2, float(img_w)))
        y2 = max(0.0, min(y2, float(img_h)))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0.0 or bh <= 0.0:
            skipped_boxes += 1
            continue

        xc = (x1 + bw / 2.0) / img_w
        yc = (y1 + bh / 2.0) / img_h
        wn = bw / img_w
        hn = bh / img_h

        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
        valid_boxes += 1

    return yolo_lines, valid_boxes, skipped_boxes


def transfer_image(src: Path, dst: Path, mode: str) -> None:
    """Transfer an image from `src` to `dst` using the selected mode."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copyfile(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(os.path.relpath(src, dst.parent), dst)
    elif mode == "move":
        shutil.move(str(src), str(dst))
    else:
        raise ValueError(f"Unsupported transfer mode: {mode}")


def process_single_item(
    img_name: str,
    split_name: str,
    img_dir: Path,
    lbl_dir: Path,
    darkface_dir: Path,
    mode: str,
    dry_run: bool,
) -> Tuple[str, str, int, int, bool]:
    """
    Process one image and its corresponding DarkFace label.

    Returns:
        (split_name, rel_img_path, valid_boxes, skipped_boxes, had_label)
    """
    src_img = img_dir / img_name
    stem = Path(img_name).stem
    src_lbl = lbl_dir / f"{stem}.txt"

    img_w, img_h = get_image_size(src_img)

    had_label = src_lbl.is_file()
    raw_lines: List[str] = []
    if had_label:
        with open(src_lbl, "r", encoding="utf-8", errors="ignore") as f:
            raw_lines = f.readlines()

    yolo_lines, valid_boxes, skipped_boxes = parse_darkface_raw_lines(raw_lines, img_w, img_h)

    if mode == "list-only":
        rel_path = f"{img_dir.name}/{img_name}"
        return split_name, rel_path, valid_boxes, skipped_boxes, had_label

    rel_path = f"images/{split_name}/{img_name}"
    if dry_run:
        return split_name, rel_path, valid_boxes, skipped_boxes, had_label

    dst_img = darkface_dir / "images" / split_name / img_name
    dst_lbl_yolo = darkface_dir / "labels" / split_name / f"{stem}.txt"
    dst_lbl_raw = darkface_dir / "labels_raw" / split_name / f"{stem}.txt"

    transfer_image(src_img, dst_img, mode=mode)

    with open(dst_lbl_yolo, "w", encoding="utf-8") as f:
        f.writelines(yolo_lines)

    if had_label:
        with open(dst_lbl_raw, "w", encoding="utf-8") as f:
            f.writelines(raw_lines)
    else:
        with open(dst_lbl_raw, "w", encoding="utf-8") as f:
            f.write("0\n")

    return split_name, rel_path, valid_boxes, skipped_boxes, had_label


def check_and_prepare_output_dirs(
    darkface_dir: Path, mode: str, overwrite: bool, dry_run: bool
) -> None:
    """Guard against mixing stale split directories and remove old caches on overwrite."""
    if mode == "list-only":
        existing_lists = [
            darkface_dir / f"{s}.txt" for s in SPLITS if (darkface_dir / f"{s}.txt").exists()
        ]
        if existing_lists and not overwrite and not dry_run:
            names = ", ".join(p.name for p in existing_lists)
            raise FileExistsError(
                f"Split files already exist ({names}) in {darkface_dir}. "
                "Pass --overwrite to regenerate them cleanly."
            )
        return

    target_dirs = [
        darkface_dir / sub / s
        for sub in ("images", "labels", "labels_raw")
        for s in SPLITS
    ]
    non_empty = [d for d in target_dirs if d.is_dir() and any(d.iterdir())]

    if non_empty and not overwrite and not dry_run:
        sample = ", ".join(str(d.relative_to(darkface_dir)) for d in non_empty[:3])
        raise FileExistsError(
            f"Target split directories already contain files (e.g. {sample}). "
            "Refusing to mix old and new splits (data leakage risk). "
            "Pass --overwrite to clean and rebuild them."
        )

    if dry_run:
        return

    if overwrite:
        for d in target_dirs:
            if d.is_dir():
                shutil.rmtree(d)
        # Clean stale label/GT caches so YOLODataset & measure.py re-index the new split
        for pattern in ("images/*_labels.cache.pt", "*.labels.cache.pt", "labels_raw/*/.gt_cache_*.json"):
            for cache_file in darkface_dir.glob(pattern):
                if cache_file.is_file():
                    cache_file.unlink()

    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)


def write_yaml_config(darkface_dir: Path, mode: str) -> Path:
    """Write `darkface.yaml` matching `dataset/ExDark/exdark.yaml` convention."""
    try:
        rel_root = darkface_dir.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        rel_root = darkface_dir.as_posix()

    if mode == "list-only":
        train_entry = "train.txt"
        val_entry = "val.txt"
        test_entry = "test.txt"
    else:
        train_entry = "images/train"
        val_entry = "images/val"
        test_entry = "images/test"

    yaml_content = (
        "# DarkFace Dataset Configuration for YOLO\n"
        f"path: {rel_root}  # dataset root dir\n"
        f"train: {train_entry}\n"
        f"val: {val_entry}\n"
        f"test: {test_entry}\n\n"
        "# Classes (1 category)\n"
        "nc: 1\n"
        "names:\n"
        "  0: Face\n"
    )
    yaml_path = darkface_dir / "darkface.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return yaml_path


def split_darkface(
    darkface_dir: Path,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    seed: int = DEFAULT_SEED,
    mode: str = "copy",
    workers: int = 8,
    overwrite: bool = False,
    dry_run: bool = False,
    from_manifest: Optional[Path] = None,
) -> Dict:
    """Execute the deterministic train/val/test split for DarkFace."""
    darkface_dir = darkface_dir.resolve()
    img_dir, lbl_dir = find_source_dirs(darkface_dir)

    image_names, missing_labels = collect_paired_samples(img_dir, lbl_dir)
    if missing_labels:
        print(
            f"[Warning] {len(missing_labels)} images have no matching label in {lbl_dir} "
            f"(first 5: {missing_labels[:5]})"
        )

    if from_manifest is not None:
        splits, manifest_seed, manifest_ratios = load_splits_from_manifest(
            from_manifest.resolve(), image_names
        )
        if manifest_seed is not None:
            seed = int(manifest_seed)
        if isinstance(manifest_ratios, list) and len(manifest_ratios) == 3:
            ratios = tuple(float(x) for x in manifest_ratios)
        print(f"Loaded split assignments from manifest: {from_manifest}")
    else:
        splits = build_deterministic_splits(image_names, ratios=ratios, seed=seed)

    fingerprint = compute_split_fingerprint(splits)
    check_and_prepare_output_dirs(darkface_dir, mode=mode, overwrite=overwrite, dry_run=dry_run)

    # Flatten work items in deterministic order
    work_items: List[Tuple[str, str]] = [
        (img_name, split_name)
        for split_name in SPLITS
        for img_name in splits[split_name]
    ]

    split_paths: Dict[str, List[str]] = {s: [] for s in SPLITS}
    box_counts: Dict[str, int] = {s: 0 for s in SPLITS}
    skipped_counts: Dict[str, int] = {s: 0 for s in SPLITS}
    missing_lbl_counts: Dict[str, int] = {s: 0 for s in SPLITS}

    action_desc = "Dry-run scanning" if dry_run else f"Splitting ({mode})"
    max_workers = max(1, int(workers))

    if max_workers == 1:
        iterator = (
            process_single_item(name, s_name, img_dir, lbl_dir, darkface_dir, mode, dry_run)
            for name, s_name in work_items
        )
        results = list(tqdm(iterator, total=len(work_items), desc=action_desc))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    process_single_item,
                    name,
                    s_name,
                    img_dir,
                    lbl_dir,
                    darkface_dir,
                    mode,
                    dry_run,
                )
                for name, s_name in work_items
            ]
            results = [
                f.result()
                for f in tqdm(futures, total=len(futures), desc=action_desc)
            ]

    for s_name, rel_path, n_valid, n_skipped, had_lbl in results:
        split_paths[s_name].append(rel_path)
        box_counts[s_name] += n_valid
        skipped_counts[s_name] += n_skipped
        if not had_lbl:
            missing_lbl_counts[s_name] += 1

    # Sort paths deterministically
    for s in SPLITS:
        split_paths[s].sort(key=lambda p: natural_sort_key(Path(p).name))

    total_images = sum(len(splits[s]) for s in SPLITS)
    total_boxes = sum(box_counts.values())
    total_skipped = sum(skipped_counts.values())
    total_r = sum(ratios)
    norm_ratios = {
        "train": round(ratios[0] / total_r, 6),
        "val": round(ratios[1] / total_r, 6),
        "test": round(ratios[2] / total_r, 6),
    }

    manifest = {
        "dataset": "DarkFace",
        "seed": seed,
        "ratios_raw": [float(r) for r in ratios],
        "ratios_normalized": norm_ratios,
        "mode": mode,
        "sha256": fingerprint,
        "total_images": total_images,
        "total_boxes": total_boxes,
        "skipped_invalid_boxes": total_skipped,
        "counts": {
            s: {
                "images": len(splits[s]),
                "image_pct": round(100.0 * len(splits[s]) / max(1, total_images), 2),
                "boxes": box_counts[s],
                "box_pct": round(100.0 * box_counts[s] / max(1, total_boxes), 2),
                "avg_boxes_per_image": round(box_counts[s] / max(1, len(splits[s])), 2),
                "skipped_invalid_boxes": skipped_counts[s],
                "missing_labels": missing_lbl_counts[s],
            }
            for s in SPLITS
        },
        "splits": splits,
    }

    if not dry_run:
        # Write train.txt, val.txt, test.txt
        for s in SPLITS:
            txt_path = darkface_dir / f"{s}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                for rel_p in split_paths[s]:
                    f.write(rel_p.replace("\\", "/") + "\n")
            print(f"Wrote {len(split_paths[s]):>5d} paths -> {txt_path}")

        # Write darkface.yaml
        yaml_path = write_yaml_config(darkface_dir, mode=mode)
        print(f"Wrote YOLO config     -> {yaml_path}")

        # Write split_manifest.json
        manifest_path = darkface_dir / "split_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote split manifest  -> {manifest_path}")

    # Print formatted summary
    print("\n" + "=" * 78)
    print(f"DARKFACE SPLIT SUMMARY {'(DRY RUN - NO FILES WRITTEN)' if dry_run else ''}")
    print("=" * 78)
    print(f"Dataset dir : {darkface_dir}")
    print(f"Seed        : {seed}")
    print(f"Ratios      : {ratios[0]:g}:{ratios[1]:g}:{ratios[2]:g} "
          f"({norm_ratios['train']*100:.1f}% / {norm_ratios['val']*100:.1f}% / {norm_ratios['test']*100:.1f}%)")
    print(f"Mode        : {mode}")
    print(f"SHA-256     : {fingerprint}")
    print("-" * 78)
    print(f"{'Split':<8} {'Images':>8} {'Img %':>8} {'Faces':>10} {'Face %':>8} {'Faces/Img':>10} {'Skipped':>8}")
    print("-" * 78)
    for s in SPLITS:
        st = manifest["counts"][s]
        print(
            f"{s:<8} {st['images']:>8d} {st['image_pct']:>7.2f}% "
            f"{st['boxes']:>10d} {st['box_pct']:>7.2f}% "
            f"{st['avg_boxes_per_image']:>10.2f} {st['skipped_invalid_boxes']:>8d}"
        )
    print("-" * 78)
    avg_total = total_boxes / max(1, total_images)
    print(
        f"{'Total':<8} {total_images:>8d} {100.0:>7.2f}% "
        f"{total_boxes:>10d} {100.0:>7.2f}% {avg_total:>10.2f} {total_skipped:>8d}"
    )
    print("=" * 78)

    return manifest


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    default_data_dir = project_root / "dataset" / "DarkFace"

    parser = argparse.ArgumentParser(
        description="Deterministically split DarkFace dataset into Train / Val / Test (default 8:1:1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        type=Path,
        default=default_data_dir,
        help="Path to DarkFace dataset root containing image/ and label/ directories.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_RATIOS),
        help="Train/Val/Test split ratios (e.g. '8 1 1' or '0.8 0.1 0.1').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Fixed random seed for reproducible dataset splitting.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=("copy", "hardlink", "symlink", "move", "list-only"),
        help=(
            "How to materialize split images: "
            "'copy' (portable, matches split_exdark.py), "
            "'hardlink' (zero extra disk space on same volume), "
            "'symlink', 'move', or 'list-only' (only writes .txt lists & manifest)."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads for image/label processing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clean and rebuild existing split folders / txt files if already present.",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Compute and display split statistics without creating or modifying any files.",
    )
    parser.add_argument(
        "--from-manifest",
        "--from_manifest",
        type=Path,
        default=None,
        help="Optional path to an existing split_manifest.json to reproduce the exact same split.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_darkface(
        darkface_dir=args.data_dir,
        ratios=args.ratios,
        seed=args.seed,
        mode=args.mode,
        workers=args.workers,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        from_manifest=args.from_manifest,
    )
