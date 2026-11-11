"""
PyTorch Dataset Classes for Low-Light Object Detection (ExDark, DarkFace, VOC, Custom YOLO).
Implements the complete YOLOv5 training & evaluation data pipeline:
  - Mosaic-4 augmentation (`load_mosaic`) + Random Affine/Perspective (`random_perspective`)
  - MixUp augmentation
  - Aspect-ratio preserving Letterbox resizing (`letterbox`)
  - HSV color-space jitter (`augment_hsv`) & Random Gamma curve (`random_gamma`)
  - Horizontal / Vertical flips (`fliplr`, `flipud`)
  - Standard YOLOv5 batch collation (`collate_fn`)
"""

import glob
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.util import (
    DARKFACE_CLASSES,
    EXDARK_CLASSES,
    EXDARK_CLASS_TO_ID,
    IMG_EXTENSIONS,
    augment_hsv,
    clip_coords,
    img2label_paths,
    is_image_file,
    letterbox,
    parse_label_file,
    random_gamma,
    random_perspective,
    xywhn2xyxy,
    xyxy2xywhn,
)


DEFAULT_AUG_HYP: Dict[str, Any] = {
    "mosaic": 1.0,
    "mixup": 0.0,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "fliplr": 0.5,
    "flipud": 0.0,
    "gamma": False,
    "start_gamma": 60,
    "end_gamma": 120,
    "cache_images": False,
}


def _extract_aug_hyp(hyp: Optional[Union[Dict[str, Any], Any]] = None) -> Dict[str, Any]:
    """Merge default augmentation hyperparameters with dict or argparse Namespace."""
    h = dict(DEFAULT_AUG_HYP)
    if hyp is None:
        return h
    if isinstance(hyp, dict):
        h.update(hyp)
    else:
        for k in DEFAULT_AUG_HYP:
            if hasattr(hyp, k):
                h[k] = getattr(hyp, k)
    return h


class YOLODataset(Dataset):
    """
    Standard YOLOv5 Object Detection Dataset.

    Returns per item:
        img (torch.Tensor): Float32 RGB tensor of shape (3, img_size, img_size) normalized in [0.0, 1.0].
        labels_out (torch.Tensor): Float32 tensor of shape (N_boxes, 6) with rows
            `[batch_idx(0), class_id, x_center, y_center, width, height]` normalized in [0.0, 1.0].
        img_path (str): Path to the image file.
        shapes (Tuple): `((h0, w0), ((gain_h, gain_w), (pad_w, pad_h)))` for `scale_boxes` during evaluation.
    """

    def __init__(
        self,
        path: Union[str, Path, List[str]],
        img_size: int = 640,
        augment: bool = False,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        stride: int = 32,
        pad: float = 0.0,
        class_names: Optional[List[str]] = None,
        label_dir: Optional[str] = None,
        cache_images: Optional[bool] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.augment = augment
        self.hyp = _extract_aug_hyp(hyp)
        self.mosaic = self.augment and (self.hyp.get("mosaic", 1.0) > 0)
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.pad = pad
        self.class_names = class_names or EXDARK_CLASSES
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.cache_images = (
            bool(self.hyp.get("cache_images", False))
            if cache_images is None
            else bool(cache_images)
        )

        # Collect image files
        self.im_files = self._resolve_image_files(path)
        if not self.im_files:
            raise FileNotFoundError(f"No images found at path: {path}")

        # Resolve corresponding label files
        if label_dir is not None and os.path.isdir(label_dir):
            self.label_files = [
                os.path.join(label_dir, Path(p).stem + ".txt") for p in self.im_files
            ]
        else:
            self.label_files = img2label_paths(self.im_files)

        self.n = len(self.im_files)
        self.indices = range(self.n)

        # RAM cache containers for resized images and dimensions
        self.ims: List[Optional[np.ndarray]] = [None] * self.n
        self.im_hw0: List[Optional[Tuple[int, int]]] = [None] * self.n
        self.im_hw: List[Optional[Tuple[int, int]]] = [None] * self.n

        # Load or build fast disk cache of labels (YOLOv5 .cache style) to avoid slow file I/O
        self.labels: List[Optional[np.ndarray]] = self._load_or_create_label_cache(path)

        # Optionally cache resized images in RAM (enabled on Linux/Kaggle or when requested)
        if self.cache_images and os.name != "nt":
            self._cache_images_to_ram(path)

    def _load_or_create_label_cache(
        self, path: Union[str, Path, List[str]]
    ) -> List[Optional[np.ndarray]]:
        """
        Cache parsed normalized YOLO labels into a single `.cache.pt` file so workers
        do not open thousands of small .txt files on disk across epochs.
        """
        cache_path = None
        if isinstance(path, (str, Path)):
            p_obj = Path(path)
            if p_obj.is_dir():
                cache_path = p_obj.parent / f"{p_obj.name}_labels.cache.pt"
            elif p_obj.is_file():
                cache_path = p_obj.with_suffix(".labels.cache.pt")

        if cache_path is not None and cache_path.exists():
            try:
                cache_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                if (
                    isinstance(cache_data, dict)
                    and cache_data.get("n") == self.n
                    and len(cache_data.get("labels", [])) == self.n
                ):
                    return cache_data["labels"]
            except Exception:
                pass

        labels: List[Optional[np.ndarray]] = [None] * self.n
        for i, lp in enumerate(self.label_files):
            if not os.path.exists(lp):
                labels[i] = np.zeros((0, 5), dtype=np.float32)
                continue
            try:
                with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [
                        ln.strip()
                        for ln in f.readlines()
                        if ln.strip() and not ln.strip().startswith("%")
                    ]
                if not lines:
                    labels[i] = np.zeros((0, 5), dtype=np.float32)
                    continue
                first_parts = lines[0].split()
                if len(first_parts) == 5:
                    vals = [float(v) for v in first_parts]
                    if all(0.0 <= v <= 1.0 for v in vals[1:]):
                        labels[i] = parse_label_file(lp, class_to_id=self.class_to_id)
            except Exception:
                labels[i] = None

        if cache_path is not None:
            try:
                torch.save({"n": self.n, "labels": labels}, cache_path)
            except Exception:
                pass
        return labels

    @staticmethod
    def _resolve_image_files(path: Union[str, Path, List[str]]) -> List[str]:
        """Resolve image paths from a directory, a .txt file list, or a list of paths."""
        files: List[str] = []
        for p in [path] if isinstance(path, (str, Path)) else path:
            p_obj = Path(p)
            if p_obj.is_dir():
                for root, _, fnames in os.walk(str(p_obj)):
                    for fn in fnames:
                        if is_image_file(fn):
                            files.append(os.path.join(root, fn))
            elif p_obj.is_file() and p_obj.suffix.lower() == ".txt":
                parent = p_obj.parent
                with open(p_obj, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        candidate = line if os.path.exists(line) else str(parent / line)
                        if os.path.exists(candidate):
                            files.append(candidate)
            elif p_obj.is_file() and is_image_file(str(p_obj)):
                files.append(str(p_obj))
        return sorted(files)

    def _preload_yolo_labels(self) -> None:
        """
        Pre-load labels that are already in normalized YOLO format.
        Pixel-format labels (DarkFace / ExDark raw) are parsed lazily when image dimensions (h0, w0) are read.
        """
        for i, lp in enumerate(self.label_files):
            if not os.path.exists(lp):
                self.labels[i] = np.zeros((0, 5), dtype=np.float32)
                continue
            try:
                with open(lp, "r", encoding="utf-8", errors="ignore") as f:
                    lines = [
                        ln.strip()
                        for ln in f.readlines()
                        if ln.strip() and not ln.strip().startswith("%")
                    ]
                if not lines:
                    self.labels[i] = np.zeros((0, 5), dtype=np.float32)
                    continue
                first_parts = lines[0].split()
                # Standard YOLO line has 5 floats with coordinates in [0, 1]
                if len(first_parts) == 5:
                    vals = [float(v) for v in first_parts]
                    if all(0.0 <= v <= 1.0 for v in vals[1:]):
                        self.labels[i] = parse_label_file(
                            lp, class_to_id=self.class_to_id
                        )
            except Exception:
                self.labels[i] = None

    def _get_labels(self, index: int, h0: int, w0: int) -> np.ndarray:
        """Return normalized [cls, xc, yc, w, h] labels for image at `index`."""
        lbl = self.labels[index]
        if lbl is None:
            lbl = parse_label_file(
                self.label_files[index],
                img_w=w0,
                img_h=h0,
                class_to_id=self.class_to_id,
            )
            self.labels[index] = lbl
        return lbl.copy()

    def __len__(self) -> int:
        return len(self.im_files)

    def _load_image_from_disk(
        self, i: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """Read and resize 1 image from disk at index `i`."""
        f = self.im_files[i]
        im = cv2.imread(f)  # BGR
        if im is None:
            raise FileNotFoundError(f"Image Not Found {f}")
        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = (
                cv2.INTER_LINEAR
                if (self.augment or r > 1)
                else cv2.INTER_AREA
            )
            im = cv2.resize(
                im, (int(w0 * r), int(h0 * r)), interpolation=interp
            )
        return im, (h0, w0), im.shape[:2]

    def _cache_images_to_ram(
        self,
        path: Union[str, Path, List[str]],
        max_gb: float = 6.0,
    ) -> None:
        """
        Pre-load resized images into RAM using a thread pool to eliminate disk I/O
        and repeated `cv2.imread` + `cv2.resize` overhead during Mosaic and Validation.
        Caps memory usage at `max_gb` per process so multi-GPU runs stay within system RAM.
        """
        max_bytes = int(max_gb * (1 << 30))
        used_bytes = 0
        any_label_resolved = False

        def _worker(idx: int):
            im, hw0, hw = self._load_image_from_disk(idx)
            lbl = self.labels[idx]
            if lbl is None:
                lbl = parse_label_file(
                    self.label_files[idx],
                    img_w=hw0[1],
                    img_h=hw0[0],
                    class_to_id=self.class_to_id,
                )
            return idx, im, hw0, hw, lbl

        with ThreadPoolExecutor(max_workers=8) as pool:
            for idx, im, hw0, hw, lbl in pool.map(_worker, range(self.n)):
                if self.labels[idx] is None and lbl is not None:
                    self.labels[idx] = lbl
                    any_label_resolved = True
                if used_bytes + im.nbytes <= max_bytes:
                    self.ims[idx] = im
                    self.im_hw0[idx] = hw0
                    self.im_hw[idx] = hw
                    used_bytes += im.nbytes

        if any_label_resolved and isinstance(path, (str, Path)):
            p_obj = Path(path)
            cache_path = (
                p_obj.parent / f"{p_obj.name}_labels.cache.pt"
                if p_obj.is_dir()
                else p_obj.with_suffix(".labels.cache.pt")
            )
            try:
                torch.save({"n": self.n, "labels": self.labels}, cache_path)
            except Exception:
                pass

    def load_image(self, i: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        """
        Load 1 image from dataset index `i` (from RAM cache if available, otherwise disk).
        Returns `(im, (h0, w0), (h, w))` where `im` is BGR uint8.
        """
        im = self.ims[i]
        if im is None:
            return self._load_image_from_disk(i)
        return im, self.im_hw0[i], self.im_hw[i]

    def load_mosaic(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic."""
        labels4 = []
        s = self.img_size
        yc, xc = (
            int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border
        )  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)

        for i, idx in enumerate(indices):
            img, (h0, w0), (h, w) = self.load_image(idx)

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    0,
                    w,
                    min(y2a - y1a, h),
                )
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = (
                    xc,
                    yc,
                    min(xc + w, s * 2),
                    min(s * 2, yc + h),
                )
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            labels = self._get_labels(idx, h0, w0)
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:], w, h, padw, padh
                )  # normalized xywh to pixel xyxy
            labels4.append(labels)

        # Concat/clip labels
        labels4 = np.concatenate(labels4, 0)
        for x in labels4[:, 1:]:
            np.clip(x, 0, 2 * s, out=x)

        # Augment with random perspective
        img4, labels4 = random_perspective(
            img4,
            labels4,
            degrees=self.hyp["degrees"],
            translate=self.hyp["translate"],
            scale=self.hyp["scale"],
            shear=self.hyp["shear"],
            perspective=self.hyp["perspective"],
            border=self.mosaic_border,
        )  # border to remove

        return img4, labels4

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, str, Tuple[Tuple[int, int], Tuple[Tuple[float, float], Tuple[float, float]]]]:
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]

        if mosaic:
            # Load 4-image mosaic
            img, labels = self.load_mosaic(index)
            shapes = None

            # MixUp augmentation
            if random.random() < hyp["mixup"]:
                img2, labels2 = self.load_mosaic(random.randint(0, self.n - 1))
                r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
                img = (img * r + img2 * (1 - r)).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load single image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            img, ratio, pad = letterbox(
                img,
                self.img_size,
                auto=False,
                scaleup=self.augment,
                stride=self.stride,
            )
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

            labels = self._get_labels(index, h0, w0)
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(
                    labels[:, 1:],
                    ratio[0] * w,
                    ratio[1] * h,
                    padw=pad[0],
                    padh=pad[1],
                )

            if self.augment:
                img, labels = random_perspective(
                    img,
                    labels,
                    degrees=hyp["degrees"],
                    translate=hyp["translate"],
                    scale=hyp["scale"],
                    shear=hyp["shear"],
                    perspective=hyp["perspective"],
                )

        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(
                labels[:, 1:5],
                w=img.shape[1],
                h=img.shape[0],
                clip=True,
                eps=1e-3,
            )

        if self.augment:
            # Random Gamma curve for low-light illumination generalization
            if hyp.get("gamma", False):
                img = random_gamma(
                    img,
                    start_gamma=int(hyp.get("start_gamma", 60)),
                    end_gamma=int(hyp.get("end_gamma", 120)),
                )

            # HSV color-space jitter
            augment_hsv(
                img,
                hgain=hyp["hsv_h"],
                sgain=hyp["hsv_s"],
                vgain=hyp["hsv_v"],
            )

            # Flip up-down
            if random.random() < hyp["flipud"]:
                img = np.flipud(img)
                if nl:
                    labels[:, 2] = 1 - labels[:, 2]

            # Flip left-right
            if random.random() < hyp["fliplr"]:
                img = np.fliplr(img)
                if nl:
                    labels[:, 1] = 1 - labels[:, 1]

        labels_out = torch.zeros((nl, 6), dtype=torch.float32)
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)

        # Convert BGR HWC to RGB CHW, float32 in [0.0, 1.0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0

        return torch.from_numpy(img), labels_out, self.im_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        """
        Standard YOLOv5 DataLoader collate_fn.
        Assigns batch index `i` to `label[:, 0]` before concatenating targets.
        """
        im, label, path, shapes = zip(*batch)  # transposed
        for i, lb in enumerate(label):
            lb[:, 0] = i  # add target image index for build_targets()
        return torch.stack(im, 0), torch.cat(label, 0), path, shapes


class ExDarkDataset(YOLODataset):
    """
    Exclusively Dark (ExDark) Dataset Loader (12 classes).
    Supports:
      - `data_dir` pointing directly to `dataset/ExDark/images/train` (or `val`, `test`)
      - `data_dir` pointing to `dataset/ExDark` root + `split in {'train', 'val', 'test'}`
      - `data_dir` pointing to `dataset/ExDark/train.txt`
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "./dataset/ExDark",
        split: str = "train",
        img_size: int = 640,
        augment: Optional[bool] = None,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        cache_images: Optional[bool] = None,
    ):
        data_path = Path(data_dir)
        if augment is None:
            augment = split == "train"

        # Resolve split directory or txt file if root folder is given
        if data_path.is_dir() and (data_path / "images" / split).exists():
            resolved_path = data_path / "images" / split
        elif data_path.is_dir() and (data_path / f"{split}.txt").exists():
            resolved_path = data_path / f"{split}.txt"
        else:
            resolved_path = data_path

        super().__init__(
            path=resolved_path,
            img_size=img_size,
            augment=augment,
            hyp=hyp,
            class_names=EXDARK_CLASSES,
            cache_images=cache_images,
        )


class DarkFaceDataset(YOLODataset):
    """
    DarkFace Low-Light Face Detection Dataset Loader (1 class: 'Face').
    Supports:
      - Unsplit layout: `dataset/DarkFace/image/*.png` + `dataset/DarkFace/label/*.txt`
        (automatically performs a deterministic 85% train / 15% val split based on sorted filenames)
      - Split layout: `dataset/DarkFace/images/{train,val}` + `dataset/DarkFace/labels/{train,val}`
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "./dataset/DarkFace",
        split: str = "train",
        img_size: int = 640,
        augment: Optional[bool] = None,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        val_ratio: float = 0.15,
        seed: int = 42,
        cache_images: Optional[bool] = None,
    ):
        data_path = Path(data_dir)
        if augment is None:
            augment = split == "train"

        label_dir = None
        if data_path.is_dir() and (data_path / "images" / split).exists():
            resolved_path: Union[Path, List[str]] = data_path / "images" / split
        elif data_path.is_dir() and (data_path / "image").exists():
            all_imgs = self._resolve_image_files(data_path / "image")
            rng = random.Random(seed)
            shuffled = list(all_imgs)
            rng.shuffle(shuffled)
            n_val = max(1, int(len(shuffled) * val_ratio))
            if split == "train":
                resolved_path = sorted(shuffled[n_val:])
            elif split in {"val", "test"}:
                resolved_path = sorted(shuffled[:n_val])
            else:
                resolved_path = all_imgs
            if (data_path / "label").exists():
                label_dir = str(data_path / "label")
        else:
            resolved_path = data_path

        super().__init__(
            path=resolved_path,
            img_size=img_size,
            augment=augment,
            hyp=hyp,
            class_names=DARKFACE_CLASSES,
            label_dir=label_dir,
            cache_images=cache_images,
        )


# Standalone alias for DataLoader collate_fn
collate_fn = YOLODataset.collate_fn
