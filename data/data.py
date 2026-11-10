"""
Dataset and DataLoader Builder Functions (`data/data.py`).
Follows the structure of IGMamba-EdgeLCA-LLIE (`data/data.py`) while supporting
YOLOv5 Object Detection datasets (ExDark, DarkFace, VOC, Custom YOLO).
"""

from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import (
    DarkFaceDataset,
    ExDarkDataset,
    YOLODataset,
    collate_fn,
)


def get_exdark_training_set(
    data_dir: str = "./dataset/ExDark/images/train",
    size: int = 640,
    hyp: Optional[Union[Dict[str, Any], Any]] = None,
) -> ExDarkDataset:
    """Return ExDark training dataset with Mosaic + HSV + affine augmentations enabled."""
    return ExDarkDataset(
        data_dir=data_dir,
        split="train",
        img_size=size,
        augment=True,
        hyp=hyp,
    )


def get_exdark_val_set(
    data_dir: str = "./dataset/ExDark/images/val",
    size: int = 640,
) -> ExDarkDataset:
    """Return ExDark validation dataset (letterbox only, no random augmentation)."""
    return ExDarkDataset(
        data_dir=data_dir,
        split="val",
        img_size=size,
        augment=False,
    )


def get_exdark_test_set(
    data_dir: str = "./dataset/ExDark/images/test",
    size: int = 640,
) -> ExDarkDataset:
    """Return ExDark official test dataset (letterbox only, no random augmentation)."""
    return ExDarkDataset(
        data_dir=data_dir,
        split="test",
        img_size=size,
        augment=False,
    )


def get_darkface_training_set(
    data_dir: str = "./dataset/DarkFace",
    size: int = 640,
    hyp: Optional[Union[Dict[str, Any], Any]] = None,
) -> DarkFaceDataset:
    """Return DarkFace training dataset with augmentations enabled."""
    return DarkFaceDataset(
        data_dir=data_dir,
        split="train",
        img_size=size,
        augment=True,
        hyp=hyp,
    )


def get_darkface_val_set(
    data_dir: str = "./dataset/DarkFace",
    size: int = 640,
) -> DarkFaceDataset:
    """Return DarkFace validation dataset (letterbox only)."""
    return DarkFaceDataset(
        data_dir=data_dir,
        split="val",
        img_size=size,
        augment=False,
    )


def get_training_set(
    data_dir: str,
    size: int = 640,
    dataset_name: str = "exdark",
    hyp: Optional[Union[Dict[str, Any], Any]] = None,
) -> Dataset:
    """Generic training set builder based on dataset name."""
    name = dataset_name.lower()
    if name == "exdark":
        return get_exdark_training_set(data_dir, size=size, hyp=hyp)
    elif name == "darkface":
        return get_darkface_training_set(data_dir, size=size, hyp=hyp)
    else:
        return YOLODataset(
            path=data_dir,
            img_size=size,
            augment=True,
            hyp=hyp,
        )


def get_eval_set(
    data_dir: str,
    size: int = 640,
    dataset_name: str = "exdark",
) -> Dataset:
    """Generic evaluation/validation set builder based on dataset name."""
    name = dataset_name.lower()
    if name == "exdark":
        return get_exdark_val_set(data_dir, size=size)
    elif name == "darkface":
        return get_darkface_val_set(data_dir, size=size)
    else:
        return YOLODataset(
            path=data_dir,
            img_size=size,
            augment=False,
        )


def create_dataloader(
    opt: Any,
    split: str = "train",
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
) -> Tuple[DataLoader, Dataset]:
    """
    Create PyTorch DataLoader and Dataset directly from `opt` (parsed by `data/options.py`).

    Args:
        opt: Parsed arguments from `option().parse_args()`.
        split (str): One of `'train'`, `'val'`, or `'test'`.
        batch_size (int, optional): Override `opt.batchSize`.
        shuffle (bool, optional): Override `opt.shuffle` (defaults to True for train, False for val/test).

    Returns:
        dataloader (DataLoader): PyTorch DataLoader using `YOLODataset.collate_fn`.
        dataset (Dataset): Underlying Dataset instance.
    """
    ds_name = getattr(opt, "dataset", "exdark").lower()
    img_size = getattr(opt, "imgSize", 640)
    bs = batch_size if batch_size is not None else getattr(opt, "batchSize", 16)
    workers = getattr(opt, "threads", 4)
    pin_memory = getattr(opt, "gpu_mode", True) and torch.cuda.is_available()

    if shuffle is None:
        shuffle = getattr(opt, "shuffle", True) if split == "train" else False

    if ds_name == "exdark":
        if split == "train":
            path = getattr(opt, "data_train_exdark", "./dataset/ExDark/images/train")
            dataset = get_exdark_training_set(path, size=img_size, hyp=opt)
        elif split == "val":
            path = getattr(opt, "data_val_exdark", "./dataset/ExDark/images/val")
            dataset = get_exdark_val_set(path, size=img_size)
        else:
            path = getattr(opt, "data_test_exdark", "./dataset/ExDark/images/test")
            dataset = get_exdark_test_set(path, size=img_size)
    elif ds_name == "darkface":
        if split == "train":
            path = getattr(opt, "data_train_darkface", "./dataset/DarkFace")
            dataset = get_darkface_training_set(path, size=img_size, hyp=opt)
        else:
            path = getattr(opt, "data_val_darkface", "./dataset/DarkFace")
            dataset = get_darkface_val_set(path, size=img_size)
    elif ds_name == "voc":
        path = (
            getattr(opt, "data_train_voc", "./dataset/VOC/images/train")
            if split == "train"
            else getattr(opt, "data_val_voc", "./dataset/VOC/images/val")
        )
        dataset = YOLODataset(
            path=path,
            img_size=img_size,
            augment=(split == "train"),
            hyp=opt,
        )
    else:
        path = (
            getattr(opt, "data_train_custom", "./dataset/custom/images/train")
            if split == "train"
            else getattr(opt, "data_val_custom", "./dataset/custom/images/val")
        )
        dataset = YOLODataset(
            path=path,
            img_size=img_size,
            augment=(split == "train"),
            hyp=opt,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=min(bs, len(dataset)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=(workers > 0),
        prefetch_factor=(4 if workers > 0 else None),
        collate_fn=collate_fn,
        drop_last=False,
    )
    return dataloader, dataset
