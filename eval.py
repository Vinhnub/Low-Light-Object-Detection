"""
Evaluation & Testing Script (`eval.py`) for Low-Light Object Detection (YOLOv5).

Supports evaluating any trained checkpoint (`.pth` / `.pt`) on:
  - ExDark official splits (`--split test` [2,563 images] or `--split val` [1,800 images])
  - DarkFace, VOC, or custom image directories

Computes academic metrics (`mAP@0.50`, `mAP@0.75`, `mAP@0.50:0.95`, per-class AP,
Precision, Recall, F1) using `ObjectDetectionEvaluator` in `measure.py`, and optionally
exports `.txt` prediction files, JSON summaries, and visualized bounding-box images.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data import (
    get_darkface_val_set,
    get_exdark_test_set,
    get_exdark_val_set,
)
from data.dataset import YOLODataset, collate_fn
from data.util import DARKFACE_CLASSES, EXDARK_CLASSES, xywhn2xyxy
from measure import (
    VOC_CLASSES,
    ObjectDetectionEvaluator,
    print_academic_report,
)
from net.Yolov5 import (
    DetectionModel,
    load_yolov5_checkpoint,
    non_max_suppression,
    scale_boxes,
    yolov5l,
    yolov5m,
    yolov5n,
    yolov5s,
    yolov5x,
)


COLOR_PALETTE = [
    (0, 255, 255), (255, 128, 0), (50, 255, 50), (0, 165, 255),
    (255, 0, 255), (255, 255, 0), (0, 75, 255), (180, 105, 255),
    (0, 215, 255), (226, 43, 138), (128, 255, 0), (0, 128, 255),
]


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_class_names(dataset: str, num_classes: Optional[int] = None) -> List[str]:
    ds = dataset.lower()
    if ds == "exdark":
        return EXDARK_CLASSES
    if ds == "darkface":
        return DARKFACE_CLASSES
    if ds == "voc":
        return VOC_CLASSES
    if num_classes is None:
        return EXDARK_CLASSES
    return [str(i) for i in range(num_classes)]


def build_eval_model(
    weights_path: str,
    model_name: Optional[str] = None,
    num_classes: Optional[int] = None,
    modular: bool = False,
    cfg: str = "",
    device: str = "cuda",
    fuse: bool = False,
) -> tuple:
    """Build YOLOv5 architecture and load weights from checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Peek checkpoint metadata if available so model_name / num_classes auto-match
    ckpt_meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_meta, dict):
        if not model_name:
            model_name = ckpt_meta.get("model_name", "yolov5s")
        if num_classes is None and "num_classes" in ckpt_meta:
            num_classes = int(ckpt_meta["num_classes"])

    model_name = (model_name or "yolov5s").lower()
    num_classes = num_classes or 12

    model_builders = {
        "yolov5n": yolov5n,
        "yolov5s": yolov5s,
        "yolov5m": yolov5m,
        "yolov5l": yolov5l,
        "yolov5x": yolov5x,
    }

    if cfg and os.path.exists(cfg):
        model = DetectionModel(cfg=cfg, ch=3, nc=num_classes)
    else:
        builder = model_builders.get(model_name, yolov5s)
        model = builder(num_classes=num_classes, in_channels=3, modular=modular)

    load_yolov5_checkpoint(model, weights_path, model_name=model_name, verbose=True)
    model = model.to(device).eval()

    if fuse and hasattr(model, "fuse"):
        model.fuse()

    return model, model_name, num_classes


def draw_detections(
    img_path: str,
    detections: np.ndarray,
    class_names: List[str],
    save_path: str,
    vis_conf: float = 0.25,
) -> None:
    """Draw predicted bounding boxes above `vis_conf` onto the original image."""
    img = cv2.imread(img_path)
    if img is None:
        return

    for *xyxy, conf, cls_id in detections:
        if conf < vis_conf:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        cid = int(cls_id)
        color = COLOR_PALETTE[cid % len(COLOR_PALETTE)]
        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        label = f"{cname} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty1 = max(y1, th + baseline + 4)
        cv2.rectangle(img, (x1, ty1 - th - baseline - 4), (x1 + tw + 4, ty1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, ty1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def eval(
    model: torch.nn.Module,
    testing_data_loader: DataLoader,
    model_path: Optional[str] = None,
    output_folder: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    device: str = "cuda",
    conf_thres: float = 0.001,
    iou_thres: float = 0.6,
    max_det: int = 300,
    method: str = "coco",
    half: bool = False,
    save_txt: bool = True,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    vis_conf: float = 0.25,
    desc: str = "Evaluating",
) -> Dict:
    """
    Unified evaluation function for both `train.py`/`train_ddp.py` and standalone `eval.py`.
    Runs YOLOv5 inference, NMS, coordinate rescaling to original image resolution,
    exports prediction `.txt` files (and optional visualizations), and computes
    `mAP@0.50`, `mAP@0.75`, `mAP@0.50:0.95` via `ObjectDetectionEvaluator` in `measure.py`.
    """
    if class_names is None:
        class_names = EXDARK_CLASSES

    if model_path is not None and os.path.exists(model_path):
        load_yolov5_checkpoint(model, model_path, verbose=False)
        print(f"===> Loaded checkpoint for evaluation: {model_path}")

    torch.set_grad_enabled(False)
    model.eval()
    use_half = half and device.startswith("cuda")
    if use_half:
        model.half()

    evaluator = ObjectDetectionEvaluator(class_names=class_names, method=method)

    if save_txt and output_folder:
        os.makedirs(output_folder, exist_ok=True)
    if save_vis and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)

    total_images = 0
    total_infer_time = 0.0

    for batch in tqdm(testing_data_loader, desc=desc):
        imgs, targets, paths, shapes = batch[0], batch[1], batch[2], batch[3]
        imgs = imgs.to(device, non_blocking=True)
        if use_half:
            imgs = imgs.half()

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        pred_out, _ = model(imgs)
        dets_batch = non_max_suppression(
            pred_out.float(),
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )

        if device.startswith("cuda"):
            torch.cuda.synchronize()
        total_infer_time += time.perf_counter() - t0
        total_images += imgs.shape[0]

        for si, det in enumerate(dets_batch):
            img_path = paths[si]
            stem = Path(img_path).stem
            h0, w0 = shapes[si][0]
            ratio_pad = shapes[si][1]

            # Add Ground Truth boxes rescaled to original pixel dimensions
            t_si = targets[targets[:, 0] == si]
            if len(t_si):
                t_np = t_si[:, 1:].cpu().numpy()
                t_xyxy = xywhn2xyxy(
                    t_np[:, 1:5], w=imgs.shape[3], h=imgs.shape[2]
                )
                t_xyxy_orig = scale_boxes(
                    imgs.shape[2:],
                    torch.from_numpy(t_xyxy),
                    (h0, w0),
                    ratio_pad=ratio_pad,
                ).numpy()
                for cid, box in zip(t_np[:, 0].astype(int), t_xyxy_orig):
                    evaluator.add_ground_truth(stem, int(cid), box.tolist())
            else:
                evaluator.image_ids.add(stem)

            pred_lines = []
            det_np = np.zeros((0, 6), dtype=np.float32)
            if len(det):
                det_clone = det.clone()
                det_clone[:, :4] = scale_boxes(
                    imgs.shape[2:],
                    det_clone[:, :4],
                    (h0, w0),
                    ratio_pad=ratio_pad,
                )
                det_np = det_clone.cpu().numpy()
                for *xyxy, conf, cls_id in det_np:
                    cid = int(cls_id)
                    score = float(conf)
                    box = [float(v) for v in xyxy]
                    evaluator.add_prediction(stem, cid, score, box)
                    cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                    pred_lines.append(
                        f"{cname} {score:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}\n"
                    )

            if save_txt and output_folder:
                with open(
                    os.path.join(output_folder, f"{stem}.txt"), "w", encoding="utf-8"
                ) as pf:
                    pf.writelines(pred_lines)

            if save_vis and vis_dir:
                vis_path = os.path.join(vis_dir, Path(img_path).name)
                draw_detections(
                    img_path, det_np, class_names, vis_path, vis_conf=vis_conf
                )

    torch.set_grad_enabled(True)
    results = evaluator.evaluate(conf_threshold=conf_thres)
    print_academic_report(results, class_names)

    if total_images > 0 and total_infer_time > 0:
        ms_per_img = (total_infer_time / total_images) * 1000.0
        fps = total_images / total_infer_time
        print(
            f"===> Inference Speed ({total_images} images): "
            f"{ms_per_img:.2f} ms/img | {fps:.2f} FPS (incl. NMS)"
        )
        results["speed_ms_per_img"] = ms_per_img
        results["fps"] = fps

    return results


# Backward-compatible aliases
eval_model = eval
evaluate_detection = eval


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate / Test YOLOv5 on Low-Light Object Detection Benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained checkpoint (.pth / .pt), e.g. weights/training/epoch_100_best_mAP50.pth",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="exdark",
        choices=["exdark", "darkface", "voc", "custom"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "train"],
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="Optional override path to images directory or .txt split file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="YOLOv5 variant (auto-detected from checkpoint if omitted)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional path to custom model YAML file",
    )
    parser.add_argument(
        "--modular",
        type=_str2bool,
        default=False,
        help="Whether checkpoint uses modular YOLOv5Backbone+YOLOv5Neck+Detect",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from dataset/checkpoint if omitted)",
    )
    parser.add_argument("--imgSize", type=int, default=640, help="Inference image size")
    parser.add_argument("--batchSize", type=int, default=16, help="Evaluation batch size")
    parser.add_argument("--threads", type=int, default=0, help="DataLoader workers (0 avoids Windows paging file overhead)")
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.001,
        help="Confidence threshold for NMS during mAP evaluation",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.6,
        help="IoU threshold for NMS during mAP evaluation",
    )
    parser.add_argument("--max_det", type=int, default=300, help="Max detections per image")
    parser.add_argument(
        "--method",
        type=str,
        default="coco",
        choices=["coco", "voc"],
        help="AP interpolation protocol: 'coco' (101-point) or 'voc' (continuous AUC)",
    )
    parser.add_argument(
        "--fuse",
        type=_str2bool,
        default=True,
        help="Fuse Conv2d and BatchNorm2d layers for faster inference",
    )
    parser.add_argument(
        "--half",
        type=_str2bool,
        default=False,
        help="Use FP16 half precision on CUDA",
    )
    parser.add_argument("--gpu_mode", type=_str2bool, default=True, help="Use CUDA GPU")
    parser.add_argument(
        "--save_txt",
        type=_str2bool,
        default=True,
        help="Save per-image prediction .txt files",
    )
    parser.add_argument(
        "--save_vis",
        type=_str2bool,
        default=False,
        help="Save visualized images with predicted bounding boxes",
    )
    parser.add_argument(
        "--vis_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for drawing boxes when --save_vis True",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Directory to save predictions/visualizations (default: ./results/eval/<dataset>_<split>)",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional path to save evaluation metrics as JSON",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = "cuda" if (args.gpu_mode and torch.cuda.is_available()) else "cpu"
    if args.gpu_mode and not torch.cuda.is_available():
        print("[Warn] CUDA not available, falling back to CPU.")

    ds_name = args.dataset.lower()
    if args.num_classes is None:
        if ds_name == "exdark":
            args.num_classes = len(EXDARK_CLASSES)
        elif ds_name == "darkface":
            args.num_classes = len(DARKFACE_CLASSES)
        elif ds_name == "voc":
            args.num_classes = len(VOC_CLASSES)

    model, model_name, num_classes = build_eval_model(
        weights_path=args.weights,
        model_name=args.model or None,
        num_classes=args.num_classes,
        modular=args.modular,
        cfg=args.cfg,
        device=device,
        fuse=args.fuse,
    )
    class_names = get_class_names(ds_name, num_classes=num_classes)

    # Load target split dataset
    if ds_name == "exdark":
        if args.split == "test":
            path = args.data_dir or "./dataset/ExDark/images/test"
            dataset = get_exdark_test_set(path, size=args.imgSize)
        elif args.split == "val":
            path = args.data_dir or "./dataset/ExDark/images/val"
            dataset = get_exdark_val_set(path, size=args.imgSize)
        else:
            path = args.data_dir or "./dataset/ExDark/images/train"
            dataset = YOLODataset(
                path=path, img_size=args.imgSize, augment=False, class_names=class_names
            )
    elif ds_name == "darkface":
        path = args.data_dir or "./dataset/DarkFace"
        dataset = get_darkface_val_set(path, size=args.imgSize)
    else:
        default_dir = (
            f"./dataset/{args.dataset.upper()}/images/{args.split}"
            if ds_name == "voc"
            else f"./dataset/custom/images/{args.split}"
        )
        path = args.data_dir or default_dir
        dataset = YOLODataset(
            path=path, img_size=args.imgSize, augment=False, class_names=class_names
        )

    pin_memory = device.startswith("cuda")
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=min(args.batchSize, len(dataset)),
        shuffle=False,
        num_workers=args.threads,
        pin_memory=pin_memory,
        persistent_workers=(args.threads > 0),
        collate_fn=collate_fn,
    )

    out_root = args.output_dir or os.path.join("results", "eval", f"{ds_name}_{args.split}")
    pred_dir = os.path.join(out_root, "preds") if args.save_txt else None
    vis_dir = os.path.join(out_root, "visualized") if args.save_vis else None

    print(
        f"===> Testing {model_name} on {ds_name} [{args.split}] "
        f"({len(dataset)} images, imgSize={args.imgSize}, device={device})"
    )

    results = eval(
        model=model,
        testing_data_loader=data_loader,
        model_path=None,
        output_folder=pred_dir,
        class_names=class_names,
        device=device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        method=args.method,
        half=args.half,
        save_txt=args.save_txt,
        save_vis=args.save_vis,
        vis_dir=vis_dir,
        vis_conf=args.vis_conf,
        desc=f"Testing [{args.split}]",
    )

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=4)
        print(f"===> Evaluation JSON saved to: {args.save_json}")
