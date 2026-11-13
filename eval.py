"""
Evaluation & Testing Script (`eval.py`) for Low-Light Object Detection (YOLOv5).

Supports evaluating any trained checkpoint (`.pth` / `.pt`) on:
  - ExDark official splits (`--split test` [2,563 images] or `--split val` [1,800 images])
  - DarkFace, VOC, or custom image directories

Computes academic metrics (`mAP@0.50`, `mAP@0.75`, `mAP@0.50:0.95`, per-class AP,
Precision, Recall, F1) using `ObjectDetectionEvaluator` in `measure.py`, and optionally
exports `.txt` prediction files, JSON summaries, visualized bounding-box images, and
side-by-side Ground Truth vs Output comparison images.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from data.util import DARKFACE_CLASSES, EXDARK_CLASSES, parse_label_file, xywhn2xyxy
from measure import VOC_CLASSES, metrics
from net import (
    SUPPORTED_MODELS,
    build_yolo_model,
    infer_model_from_checkpoint,
    load_yolo_checkpoint,
    non_max_suppression,
)
from net.Yolov11.utils import (
    _register_ultralytics_unpickle_aliases,
    scale_boxes,
    xywh2xyxy,
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
    """Build YOLOv5 / YOLOv11 / YOLOv26 architecture and load weights from checkpoint."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    # Peek checkpoint metadata if available so model_name / num_classes auto-match
    _register_ultralytics_unpickle_aliases()
    ckpt_meta = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not model_name:
        model_name = infer_model_from_checkpoint(ckpt_meta, fallback_path=weights_path)
    if num_classes is None and isinstance(ckpt_meta, dict):
        if "num_classes" in ckpt_meta:
            num_classes = int(ckpt_meta["num_classes"])
        elif hasattr(ckpt_meta.get("model"), "nc"):
            num_classes = int(ckpt_meta["model"].nc)
        elif isinstance(getattr(ckpt_meta.get("model"), "yaml", None), dict) and "nc" in ckpt_meta["model"].yaml:
            num_classes = int(ckpt_meta["model"].yaml["nc"])

    model_name = (model_name or "yolov5s").lower()
    num_classes = num_classes or 12

    cfg_path = cfg if (cfg and os.path.exists(cfg)) else None
    model = build_yolo_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=3,
        cfg=cfg_path,
        modular=modular,
    )

    load_yolo_checkpoint(model, weights_path, model_name=model_name, verbose=True)
    model = model.to(device).eval()

    if fuse and hasattr(model, "fuse"):
        model.fuse()

    return model, model_name, num_classes


def _ensure_parent_dir(file_path: str) -> None:
    """Create parent directory for `file_path` if non-empty."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    class_names: List[str],
    show_conf: bool = True,
    conf_thres: float = 0.0,
    line_thickness: int = 1,
    show_labels: bool = True,
) -> Tuple[np.ndarray, int]:
    """
    Draw bounding boxes onto a copy of `img`.

    Args:
        img: BGR uint8 image array.
        boxes: Array of shape (N, 6) `[x1, y1, x2, y2, conf, cls_id]`
               or (N, 5) `[x1, y1, x2, y2, cls_id]`.
        class_names: List of class names indexed by `cls_id`.
        show_conf: Whether to filter by `conf_thres` and append confidence to labels.
        conf_thres: Minimum confidence when `show_conf` is True.
        line_thickness: Bounding box outline thickness in pixels (default: 1).
        show_labels: Whether to draw compact semi-transparent class/conf tags.

    Returns:
        `(annotated_img, num_drawn_boxes)`.
    """
    canvas = img.copy()
    if boxes is None or len(boxes) == 0:
        return canvas, 0

    h, w = canvas.shape[:2]
    font_scale = max(0.32, min(0.45, min(h, w) / 1400.0))
    font_thickness = 1
    box_thickness = max(1, int(line_thickness))
    drawn = 0

    for row in boxes:
        x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
        conf = float(row[4]) if len(row) >= 6 else 1.0
        cls_id = int(row[5]) if len(row) >= 6 else int(row[4])

        if show_conf and conf < conf_thres:
            continue

        x1_i = max(0, min(int(round(float(x1))), w - 1))
        y1_i = max(0, min(int(round(float(y1))), h - 1))
        x2_i = max(0, min(int(round(float(x2))), w - 1))
        y2_i = max(0, min(int(round(float(y2))), h - 1))
        if x2_i <= x1_i or y2_i <= y1_i:
            continue

        color = COLOR_PALETTE[cls_id % len(COLOR_PALETTE)]
        cv2.rectangle(canvas, (x1_i, y1_i), (x2_i, y2_i), color, box_thickness)

        if show_labels:
            cname = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            label = f"{cname} {conf:.2f}" if show_conf else cname
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            tag_h = th + baseline + 2
            tag_w = tw + 4
            by2 = min(h, max(y1_i, tag_h))
            by1 = max(0, by2 - tag_h)
            bx1 = min(x1_i, max(0, w - tag_w))
            bx2 = min(w, bx1 + tag_w)

            if bx2 > bx1 and by2 > by1:
                roi = canvas[by1:by2, bx1:bx2]
                color_arr = np.array(color, dtype=np.float32)
                blended = (roi.astype(np.float32) * 0.35 + color_arr * 0.65).astype(
                    np.uint8
                )
                roi[:] = blended
                mean_bgr = blended.reshape(-1, 3).mean(axis=0)
                brightness = (
                    mean_bgr[0] * 0.114 + mean_bgr[1] * 0.587 + mean_bgr[2] * 0.299
                )
                text_color = (0, 0, 0) if brightness > 135 else (255, 255, 255)
                cv2.putText(
                    canvas,
                    label,
                    (bx1 + 2, by2 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA,
                )

        drawn += 1

    return canvas, drawn


def _put_header_text(
    canvas: np.ndarray,
    text: str,
    fallback_text: str,
    x_offset: int,
    panel_w: int,
    header_h: int,
    color: Tuple[int, int, int],
) -> None:
    """Draw centered, auto-fitted header text inside a panel header bar."""
    max_w = max(20, panel_w - 16)
    scale = min(0.65, max(0.35, panel_w / 600.0))
    thickness = 2 if scale >= 0.55 else 1
    chosen = text
    (tw, th), _ = cv2.getTextSize(chosen, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    if tw > max_w:
        chosen = fallback_text
        (tw, th), _ = cv2.getTextSize(
            chosen, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )
    while tw > max_w and scale > 0.28:
        scale -= 0.04
        thickness = 1
        (tw, th), _ = cv2.getTextSize(
            chosen, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
        )

    tx = x_offset + max(4, (panel_w - tw) // 2)
    ty = max(th + 2, (header_h + th) // 2 - 1)
    cv2.putText(
        canvas,
        chosen,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _resolve_gt_boxes(
    targets: Optional[torch.Tensor],
    batch_idx: int,
    in_shape: Tuple[int, int],
    orig_shape: Tuple[int, int],
    ratio_pad,
    img_path: str,
    gt_dir: Optional[str] = None,
    class_to_id: Optional[Dict[str, int]] = None,
) -> np.ndarray:
    """
    Reconstruct Ground Truth boxes `[x1, y1, x2, y2, 1.0, cls_id]` in original image
    coordinates from DataLoader `targets`, falling back to `gt_dir` if empty.
    """
    if targets is not None and len(targets) > 0:
        gt_rows = targets[targets[:, 0] == batch_idx]
        if len(gt_rows) > 0:
            h_in, w_in = in_shape
            boxes = xywh2xyxy(gt_rows[:, 2:6].detach().cpu().clone().float())
            boxes[:, [0, 2]] *= float(w_in)
            boxes[:, [1, 3]] *= float(h_in)
            boxes = scale_boxes(in_shape, boxes, orig_shape, ratio_pad=ratio_pad)
            cls_ids = gt_rows[:, 1:2].detach().cpu().float()
            confs = torch.ones_like(cls_ids)
            return torch.cat((boxes, confs, cls_ids), dim=1).numpy()

    if gt_dir and os.path.isdir(gt_dir):
        h0, w0 = orig_shape
        img_name = Path(img_path).name
        stem = Path(img_path).stem
        for cand in (
            os.path.join(gt_dir, f"{stem}.txt"),
            os.path.join(gt_dir, f"{img_name}.txt"),
        ):
            if os.path.exists(cand):
                parsed = parse_label_file(
                    cand, img_w=w0, img_h=h0, class_to_id=class_to_id
                )
                if len(parsed) > 0:
                    boxes_xyxy = xywhn2xyxy(parsed[:, 1:5], w=float(w0), h=float(h0))
                    confs = np.ones((len(parsed), 1), dtype=np.float32)
                    cls_ids = parsed[:, 0:1].astype(np.float32)
                    return np.hstack((boxes_xyxy, confs, cls_ids)).astype(np.float32)
                break

    return np.zeros((0, 6), dtype=np.float32)


def draw_detections(
    img_path: str,
    detections: np.ndarray,
    class_names: List[str],
    save_path: str,
    vis_conf: float = 0.25,
    img: Optional[np.ndarray] = None,
    line_thickness: int = 1,
    show_labels: bool = True,
) -> None:
    """Draw predicted bounding boxes above `vis_conf` onto the original image."""
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return

    annotated, _ = _draw_boxes(
        img,
        detections,
        class_names,
        show_conf=True,
        conf_thres=vis_conf,
        line_thickness=line_thickness,
        show_labels=show_labels,
    )
    _ensure_parent_dir(save_path)
    cv2.imwrite(save_path, annotated)


def draw_comparison(
    img_path: str,
    gt_boxes: np.ndarray,
    detections: np.ndarray,
    class_names: List[str],
    save_path: str,
    vis_conf: float = 0.25,
    img: Optional[np.ndarray] = None,
    line_thickness: int = 1,
    show_labels: bool = True,
) -> None:
    """
    Save a side-by-side comparison image with Ground Truth boxes on the left
    and predicted Output boxes (above `vis_conf`) on the right.
    """
    if img is None:
        img = cv2.imread(img_path)
    if img is None:
        return

    h, w = img.shape[:2]
    gt_panel, n_gt = _draw_boxes(
        img,
        gt_boxes,
        class_names,
        show_conf=False,
        line_thickness=line_thickness,
        show_labels=show_labels,
    )
    pred_panel, n_pred = _draw_boxes(
        img,
        detections,
        class_names,
        show_conf=True,
        conf_thres=vis_conf,
        line_thickness=line_thickness,
        show_labels=show_labels,
    )

    header_h = max(28, min(48, int(round(h * 0.08))))
    gap = 4
    canvas = np.full((h + header_h, w * 2 + gap, 3), 30, dtype=np.uint8)
    canvas[:, w : w + gap] = (140, 140, 140)
    cv2.line(canvas, (0, header_h - 1), (w * 2 + gap - 1, header_h - 1), (90, 90, 90), 1)

    canvas[header_h:, :w] = gt_panel
    canvas[header_h:, w + gap :] = pred_panel

    _put_header_text(
        canvas,
        f"Ground Truth ({n_gt})",
        f"GT ({n_gt})",
        0,
        w,
        header_h,
        (100, 255, 100),
    )
    _put_header_text(
        canvas,
        f"Output ({n_pred}, conf>={vis_conf:.2f})",
        f"Output ({n_pred})",
        w + gap,
        w,
        header_h,
        (0, 255, 255),
    )

    _ensure_parent_dir(save_path)
    cv2.imwrite(save_path, canvas)


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
    half: bool = False,
    save_txt: bool = True,
    save_vis: bool = False,
    vis_dir: Optional[str] = None,
    vis_conf: float = 0.25,
    save_compare: bool = False,
    compare_dir: Optional[str] = None,
    gt_dir: Optional[str] = None,
    vis_max: int = -1,
    vis_thickness: int = 1,
    vis_labels: bool = True,
    desc: str = "Evaluating",
    return_preds: bool = False,
    verbose: bool = True,
):
    """
    Run YOLOv5 inference, NMS, and coordinate rescaling to original image resolution,
    optionally saving prediction `.txt` files (`<class> <conf> <x1> <y1> <x2> <y2>`) into `output_folder`,
    single output visualizations (`save_vis`), side-by-side GT vs Output comparison images (`save_compare`),
    and/or returning in-memory predictions (`return_preds=True`) to avoid disk I/O overhead.
    Metric calculation is delegated strictly to `metrics()` in `measure.py`.
    """
    if class_names is None:
        class_names = EXDARK_CLASSES
    class_to_id = {name: i for i, name in enumerate(class_names)}
    for i, name in enumerate(class_names):
        class_to_id.setdefault(name.lower(), i)

    if model_path is not None and os.path.exists(model_path):
        head = getattr(model, "head", None) or (
            model.model[-1] if hasattr(model, "model") else None
        )
        if head is not None and getattr(head, "one2one_cv2", None) is not None:
            hint_name = "yolov26s"
        elif head is not None and hasattr(head, "cv2"):
            hint_name = "yolov11s"
        else:
            hint_name = infer_model_from_checkpoint({}, fallback_path=model_path) or "yolov5s"
        load_yolo_checkpoint(model, model_path, model_name=hint_name, verbose=False)
        if verbose:
            print(f"===> Pre-trained model is loaded from: {model_path}")

    was_training = model.training
    model.eval()
    is_cuda = str(device).startswith("cuda")
    use_half = half and is_cuda

    if save_txt and output_folder:
        os.makedirs(output_folder, exist_ok=True)
        # Clear previous .txt prediction files in output_folder so splits/epochs never mix
        for old_f in os.listdir(output_folder):
            if old_f.endswith(".txt"):
                try:
                    os.remove(os.path.join(output_folder, old_f))
                except OSError:
                    pass
    if save_vis and vis_dir is None and output_folder:
        vis_dir = os.path.join(os.path.dirname(os.path.normpath(output_folder)), "visualized")
    if save_compare and compare_dir is None and output_folder:
        compare_dir = os.path.join(os.path.dirname(os.path.normpath(output_folder)), "compare")

    if save_vis and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    if save_compare and compare_dir:
        os.makedirs(compare_dir, exist_ok=True)

    total_images = 0
    saved_vis_count = 0
    in_memory_preds: Dict[str, list] = {}
    iterator = tqdm(testing_data_loader, desc=desc) if verbose else testing_data_loader

    if is_cuda:
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    with torch.no_grad():
        for batch in iterator:
            imgs, targets, paths, shapes = batch[0], batch[1], batch[2], batch[3]
            imgs = imgs.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_half):
                pred_out, _ = model(imgs)

            is_e2e = bool(getattr(model, "end2end", False))
            dets_batch = non_max_suppression(
                pred_out.float(),
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                multi_label=(conf_thres <= 0.01) and not is_e2e,
                max_det=max_det,
                end2end=is_e2e,
            )
            total_images += imgs.shape[0]

            for si, det in enumerate(dets_batch):
                img_path = paths[si]
                stem = Path(img_path).stem
                h0, w0 = shapes[si][0]
                ratio_pad = shapes[si][1]

                pred_lines = []
                stem_preds = []
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
                    for row in det_np:
                        x1, y1, x2, y2, score, cls_id = (
                            float(row[0]),
                            float(row[1]),
                            float(row[2]),
                            float(row[3]),
                            float(row[4]),
                            int(row[5]),
                        )
                        if return_preds:
                            stem_preds.append((cls_id, score, [x1, y1, x2, y2]))
                        if save_txt and output_folder:
                            cname = (
                                class_names[cls_id]
                                if 0 <= cls_id < len(class_names)
                                else str(cls_id)
                            )
                            pred_lines.append(
                                f"{cname} {score:.6f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n"
                            )

                if return_preds:
                    in_memory_preds[stem] = stem_preds

                if save_txt and output_folder:
                    with open(
                        os.path.join(output_folder, f"{stem}.txt"), "w", encoding="utf-8"
                    ) as pf:
                        pf.writelines(pred_lines)

                do_vis = bool(save_vis and vis_dir)
                do_compare = bool(save_compare and compare_dir)
                if (do_vis or do_compare) and (vis_max < 0 or saved_vis_count < vis_max):
                    orig_img = cv2.imread(img_path)
                    if orig_img is not None:
                        img_name = Path(img_path).name
                        if do_vis:
                            vis_path = os.path.join(vis_dir, img_name)
                            draw_detections(
                                img_path,
                                det_np,
                                class_names,
                                vis_path,
                                vis_conf=vis_conf,
                                img=orig_img,
                                line_thickness=vis_thickness,
                                show_labels=vis_labels,
                            )
                        if do_compare:
                            gt_np = _resolve_gt_boxes(
                                targets=targets,
                                batch_idx=si,
                                in_shape=imgs.shape[2:],
                                orig_shape=(h0, w0),
                                ratio_pad=ratio_pad,
                                img_path=img_path,
                                gt_dir=gt_dir,
                                class_to_id=class_to_id,
                            )
                            compare_path = os.path.join(compare_dir, img_name)
                            draw_comparison(
                                img_path,
                                gt_np,
                                det_np,
                                class_names,
                                compare_path,
                                vis_conf=vis_conf,
                                img=orig_img,
                                line_thickness=vis_thickness,
                                show_labels=vis_labels,
                            )
                        saved_vis_count += 1

    if is_cuda:
        torch.cuda.synchronize()
    total_infer_time = time.perf_counter() - t_start

    if verbose and total_images > 0 and total_infer_time > 0:
        ms_per_img = (total_infer_time / total_images) * 1000.0
        fps = total_images / total_infer_time
        print(
            f"===> End evaluation ({total_images} images): "
            f"{ms_per_img:.2f} ms/img | {fps:.2f} FPS (incl. NMS)"
        )
    if verbose and save_vis and vis_dir:
        print(f"===> Visualized output images ({saved_vis_count}) saved to: {vis_dir}")
    if verbose and save_compare and compare_dir:
        print(f"===> Comparison (GT vs Output) images ({saved_vis_count}) saved to: {compare_dir}")

    if was_training:
        model.train()
    if return_preds:
        return in_memory_preds
    return output_folder or ""


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
        "--gt_dir",
        type=str,
        default="",
        help="Optional override path to ground truth labels directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        choices=[""] + list(SUPPORTED_MODELS),
        help="YOLO variant (yolov5*/yolov11*/yolo11*/yolov26*/yolo26*; auto-detected from checkpoint if omitted)",
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
        help="Whether checkpoint uses modular Backbone + Neck + Detect architecture",
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
        "--save_compare",
        type=_str2bool,
        default=False,
        help="Save side-by-side comparison images (Ground Truth vs Output bounding boxes)",
    )
    parser.add_argument(
        "--vis_conf",
        type=float,
        default=0.25,
        help="Confidence threshold for drawing predicted boxes when --save_vis or --save_compare is True",
    )
    parser.add_argument(
        "--vis_max",
        type=int,
        default=-1,
        help="Maximum number of visualized/compare images to save (-1 saves all)",
    )
    parser.add_argument(
        "--vis_thickness",
        type=int,
        default=1,
        help="Bounding box line thickness in pixels for --save_vis / --save_compare",
    )
    parser.add_argument(
        "--vis_labels",
        type=_str2bool,
        default=True,
        help="Draw compact class/confidence tags on boxes for --save_vis / --save_compare",
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

    if args.gt_dir:
        gt_dir = args.gt_dir
    elif ds_name == "exdark":
        gt_dir = f"./dataset/ExDark/labels_raw/{args.split}"
    elif ds_name == "darkface":
        gt_dir = "./dataset/DarkFace/label"
    elif ds_name == "voc":
        gt_dir = f"./dataset/VOC/labels/{args.split}"
    else:
        gt_dir = f"./dataset/custom/labels/{args.split}"

    out_root = args.output_dir or os.path.join("results", "eval", f"{ds_name}_{args.split}")
    pred_dir = os.path.join(out_root, "preds")
    vis_dir = os.path.join(out_root, "visualized") if args.save_vis else None
    compare_dir = os.path.join(out_root, "compare") if args.save_compare else None

    print(
        f"===> Testing {model_name} on {ds_name} [{args.split}] "
        f"({len(dataset)} images, imgSize={args.imgSize}, device={device})"
    )

    preds_dict = eval(
        model=model,
        testing_data_loader=data_loader,
        model_path=None,
        output_folder=pred_dir,
        class_names=class_names,
        device=device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        max_det=args.max_det,
        half=args.half,
        save_txt=args.save_txt,
        save_vis=args.save_vis,
        vis_dir=vis_dir,
        vis_conf=args.vis_conf,
        save_compare=args.save_compare,
        compare_dir=compare_dir,
        gt_dir=gt_dir if os.path.exists(gt_dir) else None,
        vis_max=args.vis_max,
        vis_thickness=args.vis_thickness,
        vis_labels=args.vis_labels,
        desc=f"Testing [{args.split}]",
        return_preds=True,
    )

    # Delegate metric calculation to measure.py
    if os.path.exists(gt_dir):
        results = metrics(
            pred_dir=pred_dir,
            label_dir=gt_dir,
            class_names=class_names,
            dataset_type=ds_name,
            method=args.method,
            conf_thres=args.conf_thres,
            in_memory_preds=preds_dict,
        )
        if args.save_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
            with open(args.save_json, "w", encoding="utf-8") as jf:
                json.dump(results, jf, indent=4)
            print(f"===> Evaluation JSON saved to: {args.save_json}")
