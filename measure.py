"""
====================================================================================================
Academic Evaluation Tool for Object Detection: mAP@0.50, mAP@0.75, mAP@0.50:0.95 (COCO Standard)
Designed for Low-Light Object Detection benchmarks (ExDark, DarkFace, Pascal VOC, COCO).
Style & Architecture inspired by IGMamba-EdgeLCA-LLIE evaluation suite.
====================================================================================================
"""

import os
import sys
import glob
import json
import argparse
import platform
from pathlib import Path
from collections import defaultdict
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


# ==================================================================================================
# 1. Dataset Class Definitions & Metadata
# ==================================================================================================

EXDARK_CLASSES = [
    "Bicycle", "Boat", "Bottle", "Bus", "Car",
    "Cat", "Chair", "Cup", "Dog", "Motorbike",
    "People", "Table"
]

DARKFACE_CLASSES = [
    "Face"
]

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


# ==================================================================================================
# 2. Geometry & Metrics Helpers (Vectorized & Academic Standard)
# ==================================================================================================

def box_iou_numpy(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) of two sets of boxes [x1, y1, x2, y2].
    Args:
        boxes1: np.ndarray, shape (N, 4)
        boxes2: np.ndarray, shape (M, 4)
    Returns:
        iou: np.ndarray, shape (N, M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # [N, M, 2]
    lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = np.clip(rb - lt, a_min=0, a_max=None)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - inter
    iou = np.where(union > 0, inter / (union + 1e-8), 0.0)
    return iou


def compute_ap_coco_101(recalls, precisions):
    """
    Standard COCO 101-point interpolated Average Precision (AP).
    Used in MS COCO challenge and modern papers (CVPR, ICCV, ECCV).
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    recall_thresholds = np.linspace(0.0, 1.00, 101)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    q = mpre[inds]
    return float(np.mean(q))


def compute_ap_voc_continuous(recalls, precisions):
    """
    Continuous area under curve (AUC) AP calculation (PASCAL VOC 2012 / YOLO standard).
    """
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate area where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    return ap


# ==================================================================================================
# 3. Object Detection Evaluator Class
# ==================================================================================================

class ObjectDetectionEvaluator:
    """
    Evaluator to calculate:
      - mAP@0.50 (PASCAL VOC standard)
      - mAP@0.75 (Strict localization)
      - mAP@[0.50:0.95] (COCO standard primary metric)
      - Per-class APs, Precision, Recall, and F1 score
    """

    def __init__(self, class_names, iou_thresholds=None, method="coco"):
        self.class_names = list(class_names)
        self.class_to_id = {name: i for i, name in enumerate(self.class_names)}
        self.id_to_class = {i: name for i, name in enumerate(self.class_names)}
        self.method = method.lower()

        if iou_thresholds is None:
            # Default 10 IoU steps: 0.50 to 0.95 with step 0.05
            self.iou_thresholds = np.linspace(0.50, 0.95, 10).round(2)
        else:
            self.iou_thresholds = np.array(iou_thresholds, dtype=np.float32)

        # Storage structures
        # ground_truths[image_id][class_id] = list of [x1, y1, x2, y2]
        self.ground_truths = defaultdict(lambda: defaultdict(list))
        # predictions[class_id] = list of (image_id, score, [x1, y1, x2, y2])
        self.predictions = defaultdict(list)
        self.image_ids = set()

    def add_ground_truth(self, image_id, class_name_or_id, box):
        """Add a single GT box: [x1, y1, x2, y2]."""
        self.image_ids.add(image_id)
        cid = self._normalize_class_id(class_name_or_id)
        if cid is not None:
            self.ground_truths[image_id][cid].append(box)

    def add_prediction(self, image_id, class_name_or_id, score, box):
        """Add a single prediction: score, [x1, y1, x2, y2]."""
        self.image_ids.add(image_id)
        cid = self._normalize_class_id(class_name_or_id)
        if cid is not None:
            self.predictions[cid].append((image_id, float(score), box))

    def _normalize_class_id(self, class_name_or_id):
        if isinstance(class_name_or_id, int):
            if 0 <= class_name_or_id < len(self.class_names):
                return class_name_or_id
            return None
        name = str(class_name_or_id).strip()
        if name in self.class_to_id:
            return self.class_to_id[name]
        # Case-insensitive match
        for k, v in self.class_to_id.items():
            if k.lower() == name.lower():
                return v
        return None

    def evaluate(self, conf_threshold=0.001, score_threshold=None):
        """
        Run the academic evaluation protocol following LLEYOLO / COCO formulas:
          - Eq. (18): Precision = TP / (TP + FP)
          - Eq. (19): Recall    = TP / (TP + FN)
          - Eq. (20): mAP       = (1 / N) * sum_{n=1}^{N} AP(n)
          - Eq. (21): IoU       = Area(A n B) / Area(A u B)

        Returns:
            dict containing:
              - per_class: dict of {class_name: {targets, preds, tp, fp, fn, precision, recall, f1, ap50, ap75, ap50_95}}
              - total_tp, total_fp, total_fn
              - mAP50, mAP75, mAP50_95, mean_precision, mean_recall, mean_f1
        """
        results_per_class = {}
        all_aps = {round(iou, 2): [] for iou in self.iou_thresholds}

        # Store raw sorted scores, cumulative TP, and cumulative FP at IoU=0.50 per class
        class_pr_data_50 = {}

        for cid, class_name in enumerate(self.class_names):
            # Total positive samples (TP + FN) for this class
            n_gt = sum(len(self.ground_truths[img_id][cid]) for img_id in self.image_ids)

            # Filter predictions by minimum confidence threshold
            preds = [p for p in self.predictions[cid] if p[1] >= conf_threshold]
            n_preds = len(preds)

            # Sort predictions by confidence in descending order
            preds.sort(key=lambda x: x[1], reverse=True)
            scores_sorted = (
                np.array([p[1] for p in preds], dtype=np.float64)
                if n_preds > 0
                else np.array([], dtype=np.float64)
            )

            ap_per_iou = {}

            if n_gt == 0 or n_preds == 0:
                for iou_th in self.iou_thresholds:
                    ap_per_iou[round(iou_th, 2)] = 0.0
                # If n_gt == 0 but predictions exist, all predictions are FP
                cum_tp_50 = np.zeros(n_preds, dtype=np.float64)
                cum_fp_50 = np.arange(1, n_preds + 1, dtype=np.float64)
                class_pr_data_50[class_name] = (scores_sorted, cum_tp_50, cum_fp_50, n_gt)
            else:
                # Group GTs per image for fast lookup
                gt_boxes_per_img = {
                    img_id: np.array(self.ground_truths[img_id][cid], dtype=np.float32)
                    for img_id in self.image_ids
                    if len(self.ground_truths[img_id][cid]) > 0
                }

                # Group prediction indices and boxes by img_id (for images that have GT of this class)
                pred_indices_per_img = defaultdict(list)
                pred_boxes_per_img = defaultdict(list)
                for i, (img_id, _score, pred_box) in enumerate(preds):
                    if img_id in gt_boxes_per_img:
                        pred_indices_per_img[img_id].append(i)
                        pred_boxes_per_img[img_id].append(pred_box)

                # Precompute vectorized IoU matrix (K_preds_in_img x M_gts_in_img) once per image
                precomputed_img_matches = []
                for img_id, idxs in pred_indices_per_img.items():
                    gt_boxes = gt_boxes_per_img[img_id]
                    p_boxes = np.asarray(pred_boxes_per_img[img_id], dtype=np.float32)
                    ious_mat = box_iou_numpy(p_boxes, gt_boxes)  # (K, M)
                    max_idx = np.argmax(ious_mat, axis=1)        # (K,)
                    max_iou = ious_mat[np.arange(len(idxs)), max_idx]  # (K,)
                    precomputed_img_matches.append(
                        (
                            np.asarray(idxs, dtype=np.int64),
                            max_idx,
                            max_iou,
                            len(gt_boxes),
                        )
                    )

                # Evaluate for each IoU threshold (Eq. 21) using precomputed max IoUs
                for iou_th in self.iou_thresholds:
                    iou_key = round(iou_th, 2)
                    tp = np.zeros(n_preds, dtype=np.float32)
                    fp = np.ones(n_preds, dtype=np.float32)

                    for idxs, max_idx, max_iou, num_gt in precomputed_img_matches:
                        matched_gt = np.zeros(num_gt, dtype=bool)
                        for k_i in range(len(idxs)):
                            if max_iou[k_i] >= iou_th:
                                g_idx = max_idx[k_i]
                                if not matched_gt[g_idx]:
                                    matched_gt[g_idx] = True
                                    tp[idxs[k_i]] = 1.0
                                    fp[idxs[k_i]] = 0.0

                    # Compute cumulative TP and FP along confidence-sorted predictions
                    cum_tp = np.cumsum(tp, dtype=np.float64)
                    cum_fp = np.cumsum(fp, dtype=np.float64)
                    # Eq. (19): Recall = TP / (TP + FN) where TP + FN = n_gt
                    recalls = cum_tp / n_gt
                    # Eq. (18): Precision = TP / (TP + FP)
                    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

                    if self.method == "voc":
                        ap = compute_ap_voc_continuous(recalls, precisions)
                    else:
                        ap = compute_ap_coco_101(recalls, precisions)

                    ap_per_iou[iou_key] = ap

                    if abs(iou_th - 0.50) < 1e-4:
                        class_pr_data_50[class_name] = (scores_sorted, cum_tp, cum_fp, n_gt)

            ap50 = ap_per_iou.get(0.50, 0.0)
            ap75 = ap_per_iou.get(0.75, 0.0)
            ap50_95 = float(np.mean(list(ap_per_iou.values())))

            for iou_th in self.iou_thresholds:
                all_aps[round(iou_th, 2)].append(ap_per_iou[round(iou_th, 2)])

            results_per_class[class_name] = {
                "targets": n_gt,
                "preds": n_preds,
                "tp": 0,
                "fp": 0,
                "fn": n_gt,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "ap50": ap50,
                "ap75": ap75,
                "ap50_95": ap50_95,
            }

        # Determine confidence threshold for classifying positive predictions (TP, FP, FN)
        active_names = [cname for cname, res in results_per_class.items() if res["targets"] > 0]
        if score_threshold is not None:
            best_conf_threshold = float(score_threshold)
        elif conf_threshold >= 0.05:
            best_conf_threshold = float(conf_threshold)
        elif active_names:
            conf_grid = np.linspace(0.01, 0.99, 99)
            best_f1 = -1.0
            best_conf_threshold = 0.25
            for tau in conf_grid:
                p_list, r_list = [], []
                for cname in active_names:
                    scores_s, c_tp, c_fp, n_gt = class_pr_data_50[cname]
                    # Count predictions with score >= tau (scores_s is descending)
                    k = int(np.searchsorted(-scores_s, -tau, side="right"))
                    tp_c = float(c_tp[k - 1]) if k > 0 else 0.0
                    fp_c = float(c_fp[k - 1]) if k > 0 else 0.0
                    fn_c = float(n_gt) - tp_c
                    # Eq. (18): Precision = TP / (TP + FP)
                    p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                    # Eq. (19): Recall = TP / (TP + FN)
                    r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                    p_list.append(p_c)
                    r_list.append(r_c)
                mp = float(np.mean(p_list))
                mr = float(np.mean(r_list))
                f1 = 2.0 * mp * mr / (mp + mr + 1e-8)
                if f1 > best_f1:
                    best_f1 = f1
                    best_conf_threshold = float(tau)
        else:
            best_conf_threshold = 0.25

        # Compute exact TP, FP, FN, Precision (Eq. 18), and Recall (Eq. 19) for each class
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for cname in self.class_names:
            scores_s, c_tp, c_fp, n_gt = class_pr_data_50[cname]
            k = int(np.searchsorted(-scores_s, -best_conf_threshold, side="right"))
            tp_c = int(round(c_tp[k - 1])) if k > 0 else 0
            fp_c = int(round(c_fp[k - 1])) if k > 0 else 0
            fn_c = int(n_gt - tp_c)

            # Eq. (18): Precision = TP / (TP + FP)
            prec_val = float(tp_c / (tp_c + fp_c)) if (tp_c + fp_c) > 0 else 0.0
            # Eq. (19): Recall = TP / (TP + FN)
            rec_val = float(tp_c / (tp_c + fn_c)) if (tp_c + fn_c) > 0 else 0.0
            f1_val = float(2.0 * prec_val * rec_val / (prec_val + rec_val + 1e-8))

            results_per_class[cname]["tp"] = tp_c
            results_per_class[cname]["fp"] = fp_c
            results_per_class[cname]["fn"] = fn_c
            results_per_class[cname]["precision"] = prec_val
            results_per_class[cname]["recall"] = rec_val
            results_per_class[cname]["f1"] = f1_val

            total_tp += tp_c
            total_fp += fp_c
            total_fn += fn_c

        # Eq. (20): mAP = (1 / N) * sum_{n=1}^{N} AP(n) over active classes
        valid_classes = [res for res in results_per_class.values() if res["targets"] > 0]
        if not valid_classes:
            valid_classes = list(results_per_class.values())

        mAP50 = float(np.mean([res["ap50"] for res in valid_classes]))
        mAP75 = float(np.mean([res["ap75"] for res in valid_classes]))
        mAP50_95 = float(np.mean([res["ap50_95"] for res in valid_classes]))
        mean_p = float(np.mean([res["precision"] for res in valid_classes]))
        mean_r = float(np.mean([res["recall"] for res in valid_classes]))
        mean_f1 = float(2.0 * mean_p * mean_r / (mean_p + mean_r + 1e-8))

        return {
            "per_class": results_per_class,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "mAP50": mAP50,
            "mAP75": mAP75,
            "mAP50_95": mAP50_95,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
            "mean_f1": mean_f1,
            "best_conf_threshold": best_conf_threshold,
            "method": self.method,
        }


# ==================================================================================================
# 4. Universal Parsers for Ground Truth & Predictions
# ==================================================================================================

def parse_ground_truth_file(filepath, dataset_type="auto"):
    """
    Parses a single GT file and returns list of (class_label, [x1, y1, x2, y2]).
    Auto-detects DarkFace, ExDark, and standard VOC/YOLO absolute coordinates.
    """
    boxes = []
    if not os.path.exists(filepath):
        return boxes

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if not lines:
        return boxes

    # 1. DarkFace Format Detection (First line is total count N, subsequent lines: x1 y1 x2 y2)
    if dataset_type == "darkface" or (lines[0].isdigit() and len(lines) > 1 and len(lines[1].split()) == 4):
        for line in lines[1:]:
            parts = line.split()
            if len(parts) >= 4:
                x1, y1, x2, y2 = map(float, parts[:4])
                boxes.append(("Face", [x1, y1, x2, y2]))
        return boxes

    # 2. ExDark Original Format Detection (% bbGt version=3 \n ClassName left top w h ...)
    if dataset_type == "exdark" or lines[0].startswith("% bbGt"):
        for line in lines:
            if line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) >= 5:
                cname = parts[0]
                left, top, w, h = map(float, parts[1:5])
                boxes.append((cname, [left, top, left + w, top + h]))
        return boxes

    # 3. Generic Format (e.g., class_name x1 y1 x2 y2 or class_id x1 y1 x2 y2)
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            # class x1 y1 x2 y2
            c_label = parts[0]
            x1, y1, x2, y2 = map(float, parts[1:5])
            boxes.append((c_label, [x1, y1, x2, y2]))
        elif len(parts) == 4:
            # x1 y1 x2 y2 (default single class)
            x1, y1, x2, y2 = map(float, parts[:4])
            boxes.append((0, [x1, y1, x2, y2]))

    return boxes


def parse_prediction_file(filepath):
    """
    Parses a single prediction file and returns list of (class_label, score, [x1, y1, x2, y2]).
    Supported line variants:
      - <class> <conf> <x1> <y1> <x2> <y2> (Standard detection output)
      - <class> <x1> <y1> <x2> <y2> <conf>
    """
    preds = []
    if not os.path.exists(filepath):
        return preds

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) >= 6:
            # Format A: class_name/id conf x1 y1 x2 y2
            try:
                c_label = parts[0]
                conf = float(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:6])
                preds.append((c_label, conf, [x1, y1, x2, y2]))
            except ValueError:
                # Format B: class_name/id x1 y1 x2 y2 conf
                c_label = parts[0]
                x1, y1, x2, y2 = map(float, parts[1:5])
                conf = float(parts[5])
                preds.append((c_label, conf, [x1, y1, x2, y2]))
    return preds


# ==================================================================================================
# 5. Formatted Academic Output (Matching IGMamba-EdgeLCA-LLIE convention)
# ==================================================================================================

def print_academic_report(results, class_names):
    """Prints a neat, publication-ready table and summary stats."""
    per_class = results["per_class"]

    total_targets = sum(v["targets"] for v in per_class.values())
    total_tp = results.get("total_tp", sum(v.get("tp", 0) for v in per_class.values()))
    total_fp = results.get("total_fp", sum(v.get("fp", 0) for v in per_class.values()))
    total_fn = results.get("total_fn", sum(v.get("fn", 0) for v in per_class.values()))

    col_w = {
        "class": max(13, max(len(c) for c in class_names) + 2),
        "count": 7,
        "metric": 9,
    }

    border = "=" * 104
    div = "-" * 104

    print("\n" + border)
    print(f"  ACADEMIC EVALUATION REPORT (Protocol: {results['method'].upper()})")
    print(border)
    header = (
        f"{'Class':<{col_w['class']}} | "
        f"{'Targets':>{col_w['count']}} | "
        f"{'TP':>{col_w['count']}} | "
        f"{'FP':>{col_w['count']}} | "
        f"{'FN':>{col_w['count']}} | "
        f"{'Precision':>{col_w['metric']}} | "
        f"{'Recall':>{col_w['metric']}} | "
        f"{'mAP@0.50':>{col_w['metric']}} | "
        f"{'mAP@0.75':>{col_w['metric']}} | "
        f"{'mAP@50:95':>{col_w['metric']}}"
    )
    print(header)
    print(div)

    for cname in class_names:
        stat = per_class.get(cname, {})
        print(
            f"{cname:<{col_w['class']}} | "
            f"{stat.get('targets', 0):>{col_w['count']}} | "
            f"{stat.get('tp', 0):>{col_w['count']}} | "
            f"{stat.get('fp', 0):>{col_w['count']}} | "
            f"{stat.get('fn', 0):>{col_w['count']}} | "
            f"{stat.get('precision', 0.0):>{col_w['metric']}.4f} | "
            f"{stat.get('recall', 0.0):>{col_w['metric']}.4f} | "
            f"{stat.get('ap50', 0.0):>{col_w['metric']}.4f} | "
            f"{stat.get('ap75', 0.0):>{col_w['metric']}.4f} | "
            f"{stat.get('ap50_95', 0.0):>{col_w['metric']}.4f}"
        )

    print(div)
    # Summary line
    print(
        f"{'All (Mean)':<{col_w['class']}} | "
        f"{total_targets:>{col_w['count']}} | "
        f"{total_tp:>{col_w['count']}} | "
        f"{total_fp:>{col_w['count']}} | "
        f"{total_fn:>{col_w['count']}} | "
        f"{results['mean_precision']:>{col_w['metric']}.4f} | "
        f"{results['mean_recall']:>{col_w['metric']}.4f} | "
        f"{results['mAP50']:>{col_w['metric']}.4f} | "
        f"{results['mAP75']:>{col_w['metric']}.4f} | "
        f"{results['mAP50_95']:>{col_w['metric']}.4f}"
    )
    print(border)

    # Clean console summary matching LLEYOLO Equations (18)-(21)
    mean_p = results.get("mean_precision", 0.0)
    mean_r = results.get("mean_recall", 0.0)
    print(f"===> Precision (TP/(TP+FP)): {mean_p:.4f}  ({mean_p*100:.2f}%)")
    print(f"===> Recall    (TP/(TP+FN)): {mean_r:.4f}  ({mean_r*100:.2f}%)")
    print(f"===> mAP@0.50:               {results['mAP50']:.4f}  ({results['mAP50']*100:.2f}%)")
    print(f"===> mAP@0.75:               {results['mAP75']:.4f}  ({results['mAP75']*100:.2f}%)")
    print(f"===> mAP@0.50:0.95:          {results['mAP50_95']:.4f}  ({results['mAP50_95']*100:.2f}%)")
    print(border + "\n")


# ==================================================================================================
# 6. Main Execution Pipeline (`metrics` / `run_evaluation`)
# ==================================================================================================

_GT_MEMORY_CACHE = {}


def _clean_stem(filepath: str) -> str:
    stem = Path(filepath).stem
    lower_stem = stem.lower()
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if lower_stem.endswith(ext):
            return stem[: -len(ext)]
    return stem


def metrics(
    pred_dir,
    label_dir,
    class_names=EXDARK_CLASSES,
    dataset_type="auto",
    method="coco",
    conf_thres=0.001,
    in_memory_preds=None,
):
    """
    Loads predictions either directly from `in_memory_preds` (fast RAM path) or from `.txt` files
    in `pred_dir`, and Ground Truth `.txt` files from `label_dir`.
    Computes COCO/VOC mAP (`mAP@0.50`, `mAP@0.75`, `mAP@0.50:0.95`), `Precision@0.50`, `Recall@0.50`,
    and `F1-Score@0.50`, prints the academic report, and returns the results dictionary.
    """
    evaluator = ObjectDetectionEvaluator(class_names=class_names, method=method)

    pred_stems = set()
    if in_memory_preds is not None:
        for raw_stem, preds in in_memory_preds.items():
            stem = _clean_stem(raw_stem)
            pred_stems.add(stem)
            evaluator.image_ids.add(stem)
            for c_label, score, box in preds:
                evaluator.add_prediction(stem, c_label, score, box)
    else:
        # Collect predictions from disk .txt files
        pred_files = glob.glob(os.path.join(pred_dir, "**", "*.txt"), recursive=True)
        if not pred_files:
            pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))

        for p_path in tqdm(pred_files, desc="Parsing Preds"):
            stem = _clean_stem(p_path)
            pred_stems.add(stem)
            evaluator.image_ids.add(stem)
            preds = parse_prediction_file(p_path)
            for c_label, score, box in preds:
                evaluator.add_prediction(stem, c_label, score, box)

    # Load & cache Ground Truths from label_dir (both in-memory and on-disk .gt_cache.json)
    cache_key = (os.path.abspath(label_dir), str(dataset_type).lower())
    if cache_key not in _GT_MEMORY_CACHE:
        disk_cache_path = os.path.join(label_dir, f".gt_cache_{str(dataset_type).lower()}.json")
        parsed_gts = None
        if os.path.exists(disk_cache_path):
            try:
                with open(disk_cache_path, "r", encoding="utf-8") as cf:
                    parsed_gts = json.load(cf)
            except Exception:
                parsed_gts = None

        if parsed_gts is None:
            gt_files = glob.glob(os.path.join(label_dir, "**", "*.txt"), recursive=True)
            if not gt_files:
                gt_files = glob.glob(os.path.join(label_dir, "*.txt"))

            print(f"Loading {len(gt_files)} Ground Truth files from: {label_dir}")
            parsed_gts = {}
            for g_path in tqdm(gt_files, desc="Parsing GTs"):
                stem = _clean_stem(g_path)
                parsed_gts[stem] = parse_ground_truth_file(g_path, dataset_type=dataset_type)
            try:
                with open(disk_cache_path, "w", encoding="utf-8") as cf:
                    json.dump(parsed_gts, cf)
            except Exception:
                pass
        _GT_MEMORY_CACHE[cache_key] = parsed_gts
    else:
        parsed_gts = _GT_MEMORY_CACHE[cache_key]

    # If label_dir contains a superset of splits (e.g. DarkFace label/ has all 6000 images while pred_dir has 900 val images),
    # restrict evaluation to the evaluated split stems when pred_stems is a subset of parsed_gts.
    use_subset = (
        len(pred_stems) > 0
        and len(parsed_gts) > len(pred_stems)
        and pred_stems.issubset(parsed_gts.keys())
    )
    target_gt_stems = pred_stems if use_subset else parsed_gts.keys()

    for stem in target_gt_stems:
        evaluator.image_ids.add(stem)
        for c_label, box in parsed_gts.get(stem, []):
            c_idx = int(c_label) if str(c_label).isdigit() else c_label
            evaluator.add_ground_truth(stem, c_idx, box)

    # Run evaluation
    results = evaluator.evaluate(conf_threshold=conf_thres)
    print_academic_report(results, class_names)
    return results


# Backward-compatible alias
run_evaluation = metrics


if __name__ == "__main__":
    mea_parser = argparse.ArgumentParser(
        description="Object Detection Academic Evaluation Suite (mAP50, mAP75, mAP50:95, Precision, Recall)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset presets (matching IGMamba measure.py style)
    mea_parser.add_argument("--exdark", action="store_true", help="Evaluate ExDark dataset (12 classes)")
    mea_parser.add_argument("--darkface", action="store_true", help="Evaluate DarkFace dataset (Face)")
    mea_parser.add_argument("--voc", action="store_true", help="Evaluate Pascal VOC dataset (20 classes)")
    mea_parser.add_argument(
        "--split",
        type=str,
        choices=["val", "test", "train"],
        default="test",
        help="Dataset split to measure (default: test)",
    )

    # Custom paths
    mea_parser.add_argument("--pred_dir", type=str, default="", help="Directory of prediction .txt files")
    mea_parser.add_argument("--gt_dir", type=str, default="", help="Directory of ground truth .txt files")
    mea_parser.add_argument("--conf_thres", type=float, default=0.001, help="Confidence threshold for predictions")
    mea_parser.add_argument("--method", type=str, choices=["coco", "voc"], default="coco",
                            help="Interpolation method: 'coco' (101-point, default) or 'voc' (continuous AUC)")
    mea_parser.add_argument("--save_json", type=str, default="", help="Optional path to save results as JSON")

    args = mea_parser.parse_args()

    # Preset configuration
    dataset_type = "auto"
    if args.exdark:
        classes = EXDARK_CLASSES
        dataset_type = "exdark"
        if not args.gt_dir:
            args.gt_dir = f"./dataset/ExDark/labels_raw/{args.split}"
        if not args.pred_dir:
            cand_dirs = [
                f"./results/eval/exdark_{args.split}/preds",
                f"./results/exdark_{args.split}_preds",
                "./results/exdark_preds" if args.split == "val" else "./results/exdark_test_preds",
                f"./output/ExDark/{args.split}",
            ]
            args.pred_dir = next((d for d in cand_dirs if os.path.exists(d)), cand_dirs[0])
    elif args.darkface:
        classes = DARKFACE_CLASSES
        dataset_type = "darkface"
        if not args.gt_dir:
            args.gt_dir = "./dataset/DarkFace/label"
        if not args.pred_dir:
            args.pred_dir = f"./results/eval/darkface_{args.split}/preds"
    elif args.voc:
        classes = VOC_CLASSES
        dataset_type = "voc"
        if not args.gt_dir:
            args.gt_dir = f"./dataset/VOC/labels/{args.split}"
        if not args.pred_dir:
            args.pred_dir = f"./results/eval/voc_{args.split}/preds"
    else:
        classes = EXDARK_CLASSES
        dataset_type = "exdark"
        if not args.gt_dir:
            args.gt_dir = f"./dataset/ExDark/labels_raw/{args.split}"
        if not args.pred_dir:
            args.pred_dir = f"./results/eval/exdark_{args.split}/preds"

    if not os.path.exists(args.gt_dir):
        print(f"Warning: Ground truth directory not found at: '{args.gt_dir}'.")
        print("Please check the path or provide --gt_dir <path_to_labels>.")

    if not os.path.exists(args.pred_dir):
        print(f"Notice: Prediction directory '{args.pred_dir}' does not exist yet.")
        print("Run `python eval.py --weights <ckpt>` first to generate predictions, then run `python measure.py`.")
        sys.exit(0)

    results = metrics(
        pred_dir=args.pred_dir,
        label_dir=args.gt_dir,
        class_names=classes,
        dataset_type=dataset_type,
        method=args.method,
        conf_thres=args.conf_thres,
    )

    if args.save_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)), exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to: {args.save_json}")
