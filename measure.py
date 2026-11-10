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

    def evaluate(self, conf_threshold=0.001):
        """
        Run the academic evaluation protocol.
        Returns:
            dict containing:
              - per_class: dict of {class_name: {ap50, ap75, ap50_95, precision, recall, targets, preds}}
              - mAP50, mAP75, mAP50_95, mean_precision, mean_recall
        """
        results_per_class = {}
        all_aps = {round(iou, 2): [] for iou in self.iou_thresholds}

        for cid, class_name in enumerate(self.class_names):
            # Count total ground truth boxes for this class
            n_gt = sum(len(self.ground_truths[img_id][cid]) for img_id in self.image_ids)

            # Filter predictions by confidence
            preds = [p for p in self.predictions[cid] if p[1] >= conf_threshold]
            n_preds = len(preds)

            # Sort predictions by confidence in descending order
            preds.sort(key=lambda x: x[1], reverse=True)

            ap_per_iou = {}
            prec_at_50 = 0.0
            rec_at_50 = 0.0

            if n_gt == 0:
                for iou_th in self.iou_thresholds:
                    ap_per_iou[round(iou_th, 2)] = 0.0
            elif n_preds == 0:
                for iou_th in self.iou_thresholds:
                    ap_per_iou[round(iou_th, 2)] = 0.0
            else:
                # Group GTs per image for fast lookup
                gt_boxes_per_img = {
                    img_id: np.array(self.ground_truths[img_id][cid], dtype=np.float32)
                    for img_id in self.image_ids
                    if len(self.ground_truths[img_id][cid]) > 0
                }

                # Evaluate for each IoU threshold
                for iou_th in self.iou_thresholds:
                    iou_key = round(iou_th, 2)
                    tp = np.zeros(n_preds, dtype=np.float32)
                    fp = np.zeros(n_preds, dtype=np.float32)

                    # Track matched GT flags per image
                    matched_gt = {
                        img_id: np.zeros(len(boxes), dtype=bool)
                        for img_id, boxes in gt_boxes_per_img.items()
                    }

                    for i, (img_id, score, pred_box) in enumerate(preds):
                        if img_id not in gt_boxes_per_img:
                            fp[i] = 1.0
                            continue

                        gt_boxes = gt_boxes_per_img[img_id]
                        pred_box_arr = np.array([pred_box], dtype=np.float32)

                        ious = box_iou_numpy(pred_box_arr, gt_boxes)[0]
                        max_idx = np.argmax(ious) if len(ious) > 0 else -1
                        max_iou = ious[max_idx] if max_idx >= 0 else 0.0

                        if max_iou >= iou_th:
                            if not matched_gt[img_id][max_idx]:
                                tp[i] = 1.0
                                matched_gt[img_id][max_idx] = True
                            else:
                                fp[i] = 1.0  # Duplicate detection on same GT
                        else:
                            fp[i] = 1.0

                    # Compute precision-recall curve
                    cum_tp = np.cumsum(tp)
                    cum_fp = np.cumsum(fp)
                    recalls = cum_tp / n_gt
                    precisions = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

                    if self.method == "voc":
                        ap = compute_ap_voc_continuous(recalls, precisions)
                    else:
                        ap = compute_ap_coco_101(recalls, precisions)

                    ap_per_iou[iou_key] = ap

                    if abs(iou_th - 0.50) < 1e-4:
                        prec_at_50 = float(precisions[-1]) if len(precisions) > 0 else 0.0
                        rec_at_50 = float(recalls[-1]) if len(recalls) > 0 else 0.0

            # Store per-class results
            ap50 = ap_per_iou.get(0.50, 0.0)
            ap75 = ap_per_iou.get(0.75, 0.0)
            ap50_95 = float(np.mean(list(ap_per_iou.values())))

            for iou_th in self.iou_thresholds:
                all_aps[round(iou_th, 2)].append(ap_per_iou[round(iou_th, 2)])

            results_per_class[class_name] = {
                "targets": n_gt,
                "preds": n_preds,
                "precision": prec_at_50,
                "recall": rec_at_50,
                "f1": (2 * prec_at_50 * rec_at_50 / (prec_at_50 + rec_at_50 + 1e-8)),
                "ap50": ap50,
                "ap75": ap75,
                "ap50_95": ap50_95,
            }

        # Overall summary
        mAP50 = float(np.mean([res["ap50"] for res in results_per_class.values()]))
        mAP75 = float(np.mean([res["ap75"] for res in results_per_class.values()]))
        mAP50_95 = float(np.mean([res["ap50_95"] for res in results_per_class.values()]))
        mean_p = float(np.mean([res["precision"] for res in results_per_class.values()]))
        mean_r = float(np.mean([res["recall"] for res in results_per_class.values()]))

        return {
            "per_class": results_per_class,
            "mAP50": mAP50,
            "mAP75": mAP75,
            "mAP50_95": mAP50_95,
            "mean_precision": mean_p,
            "mean_recall": mean_r,
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
    total_preds = sum(v["preds"] for v in per_class.values())

    col_w = {
        "class": max(15, max(len(c) for c in class_names) + 2),
        "count": 9,
        "metric": 10,
    }

    border = "=" * 88
    div = "-" * 88

    print("\n" + border)
    print(f"  ACADEMIC EVALUATION REPORT (Protocol: {results['method'].upper()})")
    print(border)
    header = (
        f"{'Class':<{col_w['class']}} | "
        f"{'Targets':>{col_w['count']}} | "
        f"{'Preds':>{col_w['count']}} | "
        f"{'P@0.5':>{col_w['metric']}} | "
        f"{'R@0.5':>{col_w['metric']}} | "
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
            f"{stat.get('preds', 0):>{col_w['count']}} | "
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
        f"{total_preds:>{col_w['count']}} | "
        f"{results['mean_precision']:>{col_w['metric']}.4f} | "
        f"{results['mean_recall']:>{col_w['metric']}.4f} | "
        f"{results['mAP50']:>{col_w['metric']}.4f} | "
        f"{results['mAP75']:>{col_w['metric']}.4f} | "
        f"{results['mAP50_95']:>{col_w['metric']}.4f}"
    )
    print(border)

    # Clean IGMamba-style console summary
    print(f"===> Metric: COCO Standard mAP (10 IoU steps: 0.50:0.05:0.95)")
    print(f"===> mAP@0.50:       {results['mAP50']:.4f}  ({results['mAP50']*100:.2f}%)")
    print(f"===> mAP@0.75:       {results['mAP75']:.4f}  ({results['mAP75']*100:.2f}%)")
    print(f"===> mAP@0.50:0.95:  {results['mAP50_95']:.4f}  ({results['mAP50_95']*100:.2f}%)")
    print(border + "\n")


# ==================================================================================================
# 6. Main Execution Pipeline
# ==================================================================================================

def run_evaluation(pred_dir, gt_dir, class_names, dataset_type="auto", method="coco", conf_thres=0.001):
    """Loads all prediction & GT files, executes evaluation, and returns results dict."""
    evaluator = ObjectDetectionEvaluator(class_names=class_names, method=method)

    # Collect ground truths
    gt_files = glob.glob(os.path.join(gt_dir, "**", "*.txt"), recursive=True)
    if not gt_files:
        gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))

    print(f"Loading {len(gt_files)} Ground Truth files from: {gt_dir}")
    for g_path in tqdm(gt_files, desc="Parsing GTs"):
        stem = Path(g_path).stem
        boxes = parse_ground_truth_file(g_path, dataset_type=dataset_type)
        for c_label, box in boxes:
            evaluator.add_ground_truth(stem, c_label, box)

    # Collect predictions
    pred_files = glob.glob(os.path.join(pred_dir, "**", "*.txt"), recursive=True)
    if not pred_files:
        pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))

    print(f"Loading {len(pred_files)} Prediction files from: {pred_dir}")
    for p_path in tqdm(pred_files, desc="Parsing Preds"):
        stem = Path(p_path).stem
        preds = parse_prediction_file(p_path)
        for c_label, score, box in preds:
            evaluator.add_prediction(stem, c_label, score, box)

    # Run evaluation
    print(f"\nCalculating academic metrics with protocol [{method.upper()}]...")
    results = evaluator.evaluate(conf_threshold=conf_thres)
    print_academic_report(results, class_names)
    return results


if __name__ == "__main__":
    mea_parser = argparse.ArgumentParser(
        description="Object Detection Academic Evaluation Suite (mAP50, mAP75, mAP50:95)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset presets (matching IGMamba measure.py style: --lol, --lol_v2_real, etc.)
    mea_parser.add_argument("--exdark", action="store_true", help="Evaluate ExDark dataset (12 classes)")
    mea_parser.add_argument("--darkface", action="store_true", help="Evaluate DarkFace dataset (Face)")
    mea_parser.add_argument("--voc", action="store_true", help="Evaluate Pascal VOC dataset (20 classes)")

    # Custom paths
    mea_parser.add_argument("--pred_dir", type=str, default="./output/predictions", help="Directory of prediction .txt files")
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
            args.gt_dir = "./dataset/ExDark/labels_raw/val"
    elif args.darkface:
        classes = DARKFACE_CLASSES
        dataset_type = "darkface"
        if not args.gt_dir:
            args.gt_dir = "./dataset/DarkFace/label"
    elif args.voc:
        classes = VOC_CLASSES
        dataset_type = "voc"
    else:
        # Default fallback to ExDark if no dataset flag specified
        classes = EXDARK_CLASSES
        if not args.gt_dir:
            args.gt_dir = "./dataset/ExDark/labels_raw/val"

    if not os.path.exists(args.gt_dir):
        print(f"Warning: Ground truth directory not found at: '{args.gt_dir}'.")
        print("Please check the path or provide --gt_dir <path_to_labels>.")

    if not os.path.exists(args.pred_dir):
        print(f"Notice: Prediction directory '{args.pred_dir}' does not exist yet.")
        print("Run your detection model inference to generate predictions, then re-run measure.py.")
        sys.exit(0)

    results = run_evaluation(
        pred_dir=args.pred_dir,
        gt_dir=args.gt_dir,
        class_names=classes,
        dataset_type=dataset_type,
        method=args.method,
        conf_thres=args.conf_thres,
    )

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to: {args.save_json}")
