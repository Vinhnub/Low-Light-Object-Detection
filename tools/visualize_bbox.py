"""
tools/visualize_bbox.py

Script to visualize bounding boxes and class labels from a dataset (such as ExDark or DarkFace).
Reads images and corresponding label text files, draws bounding boxes, class names,
and saves the annotated images into a new directory within the dataset.

Supported label formats:
1. YOLO format: <class_id> <x_center> <y_center> <width> <height> (normalized 0..1)
2. DarkFace format: line 1 = count N, lines 2.. = <xmin> <ymin> <xmax> <ymax> (pixels)
3. ExDark raw format: % bbGt version=3 \n <class_name> <left> <top> <width> <height> ... (pixels)
4. Auto-detection of label format.

Usage Examples:
  # Visualize 30 images from ExDark train set (YOLO format)
  python tools/visualize_bbox.py --dataset dataset/ExDark --split train --num 30

  # Visualize 30 images from DarkFace
  python tools/visualize_bbox.py --dataset dataset/DarkFace --num 30

  # Visualize all images with custom paths
  python tools/visualize_bbox.py --images dataset/ExDark/images/val --labels dataset/ExDark/labels/val --num -1

  # Random shuffle sampling
  python tools/visualize_bbox.py --dataset dataset/ExDark --split val --num 20 --shuffle
"""

import os
import sys
import glob
import random
import argparse
from typing import List, Tuple, Dict, Optional
import cv2

# Default 12 classes for ExDark
EXDARK_CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

# Distinct, vivid BGR color palette designed for high contrast on low-light / dark images
COLOR_PALETTE = [
    (0, 255, 255),    # 0: Bright Yellow
    (255, 128, 0),    # 1: Sky Blue
    (50, 255, 50),    # 2: Neon Green
    (0, 165, 255),    # 3: Bright Orange
    (255, 0, 255),    # 4: Magenta / Fuchsia
    (255, 255, 0),    # 5: Cyan
    (0, 75, 255),     # 6: Bright Red
    (180, 105, 255),  # 7: Hot Pink
    (0, 215, 255),    # 8: Gold
    (226, 43, 138),   # 9: Violet / Purple
    (128, 255, 0),    # 10: Spring Green
    (0, 128, 255),    # 11: Amber / Coral
    (255, 191, 0),    # 12: Deep Sky Blue
    (203, 192, 255),  # 13: Light Pink
    (144, 238, 144),  # 14: Light Green
    (230, 216, 173),  # 15: Light Slate Blue
]

SUPPORTED_IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP'}


def get_color(class_id: int) -> Tuple[int, int, int]:
    """Return a consistent distinct BGR color for a given class ID."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


def load_yaml_classes(yaml_path: str) -> List[str]:
    """Parse class names from a YOLO yaml configuration file."""
    classes = []
    if not os.path.exists(yaml_path):
        return classes
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        in_names = False
        for line in lines:
            line_str = line.strip()
            if line_str.startswith('names:'):
                in_names = True
                continue
            if in_names:
                if line_str and not line_str.startswith('#'):
                    # format: 0: Bicycle or - Bicycle
                    if ':' in line_str:
                        parts = line_str.split(':', 1)
                        if parts[0].strip().isdigit():
                            classes.append(parts[1].strip().strip("'\""))
                    elif line_str.startswith('-'):
                        classes.append(line_str.lstrip('-').strip().strip("'\""))
                elif line and not line[0].isspace():
                    break
    except Exception as e:
        print(f"[WARN] Error parsing yaml {yaml_path}: {e}")
    return classes


def detect_label_format(lines: List[str]) -> str:
    """Automatically identify the format of label file lines."""
    clean_lines = [l.strip() for l in lines if l.strip()]
    if not clean_lines:
        return 'empty'

    # Check for ExDark raw header
    if clean_lines[0].startswith('% bbGt') or clean_lines[0].startswith('%'):
        return 'exdark_raw'

    # Check if first line is a face count (DarkFace format)
    if clean_lines[0].isdigit():
        if len(clean_lines) == 1:
            return 'darkface'
        parts = clean_lines[1].split()
        if len(parts) == 4:
            return 'darkface'

    # Check first line tokens
    first_tokens = clean_lines[0].split()
    if len(first_tokens) >= 5:
        # Check if first token is an ExDark class name
        if first_tokens[0] in EXDARK_CLASSES:
            return 'exdark_raw'
        # Check if YOLO normalized coordinates (float in [0, 1])
        try:
            coords = [float(x) for x in first_tokens[1:5]]
            if all(0.0 <= c <= 1.05 for c in coords):
                return 'yolo'
        except ValueError:
            pass

    # Check if 4 integer/float coordinates (xmin, ymin, xmax, ymax)
    if len(first_tokens) == 4:
        return 'darkface'

    return 'yolo'


def parse_labels(
    label_path: str,
    img_w: int,
    img_h: int,
    classes: List[str],
    fmt: str = 'auto'
) -> List[Tuple[int, str, Tuple[int, int, int, int]]]:
    """
    Parse a label file and return list of (class_id, class_name, (xmin, ymin, xmax, ymax)).
    """
    if not os.path.exists(label_path):
        return []

    with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if not lines:
        return []

    active_fmt = fmt
    if active_fmt == 'auto':
        active_fmt = detect_label_format(lines)

    boxes = []

    if active_fmt == 'yolo':
        for line in lines:
            if line.startswith('#') or line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc = float(parts[1]) * img_w
                yc = float(parts[2]) * img_h
                bw = float(parts[3]) * img_w
                bh = float(parts[4]) * img_h
                xmin = int(round(xc - bw / 2.0))
                ymin = int(round(yc - bh / 2.0))
                xmax = int(round(xc + bw / 2.0))
                ymax = int(round(yc + bh / 2.0))
                cname = classes[class_id] if class_id < len(classes) else f"class_{class_id}"
                boxes.append((class_id, cname, (xmin, ymin, xmax, ymax)))
            except (ValueError, IndexError):
                continue

    elif active_fmt == 'darkface':
        # Line 0 is often face count N; subsequent lines are xmin ymin xmax ymax
        start_idx = 1 if lines[0].isdigit() else 0
        for line in lines[start_idx:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                xmin = int(round(float(parts[0])))
                ymin = int(round(float(parts[1])))
                xmax = int(round(float(parts[2])))
                ymax = int(round(float(parts[3])))
                cname = classes[0] if classes else 'face'
                boxes.append((0, cname, (xmin, ymin, xmax, ymax)))
            except (ValueError, IndexError):
                continue

    elif active_fmt == 'exdark_raw':
        # Line 0 may be % bbGt; other lines are <ClassName> <left> <top> <width> <height> ...
        class_to_id = {c: i for i, c in enumerate(classes)}
        for line in lines:
            if line.startswith('%'):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cname = parts[0]
            try:
                left = float(parts[1])
                top = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                xmin = int(round(left))
                ymin = int(round(top))
                xmax = int(round(left + w))
                ymax = int(round(top + h))
                cid = class_to_id.get(cname, 0)
                boxes.append((cid, cname, (xmin, ymin, xmax, ymax)))
            except (ValueError, IndexError):
                continue

    return boxes


def draw_bounding_boxes(
    image,
    boxes: List[Tuple[int, str, Tuple[int, int, int, int]]],
    thickness: int = 2
):
    """Draw bounding boxes with label banners onto the image."""
    img_h, img_w = image.shape[:2]

    # Adaptive font scale based on image dimensions
    font_scale = max(0.4, min(img_w, img_h) / 1000.0)
    font_thickness = max(1, int(round(font_scale * 1.5)))

    for class_id, class_name, (xmin, ymin, xmax, ymax) in boxes:
        # Clamp coordinates to image borders for safe drawing
        xmin_c = max(0, min(xmin, img_w - 1))
        ymin_c = max(0, min(ymin, img_h - 1))
        xmax_c = max(0, min(xmax, img_w - 1))
        ymax_c = max(0, min(ymax, img_h - 1))

        if xmax_c <= xmin_c or ymax_c <= ymin_c:
            continue

        color = get_color(class_id)

        # 1. Draw bounding box rectangle
        cv2.rectangle(image, (xmin_c, ymin_c), (xmax_c, ymax_c), color, thickness)

        # 2. Prepare text label
        label_text = f"{class_name}"
        (tw, th), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        # 3. Position the text banner (above box if space permits, else inside)
        if ymin_c - th - 6 > 0:
            box_text_top = ymin_c - th - 6
            box_text_bottom = ymin_c
        else:
            box_text_top = ymin_c
            box_text_bottom = ymin_c + th + 6

        box_text_right = min(img_w - 1, xmin_c + tw + 6)

        # 4. Draw filled background for label tag
        cv2.rectangle(
            image,
            (xmin_c, box_text_top),
            (box_text_right, box_text_bottom),
            color,
            -1  # Filled
        )

        # 5. Determine high-contrast text color (dark text on bright tag, white text on dark tag)
        tag_brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
        text_color = (0, 0, 0) if tag_brightness > 140 else (255, 255, 255)

        # 6. Put text
        text_origin = (xmin_c + 3, box_text_bottom - baseline - 2)
        cv2.putText(
            image,
            label_text,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            font_thickness,
            lineType=cv2.LINE_AA
        )

    return image


def find_dataset_paths(
    dataset_dir: str,
    split: str = 'train'
) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
    """
    Inspect a dataset directory and automatically discover image folder, label folder,
    recommended output folder, and class names list.
    """
    classes = []
    # Check for yaml config
    yaml_candidates = glob.glob(os.path.join(dataset_dir, '*.yaml')) + glob.glob(os.path.join(dataset_dir, '*.yml'))
    if yaml_candidates:
        classes = load_yaml_classes(yaml_candidates[0])

    # Check ExDark split style: images/train, labels/train
    cand_img = os.path.join(dataset_dir, 'images', split)
    cand_lbl = os.path.join(dataset_dir, 'labels', split)
    if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
        out_dir = os.path.join(dataset_dir, f'visualized_{split}')
        if not classes:
            classes = EXDARK_CLASSES
        return cand_img, cand_lbl, out_dir, classes

    # Check images / labels direct
    cand_img = os.path.join(dataset_dir, 'images')
    cand_lbl = os.path.join(dataset_dir, 'labels')
    if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
        out_dir = os.path.join(dataset_dir, 'visualized')
        return cand_img, cand_lbl, out_dir, classes

    # Check DarkFace style: image / label
    cand_img = os.path.join(dataset_dir, 'image')
    cand_lbl = os.path.join(dataset_dir, 'label')
    if os.path.isdir(cand_img) and os.path.isdir(cand_lbl):
        out_dir = os.path.join(dataset_dir, 'visualized')
        if not classes:
            classes = ['face']
        return cand_img, cand_lbl, out_dir, classes

    return None, None, None, classes


def match_image_and_label_files(img_dir: str, lbl_dir: str) -> List[Tuple[str, Optional[str]]]:
    """
    Pair each image file with its corresponding label file.
    Supports both <stem>.txt and <stem>.<ext>.txt
    """
    all_files = os.listdir(img_dir)
    image_files = [f for f in all_files if os.path.splitext(f)[1] in SUPPORTED_IMG_EXTS]
    image_files.sort()

    pairs = []
    for img_name in image_files:
        stem, ext = os.path.splitext(img_name)
        img_path = os.path.join(img_dir, img_name)

        # Check candidate label file names
        candidates = [
            os.path.join(lbl_dir, stem + '.txt'),
            os.path.join(lbl_dir, img_name + '.txt'),
        ]
        lbl_path = None
        for cand in candidates:
            if os.path.exists(cand):
                lbl_path = cand
                break

        pairs.append((img_path, lbl_path))

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Visualize bounding boxes and labels on images from a dataset."
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        default=None,
        help="Path to dataset root folder (e.g. 'dataset/ExDark' or 'dataset/DarkFace')."
    )
    parser.add_argument(
        '--images', '-i',
        type=str,
        default=None,
        help="Path to images directory (overrides auto-detection from --dataset)."
    )
    parser.add_argument(
        '--labels', '-l',
        type=str,
        default=None,
        help="Path to labels directory (overrides auto-detection from --dataset)."
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Path to output directory where visualized images are saved."
    )
    parser.add_argument(
        '--split', '-s',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help="Dataset split to visualize if partitioned into train/val/test (default: 'train')."
    )
    parser.add_argument(
        '--num', '-n',
        type=int,
        default=30,
        help="Number of images to visualize. Use -1 for all images (default: 30)."
    )
    parser.add_argument(
        '--shuffle',
        action='store_true',
        help="Randomly shuffle and sample images instead of taking first N."
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        default='auto',
        choices=['auto', 'yolo', 'darkface', 'exdark_raw'],
        help="Label format. 'auto' detects automatically (default: 'auto')."
    )
    parser.add_argument(
        '--classes', '-c',
        type=str,
        default=None,
        help="Comma-separated class names or path to .yaml file (e.g. 'exdark.yaml')."
    )
    parser.add_argument(
        '--thickness', '-t',
        type=int,
        default=2,
        help="Bounding box outline line thickness (default: 2)."
    )

    args = parser.parse_args()

    # 1. Determine image and label directories
    img_dir = args.images
    lbl_dir = args.labels
    out_dir = args.output
    classes = []

    if args.classes:
        if os.path.isfile(args.classes):
            classes = load_yaml_classes(args.classes)
        else:
            classes = [c.strip() for c in args.classes.split(',') if c.strip()]

    if args.dataset:
        d_img, d_lbl, d_out, d_classes = find_dataset_paths(args.dataset, split=args.split)
        if not img_dir:
            img_dir = d_img
        if not lbl_dir:
            lbl_dir = d_lbl
        if not out_dir:
            out_dir = d_out
        if not classes and d_classes:
            classes = d_classes

    # Fallback default if nothing provided
    if not img_dir or not lbl_dir:
        # Default to dataset/ExDark images/train and labels/train if present
        default_exdark = os.path.join('dataset', 'ExDark')
        if os.path.isdir(default_exdark):
            d_img, d_lbl, d_out, d_classes = find_dataset_paths(default_exdark, split=args.split)
            img_dir = img_dir or d_img
            lbl_dir = lbl_dir or d_lbl
            out_dir = out_dir or d_out
            if not classes:
                classes = d_classes

    if not img_dir or not os.path.isdir(img_dir):
        print(f"[ERROR] Image directory not found: '{img_dir}'. Please specify --images or --dataset.")
        sys.exit(1)

    if not lbl_dir or not os.path.isdir(lbl_dir):
        print(f"[ERROR] Label directory not found: '{lbl_dir}'. Please specify --labels or --dataset.")
        sys.exit(1)

    if not out_dir:
        parent_dir = os.path.dirname(os.path.abspath(img_dir))
        out_dir = os.path.join(parent_dir, 'visualized')

    if not classes:
        classes = EXDARK_CLASSES

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 65)
    print("DATASET VISUALIZATION TOOL")
    print("=" * 65)
    print(f"Images directory : {img_dir}")
    print(f"Labels directory : {lbl_dir}")
    print(f"Output directory : {out_dir}")
    print(f"Label format     : {args.format}")
    print(f"Known classes ({len(classes)}) : {classes[:5]}{'...' if len(classes) > 5 else ''}")
    print("-" * 65)

    # 2. Pair images and labels
    pairs = match_image_and_label_files(img_dir, lbl_dir)
    print(f"Found {len(pairs)} images in '{img_dir}'.")

    if not pairs:
        print("[WARN] No supported image files found!")
        sys.exit(0)

    # 3. Sampling
    if args.shuffle:
        random.seed(42)
        random.shuffle(pairs)

    if args.num > 0 and args.num < len(pairs):
        selected_pairs = pairs[:args.num]
    else:
        selected_pairs = pairs

    print(f"Processing and visualizing {len(selected_pairs)} images...")

    # 4. Draw and Save
    processed_count = 0
    total_boxes_drawn = 0

    for idx, (img_path, lbl_path) in enumerate(selected_pairs, 1):
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to load image: {img_path}")
            continue

        img_h, img_w = image.shape[:2]

        boxes = []
        if lbl_path and os.path.exists(lbl_path):
            boxes = parse_labels(
                label_path=lbl_path,
                img_w=img_w,
                img_h=img_h,
                classes=classes,
                fmt=args.format
            )

        # Draw bounding boxes onto image
        drawn_image = draw_bounding_boxes(image, boxes, thickness=args.thickness)

        # Save to output folder
        base_name = os.path.basename(img_path)
        save_path = os.path.join(out_dir, base_name)
        cv2.imwrite(save_path, drawn_image)

        processed_count += 1
        total_boxes_drawn += len(boxes)

        if idx % 10 == 0 or idx == len(selected_pairs):
            print(f"  [{idx:4d}/{len(selected_pairs):4d}] Saved: {base_name} ({len(boxes)} boxes)")

    print("-" * 65)
    print(f"[SUCCESS] Visualized {processed_count} images with {total_boxes_drawn} bounding boxes.")
    print(f"[SUCCESS] All output images saved to:\n          file:///{os.path.abspath(out_dir).replace(os.sep, '/')}")
    print("=" * 65)


if __name__ == '__main__':
    main()
