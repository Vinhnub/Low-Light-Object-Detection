"""
Script to split the ExDark dataset into Train, Val, and Test sets
based on the official benchmark partition from:
'Getting to Know Low-light Images with The Exclusively Dark Dataset' (CVIU 2019)
by Yuen Peng Loh & Chee Seng Chan.

Official Partition:
- Train: 3,000 images (250 images per class across 12 classes)
- Val:   1,800 images (150 images per class across 12 classes)
- Test:  2,563 images (remaining images across 12 classes)
Total:   7,363 images

Directory layout generated:
dataset/ExDark/
  ├── images/
  │   ├── train/ (3,000 files)
  │   ├── val/   (1,800 files)
  │   └── test/  (2,563 files)
  ├── labels/             (YOLO format: <class_id> <x_center> <y_center> <w> <h> normalized)
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels_raw/         (Original ExDark format: <class_name> <left> <top> <w> <h> ...)
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── imageclasslist.txt  (Official split metadata)
  ├── train.txt           (Image path list for training)
  ├── val.txt             (Image path list for validation)
  ├── test.txt            (Image path list for testing)
  └── exdark.yaml         (Dataset configuration for YOLO / Ultralytics)
"""

import os
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

CLASSES = [
    'Bicycle', 'Boat', 'Bottle', 'Bus', 'Car', 'Cat',
    'Chair', 'Cup', 'Dog', 'Motorbike', 'People', 'Table'
]

CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS = {i + 1: c for i, c in enumerate(CLASSES)}
SPLIT_MAP = {1: 'train', 2: 'val', 3: 'test'}

OFFICIAL_METADATA_URL = (
    'https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset'
    '/master/Groundtruth/imageclasslist.txt'
)


def ensure_metadata_file(metadata_path: str):
    """Ensure imageclasslist.txt exists, download if missing."""
    if not os.path.exists(metadata_path):
        print(f"Downloading official metadata to {metadata_path}...")
        urllib.request.urlretrieve(OFFICIAL_METADATA_URL, metadata_path)
        print("Download complete.")


def convert_raw_to_yolo(raw_lines, img_w, img_h):
    """Convert raw ExDark bounding boxes to YOLO format."""
    yolo_lines = []
    for line in raw_lines:
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        class_name = parts[0]
        if class_name not in CLASS_TO_ID:
            continue
        class_id = CLASS_TO_ID[class_name]

        try:
            x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue

        # Clamp coordinates to image bounds
        x_min = max(0.0, min(x, float(img_w)))
        y_min = max(0.0, min(y, float(img_h)))
        x_max = max(0.0, min(x + w, float(img_w)))
        y_max = max(0.0, min(y + h, float(img_h)))

        bw = x_max - x_min
        bh = y_max - y_min
        if bw <= 0 or bh <= 0:
            continue

        x_center = (x_min + bw / 2.0) / img_w
        y_center = (y_min + bh / 2.0) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        # Format normalized coordinates
        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
        )
    return yolo_lines


def process_single_item(item, exdark_dir):
    """Process a single image and its corresponding label files."""
    img_name, class_idx, split_idx = item
    class_name = ID_TO_CLASS[class_idx]
    split_name = SPLIT_MAP[split_idx]

    # Source paths
    src_img = os.path.join(exdark_dir, 'Image', class_name, img_name)
    src_lbl = os.path.join(exdark_dir, 'Label', class_name, img_name + '.txt')

    if not os.path.exists(src_img):
        return False, f"Missing image: {src_img}"
    if not os.path.exists(src_lbl):
        return False, f"Missing label: {src_lbl}"

    # Stem name without image extension
    stem = os.path.splitext(img_name)[0]

    # Destination paths
    dst_img = os.path.join(exdark_dir, 'images', split_name, img_name)
    dst_lbl_yolo = os.path.join(exdark_dir, 'labels', split_name, stem + '.txt')
    dst_lbl_raw = os.path.join(exdark_dir, 'labels_raw', split_name, stem + '.txt')

    # Copy image
    shutil.copyfile(src_img, dst_img)

    # Read raw label
    with open(src_lbl, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    # Save raw label in split
    with open(dst_lbl_raw, 'w', encoding='utf-8') as f:
        f.writelines(raw_lines)

    # Convert and save YOLO label
    with Image.open(src_img) as img:
        img_w, img_h = img.size

    yolo_lines = convert_raw_to_yolo(raw_lines, img_w, img_h)
    with open(dst_lbl_yolo, 'w', encoding='utf-8') as f:
        f.writelines(yolo_lines)

    return True, (split_name, class_name, f"images/{split_name}/{img_name}")


def split_dataset(exdark_dir: str):
    metadata_path = os.path.join(exdark_dir, 'imageclasslist.txt')
    ensure_metadata_file(metadata_path)

    # Create destination directories
    splits = ['train', 'val', 'test']
    for s in splits:
        os.makedirs(os.path.join(exdark_dir, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(exdark_dir, 'labels', s), exist_ok=True)
        os.makedirs(os.path.join(exdark_dir, 'labels_raw', s), exist_ok=True)

    # Parse metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    items = []
    for line in lines[1:]:  # skip header
        parts = line.strip().split()
        if len(parts) >= 5:
            items.append((parts[0], int(parts[1]), int(parts[4])))

    print(f"Total entries in metadata: {len(items)}")

    # Process files concurrently
    split_counts = {'train': 0, 'val': 0, 'test': 0}
    class_split_counts = {c: {'train': 0, 'val': 0, 'test': 0} for c in CLASSES}
    split_paths = {'train': [], 'val': [], 'test': []}

    print("Copying images and generating YOLO/raw labels with multithreading...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_item, item, exdark_dir) for item in items]
        for f in futures:
            success, result = f.result()
            if not success:
                print(f"[ERROR] {result}")
            else:
                s_name, c_name, rel_path = result
                split_counts[s_name] += 1
                class_split_counts[c_name][s_name] += 1
                split_paths[s_name].append(rel_path)

    # Write train.txt, val.txt, test.txt
    for s in splits:
        txt_path = os.path.join(exdark_dir, f"{s}.txt")
        # Sort paths for reproducibility
        sorted_paths = sorted(split_paths[s])
        with open(txt_path, 'w', encoding='utf-8') as f:
            for p in sorted_paths:
                f.write(p.replace('\\', '/') + '\n')
        print(f"Wrote {len(sorted_paths)} paths to {txt_path}")

    # Write exdark.yaml configuration for YOLO
    yaml_content = f"""# ExDark Dataset Configuration for YOLO
path: dataset/ExDark  # dataset root dir
train: images/train
val: images/val
test: images/test

# Classes (12 categories)
names:
  0: Bicycle
  1: Boat
  2: Bottle
  3: Bus
  4: Car
  5: Cat
  6: Chair
  7: Cup
  8: Dog
  9: Motorbike
  10: People
  11: Table
"""
    yaml_path = os.path.join(exdark_dir, 'exdark.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"Wrote YOLO config to {yaml_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SPLIT SUMMARY:")
    print("=" * 60)
    print(f"Train set: {split_counts['train']} images")
    print(f"Val set:   {split_counts['val']} images")
    print(f"Test set:  {split_counts['test']} images")
    print(f"Total:     {sum(split_counts.values())} images")
    print("-" * 60)
    print(f"{'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-" * 60)
    for c in CLASSES:
        tr = class_split_counts[c]['train']
        va = class_split_counts[c]['val']
        te = class_split_counts[c]['test']
        tot = tr + va + te
        print(f"{c:<12} {tr:>8} {va:>8} {te:>8} {tot:>8}")
    print("=" * 60)


if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.abspath(__file__))
    exdark_directory = os.path.join(root_dir, 'dataset', 'ExDark')
    split_dataset(exdark_directory)
