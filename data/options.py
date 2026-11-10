import argparse


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def option():
    # Training settings for Low-Light Object Detection (YOLOv5)
    parser = argparse.ArgumentParser(description="Low-Light Object Detection (YOLOv5)")

    # 1. Model & Training parameters
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5n",
        choices=["yolov5n", "yolov5s", "yolov5m", "yolov5l", "yolov5x"],
        help="YOLOv5 model variant (default: %(default)s)",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Optional path to custom model YAML file (e.g. net/Yolov5/configs/yolov5s.yaml)",
    )
    parser.add_argument(
        "--modular",
        type=_str2bool,
        default=False,
        help="Use explicit modular YOLOv5Backbone + YOLOv5Neck + Detect architecture",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="",
        help="Path to pretrained weights (.pt / .pth) to load before training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path to checkpoint (.pt / .pth) to resume training from",
    )

    parser.add_argument(
        "--batchSize",
        "--batch_size",
        type=int,
        default=16,
        help="training batch size",
    )
    parser.add_argument(
        "--imgSize",
        "--img_size",
        "--cropSize",
        type=int,
        default=640,
        help="input image resolution (default: 640)",
    )
    parser.add_argument(
        "--nEpochs",
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train for end",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="number of epochs to start, >0 is retrained a pre-trained pth",
    )
    parser.add_argument(
        "--snapshots",
        type=int,
        default=10,
        help="Snapshots interval (epochs) for saving checkpoints and running validation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate (1e-2 for SGD, 1e-3 for Adam/AdamW)",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final OneCycleLR learning rate fraction (lr * lrf)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.937,
        help="SGD momentum / Adam beta1",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=5e-4,
        help="Optimizer weight decay",
    )
    parser.add_argument("--gpu_mode", type=_str2bool, default=True)
    parser.add_argument("--shuffle", type=_str2bool, default=True)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="number of worker threads for dataloader to use",
    )
    parser.add_argument(
        "--amp",
        type=_str2bool,
        default=True,
        help="Use Automatic Mixed Precision (AMP) training",
    )
    parser.add_argument(
        "--ema",
        type=_str2bool,
        default=True,
        help="Use Model Exponential Moving Average (EMA)",
    )

    # 2. Scheduler & Warmup settings
    parser.add_argument("--cos_restart_cyclic", type=_str2bool, default=False)
    parser.add_argument("--cos_restart", type=_str2bool, default=True)
    parser.add_argument("--warmup_epochs", type=int, default=0, help="warmup_epochs")
    parser.add_argument(
        "--start_warmup",
        type=_str2bool,
        default=True,
        help="turn False to train without warmup",
    )

    # 3. Dataset paths & Selection
    parser.add_argument(
        "--dataset",
        type=str,
        default="exdark",
        choices=["exdark", "darkface", "voc", "custom"],
        help="Select the dataset to train/evaluate on (default: %(default)s)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=12,
        help="Number of object classes (12 for ExDark, 1 for DarkFace, 20 for VOC)",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default="./dataset/ExDark/exdark.yaml",
        help="Dataset YAML file path",
    )

    # Train datasets
    parser.add_argument(
        "--data_train_exdark",
        type=str,
        default="./dataset/ExDark/images/train",
        help="ExDark training images directory or train.txt path",
    )
    parser.add_argument(
        "--data_train_darkface",
        type=str,
        default="./dataset/DarkFace",
        help="DarkFace dataset root or train images directory",
    )
    parser.add_argument(
        "--data_train_voc",
        type=str,
        default="./dataset/VOC/images/train",
        help="VOC training images directory",
    )
    parser.add_argument(
        "--data_train_custom",
        type=str,
        default="./dataset/custom/images/train",
        help="Custom training images directory",
    )

    # Validation datasets
    parser.add_argument(
        "--data_val_exdark",
        type=str,
        default="./dataset/ExDark/images/val",
        help="ExDark validation images directory or val.txt path",
    )
    parser.add_argument(
        "--data_test_exdark",
        type=str,
        default="./dataset/ExDark/images/test",
        help="ExDark test images directory or test.txt path",
    )
    parser.add_argument(
        "--data_val_darkface",
        type=str,
        default="./dataset/DarkFace",
        help="DarkFace validation directory",
    )
    parser.add_argument(
        "--data_val_voc",
        type=str,
        default="./dataset/VOC/images/val",
        help="VOC validation images directory",
    )
    parser.add_argument(
        "--data_val_custom",
        type=str,
        default="./dataset/custom/images/val",
        help="Custom validation images directory",
    )

    parser.add_argument(
        "--val_folder",
        default="./results/",
        help="Location to save validation/evaluation results",
    )
    parser.add_argument(
        "--save_folder",
        default="./weights/training/",
        help="Location to save model checkpoints",
    )

    # 4. YOLOv5 Detection Loss weights & Hyperparameters
    parser.add_argument(
        "--box_weight",
        type=float,
        default=0.05,
        help="Bounding box regression loss gain (default: 0.05)",
    )
    parser.add_argument(
        "--obj_weight",
        type=float,
        default=1.0,
        help="Objectness confidence loss gain (default: 1.0)",
    )
    parser.add_argument(
        "--cls_weight",
        type=float,
        default=0.5,
        help="Classification loss gain (default: 0.5)",
    )
    parser.add_argument(
        "--anchor_t",
        type=float,
        default=4.0,
        help="Anchor-multiple aspect ratio threshold for target matching (default: 4.0)",
    )
    parser.add_argument(
        "--fl_gamma",
        type=float,
        default=0.0,
        help="Focal loss gamma (0.0 disables focal loss, 1.5 recommended when enabled)",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing epsilon for classification BCE (default: 0.0)",
    )
    parser.add_argument(
        "--iou_type",
        type=str,
        default="ciou",
        choices=["ciou", "diou", "giou", "eiou", "siou"],
        help="IoU loss formulation for bounding box regression (default: ciou)",
    )

    # 5. Data Augmentation & Low-Light Enhancement Curve settings
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Mosaic-4 augmentation probability (0.0 to 1.0)",
    )
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.0,
        help="MixUp augmentation probability (0.0 to 1.0)",
    )
    parser.add_argument(
        "--hsv_h",
        type=float,
        default=0.015,
        help="Image HSV-Hue augmentation (fraction)",
    )
    parser.add_argument(
        "--hsv_s",
        type=float,
        default=0.7,
        help="Image HSV-Saturation augmentation (fraction)",
    )
    parser.add_argument(
        "--hsv_v",
        type=float,
        default=0.4,
        help="Image HSV-Value (brightness) augmentation (fraction)",
    )
    parser.add_argument(
        "--degrees",
        type=float,
        default=0.0,
        help="Image rotation (+/- deg)",
    )
    parser.add_argument(
        "--translate",
        type=float,
        default=0.1,
        help="Image translation (+/- fraction)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.5,
        help="Image scale (+/- gain)",
    )
    parser.add_argument(
        "--shear",
        type=float,
        default=0.0,
        help="Image shear (+/- deg)",
    )
    parser.add_argument(
        "--fliplr",
        type=float,
        default=0.5,
        help="Image horizontal flip probability",
    )
    parser.add_argument(
        "--flipud",
        type=float,
        default=0.0,
        help="Image vertical flip probability",
    )

    # Random gamma function (low-light illumination curve augmentation)
    parser.add_argument(
        "--gamma",
        type=_str2bool,
        default=False,
        help="Apply random gamma curve augmentation for low-light generalization",
    )
    parser.add_argument("--start_gamma", type=int, default=60)
    parser.add_argument("--end_gamma", type=int, default=120)

    # 6. Gradient & Evaluation / NMS settings
    parser.add_argument(
        "--grad_detect",
        type=_str2bool,
        default=False,
        help="if gradient explosion occurs, turn-on it",
    )
    parser.add_argument(
        "--grad_clip",
        type=_str2bool,
        default=True,
        help="Clip gradient norm (max_norm=10.0, YOLOv5 standard)",
    )
    parser.add_argument(
        "--conf_thres",
        type=float,
        default=0.001,
        help="Object confidence threshold for evaluation NMS (0.001 for mAP, 0.25 for inference)",
    )
    parser.add_argument(
        "--iou_thres",
        type=float,
        default=0.6,
        help="IoU threshold for NMS (0.6 for mAP, 0.45 for inference)",
    )
    parser.add_argument(
        "--max_det",
        type=int,
        default=300,
        help="Maximum detections per image",
    )

    return parser
