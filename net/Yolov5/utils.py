"""
Utility functions for YOLOv5 architecture, weight initialization,
structural re-parameterization (Conv+BN fusion), bounding box conversion,
and Non-Maximum Suppression (NMS).
"""

import copy
import math
import os
import sys
import time
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def is_parallel(model: nn.Module) -> bool:
    """Returns True if model is of type DP or DDP."""
    return isinstance(
        model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
    )


def de_parallel(model: nn.Module) -> nn.Module:
    """De-parallelize a model: returns single-GPU model if model is of type DP or DDP."""
    return model.module if is_parallel(model) else model


class ModelEMA:
    """
    Updated Exponential Moving Average (EMA) from Ultralytics YOLOv5.
    Keeps a moving average of everything in the model state_dict (parameters and buffers).
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        tau: float = 2000.0,
        updates: int = 0,
    ):
        self.ema = copy.deepcopy(de_parallel(model)).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (
            1.0 - math.exp(-x / tau)
        )  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self._cached_model_id: Optional[int] = None
        self._ema_fp: List[torch.Tensor] = []
        self._mod_fp: List[torch.Tensor] = []

    def update(self, model: nn.Module) -> None:
        """Update EMA parameters and buffers using fused multi-tensor foreach ops."""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            raw_m = de_parallel(model)
            if self._cached_model_id != id(raw_m) or not self._ema_fp:
                msd = raw_m.state_dict()
                esd = self.ema.state_dict()
                self._ema_fp = [
                    v for k, v in esd.items() if v.dtype.is_floating_point and k in msd
                ]
                self._mod_fp = [
                    msd[k] for k, v in esd.items() if v.dtype.is_floating_point and k in msd
                ]
                self._cached_model_id = id(raw_m)
            if self._ema_fp:
                torch._foreach_mul_(self._ema_fp, d)
                torch._foreach_add_(self._ema_fp, self._mod_fp, alpha=1.0 - d)


def smart_optimizer(
    model: nn.Module,
    name: str = "Adam",
    lr: float = 1e-4,
    momentum: float = 0.937,
    decay: float = 5e-5,
    verbose: bool = True,
) -> torch.optim.Optimizer:
    """
    Ultralytics YOLOv5 3-group smart optimizer:
      - g0: BatchNorm weights (no weight_decay)
      - g1: Conv / Linear weights (with weight_decay)
      - g2: Biases (no weight_decay)
    Prevents weight_decay from shrinking BatchNorm scale (gamma) and Detect head biases.
    """
    g = [], [], []  # optimizer parameter groups: [g0 (bn), g1 (weights), g2 (biases)]
    bn = tuple(
        v for k, v in nn.__dict__.items() if "Norm" in k
    )  # normalization layers, i.e. BatchNorm2d()
    for v in de_parallel(model).modules():
        if hasattr(v, "bias") and isinstance(
            v.bias, nn.Parameter
        ):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[0].append(v.weight)
        elif hasattr(v, "weight") and isinstance(
            v.weight, nn.Parameter
        ):  # weight (with decay)
            g[1].append(v.weight)

    opt_name = name.upper()
    if opt_name == "ADAM":
        optimizer = torch.optim.Adam(
            g[2], lr=lr, betas=(momentum, 0.999)
        )  # adjust beta1 to momentum
    elif opt_name == "ADAMW":
        optimizer = torch.optim.AdamW(
            g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0
        )
    elif opt_name == "SGD":
        optimizer = torch.optim.SGD(
            g[2], lr=lr, momentum=momentum, nesterov=True
        )
    else:
        raise NotImplementedError(f"Optimizer {name} not implemented.")

    optimizer.add_param_group(
        {"params": g[1], "weight_decay": decay}
    )  # add g1 with weight_decay
    optimizer.add_param_group(
        {"params": g[0], "weight_decay": 0.0}
    )  # add g0 (BatchNorm) without weight_decay
    if verbose:
        print(
            f"===> Optimizer: {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups: "
            f"{len(g[0])} weight(decay=0.0), {len(g[1])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
    return optimizer


PRETRAINED_URLS: Dict[str, str] = {
    "yolov5n": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt",
    "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt",
    "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt",
    "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt",
    "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt",
}


def load_yolov5_checkpoint(
    model: nn.Module,
    ckpt_path: str,
    model_name: str = "yolov5s",
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Load either an official Ultralytics YOLOv5 COCO pretrained `.pt` file
    (e.g., `yolov5s.pt`, `yolov5n.pt`) or a custom training `.pth` checkpoint
    into `model` (supporting both `DetectionModel` and modular `YOLOv5`).
    Automatically downloads official `.pt` weights into `./weights/pretrained/` if `"auto"` is passed.
    """
    if not ckpt_path or str(ckpt_path).lower() in {"none", "false", "no"}:
        return None

    m_key = model_name.lower().replace(".pt", "").replace(".pth", "")
    if str(ckpt_path).lower() == "auto":
        os.makedirs("./weights/pretrained", exist_ok=True)
        ckpt_path = f"./weights/pretrained/{m_key}.pt"

    if not os.path.exists(ckpt_path) and m_key in PRETRAINED_URLS:
        os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
        url = PRETRAINED_URLS[m_key]
        if verbose:
            print(f"===> Downloading official COCO pretrained weights {m_key}.pt from {url} ...")
        urllib.request.urlretrieve(url, ckpt_path)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # Register module aliases so official Ultralytics pickled .pt files unpickle cleanly
    import net.Yolov5 as y5_pkg

    sys.modules.setdefault("models", y5_pkg)
    sys.modules.setdefault("models.common", y5_pkg.common)
    sys.modules.setdefault("models.yolo", y5_pkg.yolo)
    sys.modules.setdefault("utils", types.ModuleType("utils"))
    sys.modules.setdefault("utils.loss", types.ModuleType("utils.loss"))

    checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state_dict = checkpoint_data
    if isinstance(checkpoint_data, dict):
        if "state_dict" in checkpoint_data:
            raw_state_dict = checkpoint_data["state_dict"]
        elif "ema" in checkpoint_data and checkpoint_data["ema"] is not None:
            ema_obj = checkpoint_data["ema"]
            raw_state_dict = (
                ema_obj.float().state_dict()
                if hasattr(ema_obj, "state_dict")
                else ema_obj
            )
        elif "model" in checkpoint_data and checkpoint_data["model"] is not None:
            m_obj = checkpoint_data["model"]
            raw_state_dict = (
                m_obj.float().state_dict()
                if hasattr(m_obj, "state_dict")
                else m_obj
            )

    # Map sequential `model.0..24.*` keys to modular `backbone.*` / `neck.*` / `head.*` if needed
    seq_to_mod_prefix = {
        "model.0.": "backbone.stem.",
        "model.1.": "backbone.stage1.0.",
        "model.2.": "backbone.stage1.1.",
        "model.3.": "backbone.stage2.0.",
        "model.4.": "backbone.stage2.1.",
        "model.5.": "backbone.stage3.0.",
        "model.6.": "backbone.stage3.1.",
        "model.7.": "backbone.stage4.0.",
        "model.8.": "backbone.stage4.1.",
        "model.9.": "backbone.stage4.2.",
        "model.10.": "neck.reduce_p5.",
        "model.13.": "neck.c3_p4.",
        "model.14.": "neck.reduce_p4.",
        "model.17.": "neck.c3_p3.",
        "model.18.": "neck.down_p3.",
        "model.20.": "neck.c3_n4.",
        "model.21.": "neck.down_p4.",
        "model.23.": "neck.c3_n5.",
        "model.24.": "head.",
    }

    model_dict = model.state_dict()
    filtered_state_dict = {}
    for k, v in raw_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            v = v.float()
        if name not in model_dict:
            for seq_p, mod_p in seq_to_mod_prefix.items():
                if name.startswith(seq_p):
                    name = mod_p + name[len(seq_p) :]
                    break
        if name in model_dict and model_dict[name].shape == v.shape:
            filtered_state_dict[name] = v

    model.load_state_dict(filtered_state_dict, strict=False)
    if verbose:
        print(
            f"===> Loaded {len(filtered_state_dict)}/{len(model_dict)} weight tensors from '{ckpt_path}'"
        )
    return checkpoint_data if isinstance(checkpoint_data, dict) else None


def make_divisible(x: float, divisor: int = 8) -> int:
    """Returns nearest x divisible by divisor."""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights to standard YOLOv5 defaults."""
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True


def fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """
    Fuse Conv2d() and BatchNorm2d() layers for faster inference.
    Reference: https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    """
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # Prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(
        torch.sqrt(bn.running_var + bn.eps)
    )
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def scale_img(
    img: torch.Tensor,
    ratio: float = 1.0,
    same_shape: bool = False,
    gs: int = 32,
) -> torch.Tensor:
    """
    Scales img(bs,3,y,x) by ratio constrained to gs-multiple.
    Used for test-time augmentation (TTA).
    """
    if ratio == 1.0:
        return img
    h, w = img.shape[2:]
    s = (int(h * ratio), int(w * ratio))  # new size
    img = F.interpolate(img, size=s, mode="bilinear", align_corners=False)
    if not same_shape:  # pad/crop img
        h, w = (math.ceil(x * ratio / gs) * gs for x in (h, w))
    return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def check_anchor_order(m: nn.Module) -> None:
    """
    Check anchor order against stride order for YOLOv5 Detect() module m,
    and correct if necessary.
    """
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        m.anchors[:] = m.anchors.flip(0)


def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]."""
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def box_iou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise IoU values
    """
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def clip_boxes(boxes: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """Clip boxes (xyxy) to image shape (height, width)."""
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])
    return boxes


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: torch.Tensor,
    img0_shape: Tuple[int, int],
    ratio_pad: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> torch.Tensor:
    """Rescale boxes (xyxy) from img1_shape to img0_shape."""
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2,
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def non_max_suppression(
    prediction: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    agnostic: bool = False,
    multi_label: bool = False,
    labels: Tuple = (),
    max_det: int = 300,
    nm: int = 0,  # number of masks
    end2end: bool = False,
) -> List[torch.Tensor]:
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.
    Supports YOLOv5 `(B, N, 5+nc+nm)`, YOLOv8/YOLOv11 `(B, 4+nc+nm, N)`,
    and YOLOv26 end-to-end `(B, K, 6+nm)`.

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    if isinstance(prediction, (list, tuple)):
        # Model in validation mode: output = (inference_out, loss_out)
        prediction = prediction[0]

    # Delegate End-to-End (YOLOv26) or Anchor-Free (YOLOv11) formats cleanly
    is_e2e = end2end or (
        prediction.ndim == 3 and prediction.shape[-1] == 6 + nm and prediction.shape[1] <= 1000
    )
    is_anchor_free = (
        not is_e2e and prediction.ndim == 3 and prediction.shape[1] < prediction.shape[2]
    )
    if is_e2e or is_anchor_free:
        from net.Yolov11.utils import non_max_suppression as unified_nms

        return unified_nms(
            prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=classes,
            agnostic=agnostic,
            multi_label=multi_label,
            labels=labels,
            max_det=max_det,
            nm=nm,
            end2end=end2end,
        )

    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output
