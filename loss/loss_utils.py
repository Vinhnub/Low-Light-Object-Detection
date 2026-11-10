"""
Loss utilities and bounding box IoU metrics for YOLOv5 Object Detection.
Includes:
  - bbox_iou: Standard IoU, GIoU, DIoU, CIoU, EIoU, SIoU (for YOLOv5 box regression)
  - smooth_BCE: Label smoothing helper for binary cross-entropy
  - FocalLoss, QFocalLoss, BCEBlurWithLogitsLoss
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


def smooth_BCE(eps: float = 0.1) -> Tuple[float, float]:
    """
    Return positive and negative label smoothing BCE targets.
    Reference: https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    """
    BCEwithLogitLoss() with reduced missing label effects.
    Official YOLOv5 implementation.
    """

    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction="none")
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Wraps focal loss around existing PyTorch_modelling BCEWithLogitsLoss.
    l(x, y) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        loss_fcn: nn.Module,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class QFocalLoss(nn.Module):
    """
    Quality Focal Loss (QFL) for continuous IoU-aware targets in object detection.
    Reference: https://arxiv.org/abs/2006.04388
    """

    def __init__(
        self,
        loss_fcn: nn.Module,
        gamma: float = 1.5,
        alpha: float = 0.25,
    ):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    GIoU: bool = False,
    DIoU: bool = False,
    CIoU: bool = False,
    EIoU: bool = False,
    SIoU: bool = False,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Intersection over Union (IoU) of box1(1, 4) to box2(n, 4).

    Args:
        box1 (torch.Tensor): Predicted boxes, shape (N, 4) or (1, 4).
        box2 (torch.Tensor): Target boxes, shape (N, 4).
        xywh (bool): If True, boxes are in (x_center, y_center, w, h) format;
                     otherwise (x1, y1, x2, y2).
        GIoU (bool): Generalized IoU (https://arxiv.org/abs/1902.09630)
        DIoU (bool): Distance IoU (https://arxiv.org/abs/1911.08287)
        CIoU (bool): Complete IoU (https://arxiv.org/abs/1911.08287) - YOLOv5 default
        EIoU (bool): Efficient IoU (https://arxiv.org/abs/2101.08158)
        SIoU (bool): SCYLLA-IoU (https://arxiv.org/abs/2205.12740)
        eps (float): Small constant to avoid division by zero.

    Returns:
        iou (torch.Tensor): IoU / GIoU / DIoU / CIoU / EIoU / SIoU values.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, (b1_y2 - b1_y1).clamp(eps)
        w2, h2 = b2_x2 - b2_x1, (b2_y2 - b2_y1).clamp(eps)

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU or EIoU or SIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU:
            c2 = cw**2 + ch**2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2
                + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center dist ** 2
            if CIoU:
                v = (4 / math.pi**2) * (
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))
                ).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            elif EIoU:
                rho_w2 = (w2 - w1) ** 2
                rho_h2 = (h2 - h1) ** 2
                cw2 = cw**2 + eps
                ch2 = ch**2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)  # EIoU
            elif SIoU:
                # SIoU Loss
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw**2 + s_ch**2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma
                sin_alpha_2 = torch.abs(s_ch) / sigma
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(
                    sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1
                )
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / (cw + eps)) ** 2
                rho_y = (s_ch / (ch + eps)) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(
                    1 - torch.exp(-1 * omiga_h), 4
                )
                return iou - 0.5 * (distance_cost + shape_cost)
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    return iou
