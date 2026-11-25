"""
Loss utilities and bounding box IoU metrics for Object Detection.
Includes:
  - bbox_iou: Standard IoU, GIoU, DIoU, CIoU, EIoU, SIoU, WIoU (v1/v3)
  - smooth_BCE: Label smoothing helper for binary cross-entropy
  - BlurredAsymmetricBCELoss: Group 1 noise- & missing-label-tolerant classification loss
  - FocalLoss, QFocalLoss, BCEBlurWithLogitsLoss
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class BlurredAsymmetricBCELoss(nn.Module):
    """
    Group 1 Noise- & Missing-Label-Tolerant Classification Loss for Low-Light Object Detection.

    Integrates:
      1. Positive class weighting (`pos_weight >= 1.0`) to penalize false negatives and improve recall.
      2. Asymmetric probability margin shifting (`neg_margin >= 0`) & negative focusing (`neg_gamma >= 0`)
         on background anchors (`~fg_mask`) so massive dark background grids do not suppress dim objects.
      3. Warmup-gated high-confidence negative suppression (`neg_blur_tau < 1.0`, `neg_blur_alpha > 0`)
         using a detached exponential decay `exp(-(p - tau) / alpha)` when `p >= tau` to forgive missing
         dark annotations and unannotated out-of-taxonomy objects (e.g., COCO objects outside ExDark-12).
      4. Symmetric Reverse Cross-Entropy (`rce_weight > 0`, bounded-gradient `|p - y|`) to resist
         noisy / mislabeled ground-truth class labels.
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        rce_weight: float = 0.0,
        neg_margin: float = 0.0,
        neg_gamma: float = 0.0,
        neg_blur_tau: float = 1.0,
        neg_blur_alpha: float = 0.10,
        blur_warmup_epochs: int = 15,
        reduction: str = "none",
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.rce_weight = float(rce_weight)
        self.neg_margin = float(neg_margin)
        self.neg_gamma = float(neg_gamma)
        self.neg_blur_tau = float(neg_blur_tau)
        self.neg_blur_alpha = max(float(neg_blur_alpha), 1e-4)
        self.blur_warmup_epochs = max(int(blur_warmup_epochs), 0)
        self.reduction = reduction
        self.completed_epochs: int = 0
        self.register_buffer(
            "_pw_buf", torch.tensor([self.pos_weight], dtype=torch.float32), persistent=False
        )

    def update(self) -> None:
        """Advance completed epoch counter once per training epoch."""
        self.completed_epochs += 1

    def sync_updates(self, completed_epochs: int) -> None:
        """Synchronize completed epoch counter when resuming from a checkpoint."""
        self.completed_epochs = max(int(completed_epochs), 0)

    @property
    def is_baseline_bce(self) -> bool:
        blur_active = (
            self.neg_blur_tau < 1.0 and self.completed_epochs >= self.blur_warmup_epochs
        )
        return (
            self.pos_weight == 1.0
            and self.rce_weight <= 0.0
            and self.neg_margin <= 0.0
            and self.neg_gamma <= 0.0
            and not blur_active
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        fg_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_baseline_bce:
            return F.binary_cross_entropy_with_logits(
                pred, target, reduction=self.reduction
            )

        pw = (
            self._pw_buf.to(device=pred.device, dtype=pred.dtype)
            if self.pos_weight != 1.0
            else None
        )

        # If caller passes an empty or all-foreground tensor (e.g., YOLOv5 pcls on positives only)
        if fg_mask is not None and fg_mask.all():
            loss_fg = F.binary_cross_entropy_with_logits(
                pred, target, pos_weight=pw, reduction="none"
            )
            if self.rce_weight > 0.0:
                loss_fg = loss_fg + self.rce_weight * torch.abs(torch.sigmoid(pred) - target)
            return self._reduce(loss_fg)

        if fg_mask is not None:
            bg_mask = ~fg_mask.bool()
            if bg_mask.ndim == pred.ndim - 1:
                bg_mask = bg_mask.unsqueeze(-1).expand_as(pred)
        else:
            bg_mask = target <= 0.0

        prob = torch.sigmoid(pred)

        # 1. Foreground entries (~bg_mask): positive-weighted BCE + optional Symmetric RCE
        loss_fg = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pw, reduction="none"
        )
        if self.rce_weight > 0.0:
            loss_fg = loss_fg + self.rce_weight * torch.abs(prob - target)

        # 2. Pure background entries (bg_mask, y == 0): Asymmetric Margin + Focusing + High-Conf Blur
        if self.neg_margin > 0.0:
            prob_m = (prob - self.neg_margin).clamp(min=0.0, max=1.0 - 1e-6)
            loss_bg = -torch.log1p(-prob_m)
        else:
            prob_m = prob
            loss_bg = F.binary_cross_entropy_with_logits(
                pred, torch.zeros_like(pred), reduction="none"
            )

        if self.rce_weight > 0.0:
            loss_bg = loss_bg + self.rce_weight * prob_m

        if self.neg_gamma > 0.0:
            loss_bg = loss_bg * (prob_m ** self.neg_gamma)

        if self.neg_blur_tau < 1.0 and self.completed_epochs >= self.blur_warmup_epochs:
            with torch.no_grad():
                excess = (prob - self.neg_blur_tau).clamp(min=0.0)
                w_blur = torch.exp(-excess / self.neg_blur_alpha)
            loss_bg = loss_bg * w_blur

        loss = torch.where(bg_mask, loss_bg, loss_fg)
        return self._reduce(loss)

    def _reduce(self, loss: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
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
    WIoU: bool = False,
    wiou_iou_mean: Optional[torch.Tensor] = None,
    update_wiou_mean: bool = False,
    wiou_momentum: float = 0.02,
    wiou_alpha: float = 1.9,
    wiou_delta: float = 3.0,
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
        CIoU (bool): Complete IoU (https://arxiv.org/abs/1911.08287) - default
        EIoU (bool): Efficient IoU (https://arxiv.org/abs/2101.08158)
        SIoU (bool): SCYLLA-IoU (https://arxiv.org/abs/2205.12740)
        WIoU (bool): Wise-IoU v1/v3 (https://arxiv.org/abs/2301.10051) for noisy low-light GT boxes
        wiou_iou_mean (Optional[torch.Tensor]): Running EMA buffer of `1 - IoU` for Wise-IoU v3.
        update_wiou_mean (bool): Whether to update `wiou_iou_mean` with the current batch mean.
        wiou_momentum (float): EMA update momentum for `wiou_iou_mean` (default: 0.02).
        wiou_alpha (float): Wise-IoU v3 non-monotonic focusing alpha (default: 1.9).
        wiou_delta (float): Wise-IoU v3 outlier threshold delta (default: 3.0).
        eps (float): Small constant to avoid division by zero.

    Returns:
        iou (torch.Tensor): IoU / GIoU / DIoU / CIoU / EIoU / SIoU / WIoU effective values
                            such that `1.0 - return_value` equals the corresponding regression loss.
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
    if CIoU or DIoU or GIoU or EIoU or SIoU or WIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        if CIoU or DIoU or EIoU or SIoU or WIoU:
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
            elif WIoU:
                # Wise-IoU v1/v3: detach smallest enclosing diagonal c2 to prevent box expansion,
                # and apply non-monotonic focusing coefficient r(beta) to suppress noisy GT boxes.
                l_iou = 1.0 - iou
                r_wiou = torch.exp((rho2 / c2.detach()).clamp(max=10.0))
                if wiou_iou_mean is not None:
                    with torch.no_grad():
                        if update_wiou_mean and l_iou.numel() > 0:
                            batch_mean = l_iou.detach().float().mean()
                            wiou_iou_mean.mul_(1.0 - wiou_momentum).add_(
                                batch_mean.to(wiou_iou_mean.dtype), alpha=wiou_momentum
                            )
                        beta = l_iou.detach().float() / wiou_iou_mean.float().clamp(min=eps)
                        exponent = (beta - wiou_delta).clamp(min=-10.0, max=15.0)
                        r_focus = (
                            beta / (wiou_delta * torch.pow(wiou_alpha, exponent) + eps)
                        ).to(l_iou.dtype)
                    l_wiou = r_focus * r_wiou * l_iou
                else:
                    l_wiou = r_wiou * l_iou
                return 1.0 - l_wiou
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU
    return iou
