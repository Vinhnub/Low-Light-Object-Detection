"""
YOLOv5 Object Detection Loss Functions (`loss/losses.py`).
Based on Ultralytics YOLOv5 (utils/loss.py) and structured in the modular style
of IGMamba-EdgeLCA-LLIE.

Includes:
  - ComputeLoss (YOLOv5DetectionLoss): Full multi-scale YOLOv5 loss combining:
      1. Bounding Box Regression Loss (CIoU / DIoU / GIoU / EIoU / SIoU)
      2. Objectness Confidence Loss (BCEWithLogitsLoss + per-scale balance [4.0, 1.0, 0.4])
      3. Multi-label Classification Loss (BCEWithLogitsLoss + Label Smoothing / Focal Loss)
      4. Anchor-aspect-ratio & 3-neighbor grid cell target assignment (`build_targets`)
  - Standalone modular loss classes: `YOLOv5BoxLoss`, `YOLOv5ObjLoss`, `YOLOv5ClsLoss`.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from loss.loss_utils import (
    BCEBlurWithLogitsLoss,
    FocalLoss,
    QFocalLoss,
    bbox_iou,
    smooth_BCE,
)


# Default YOLOv5 hyperparameters (hyp.scratch-low.yaml)
DEFAULT_HYP: Dict[str, float] = {
    "box": 0.05,            # box loss gain
    "cls": 0.5,             # cls loss gain
    "cls_pw": 1.0,          # cls BCELoss positive_weight
    "obj": 1.0,             # obj loss gain (scale with pixels)
    "obj_pw": 1.0,          # obj BCELoss positive_weight
    "anchor_t": 4.0,        # anchor-multiple threshold
    "fl_gamma": 0.0,        # focal loss gamma (efficientDet default gamma=1.5)
    "label_smoothing": 0.0, # label smoothing epsilon
}


def _de_parallel(model: nn.Module) -> nn.Module:
    """De-parallelize a model: returns single-GPU model if model is DP or DDP."""
    return model.module if hasattr(model, "module") else model


def _get_detect_head(model: nn.Module) -> nn.Module:
    """Extract the Detect() head from either DetectionModel or modular YOLOv5."""
    m = _de_parallel(model)
    if hasattr(m, "head"):
        return m.head
    if hasattr(m, "model"):
        return m.model[-1]
    raise AttributeError(
        "Model must have either a '.head' attribute (YOLOv5) or '.model[-1]' (DetectionModel)."
    )


class YOLOv5BoxLoss(nn.Module):
    """
    Standalone Bounding Box Regression Loss for YOLOv5.
    Supports CIoU (default), DIoU, GIoU, EIoU, and SIoU.
    """

    def __init__(self, iou_type: str = "ciou", loss_weight: float = 0.05):
        super().__init__()
        self.iou_type = iou_type.lower()
        self.loss_weight = loss_weight
        if self.iou_type not in {"ciou", "diou", "giou", "eiou", "siou", "iou"}:
            raise ValueError(f"Unsupported iou_type: {iou_type}")

    def forward(
        self, pbox: torch.Tensor, tbox: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pbox (Tensor): Predicted boxes [N_pos, 4] in grid (x, y, w, h) coordinates.
            tbox (Tensor): Target boxes [N_pos, 4] in grid (x, y, w, h) coordinates.
        Returns:
            loss (Tensor): Scalar box regression loss (unscaled by loss_weight or scaled).
            iou (Tensor): Pairwise IoU tensor [N_pos] for objectness target weighting.
        """
        iou = bbox_iou(
            pbox,
            tbox,
            xywh=True,
            CIoU=(self.iou_type == "ciou"),
            DIoU=(self.iou_type == "diou"),
            GIoU=(self.iou_type == "giou"),
            EIoU=(self.iou_type == "eiou"),
            SIoU=(self.iou_type == "siou"),
        ).squeeze(-1)
        loss = (1.0 - iou).mean()
        return loss, iou


class YOLOv5ObjLoss(nn.Module):
    """Standalone Objectness Loss for YOLOv5."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        fl_gamma: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        pw = torch.tensor([pos_weight], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        if fl_gamma > 0.0:
            bce = FocalLoss(bce, gamma=fl_gamma)
        self.bce = bce

    def forward(self, pred_obj: torch.Tensor, target_obj: torch.Tensor) -> torch.Tensor:
        return self.bce(pred_obj, target_obj)


class YOLOv5ClsLoss(nn.Module):
    """Standalone Multi-Label Classification Loss for YOLOv5."""

    def __init__(
        self,
        pos_weight: float = 1.0,
        fl_gamma: float = 0.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        pw = torch.tensor([pos_weight], device=device)
        bce = nn.BCEWithLogitsLoss(pos_weight=pw)
        if fl_gamma > 0.0:
            bce = FocalLoss(bce, gamma=fl_gamma)
        self.bce = bce

    def forward(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        return self.bce(pred_cls, target_cls)


class ComputeLoss(nn.Module):
    """
    Official Ultralytics YOLOv5 ComputeLoss.
    Computes total loss and individual loss components (lbox, lobj, lcls)
    across all feature pyramid scales (P3/8, P4/16, P5/32).
    """

    sort_obj_iou: bool = False

    def __init__(
        self,
        model: nn.Module,
        autobalance: bool = False,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        iou_type: str = "ciou",
    ):
        super().__init__()
        device = next(model.parameters()).device  # get model device

        # Merge hyperparameters from model.hyp, passed hyp dict, or argparse Namespace
        h = dict(DEFAULT_HYP)
        if hasattr(model, "hyp") and isinstance(model.hyp, dict):
            h.update(model.hyp)
        if hyp is not None:
            if isinstance(hyp, dict):
                h.update(hyp)
            else:
                # Extract from argparse Namespace (data/options.py)
                if hasattr(hyp, "box_weight"):
                    h["box"] = float(hyp.box_weight)
                if hasattr(hyp, "obj_weight"):
                    h["obj"] = float(hyp.obj_weight)
                if hasattr(hyp, "cls_weight"):
                    h["cls"] = float(hyp.cls_weight)
                if hasattr(hyp, "anchor_t"):
                    h["anchor_t"] = float(hyp.anchor_t)
                if hasattr(hyp, "fl_gamma"):
                    h["fl_gamma"] = float(hyp.fl_gamma)
                if hasattr(hyp, "label_smoothing"):
                    h["label_smoothing"] = float(hyp.label_smoothing)
                if hasattr(hyp, "iou_type"):
                    iou_type = str(hyp.iou_type)

        m = _get_detect_head(model)  # Detect() module

        # Scale cls & box weights by number of layers (nl=3) and classes (nc)
        h["box"] *= 3.0 / m.nl
        h["cls"] *= (m.nc / 80.0) * (3.0 / m.nl)
        h["obj"] *= 1.0  # scaled dynamically if image size != 640

        self.hyp = h
        self.iou_type = iou_type.lower()

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["cls_pw"]], device=device)
        )
        BCEobj = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([h["obj_pw"]], device=device)
        )

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        # Focal loss
        g = h["fl_gamma"]  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        # Layer balance weights for P3/8, P4/16, P5/32
        self.balance = {3: [4.0, 1.0, 0.4]}.get(
            m.nl, [4.0, 1.0, 0.25, 0.06, 0.02]
        )
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
        self.autobalance = autobalance

        self.na = m.na  # number of anchors (3)
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers (3)
        self.anchors = m.anchors  # shape (nl, na, 2) in grid coordinates
        self.device = device

    def __call__(
        self,
        p: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute YOLOv5 loss.

        Args:
            p (List[Tensor]): List of 3 prediction tensors from Detect head:
                - P3/8:  [B, na, H/8,  W/8,  nc + 5]
                - P4/16: [B, na, H/16, W/16, nc + 5]
                - P5/32: [B, na, H/32, W/32, nc + 5]
            targets (Tensor): Ground-truth targets of shape [N_targets, 6],
                where each row is [image_idx, class_id, x_center, y_center, w, h]
                with (x_center, y_center, w, h) normalized to [0, 1].

        Returns:
            total_loss (Tensor): Scalar loss multiplied by batch size `bs` (for gradient accumulation compatibility).
            loss_items (Tensor): Detached tensor `torch.cat((lbox, lobj, lcls))` of shape [3] for logging.
        """
        # Handle case where model was called in eval mode returning (decoded_out, raw_preds)
        if isinstance(p, tuple) and len(p) == 2 and isinstance(p[1], list):
            p = p[1]

        device = targets.device
        lcls = torch.zeros(1, device=device)  # class loss
        lbox = torch.zeros(1, device=device)  # box loss
        lobj = torch.zeros(1, device=device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses across feature pyramid levels
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)  # target obj

            n = b.shape[0]  # number of positive targets at scale i
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                iou = bbox_iou(
                    pbox,
                    tbox[i],
                    xywh=True,
                    CIoU=(self.iou_type == "ciou"),
                    DIoU=(self.iou_type == "diou"),
                    GIoU=(self.iou_type == "giou"),
                    EIoU=(self.iou_type == "eiou"),
                    SIoU=(self.iou_type == "siou"),
                ).squeeze(-1)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = p[0].shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(
        self,
        p: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        List[torch.Tensor],
    ]:
        """
        Build targets for ComputeLoss(), matching ground-truth boxes to anchors
        via width/height aspect ratio threshold (`anchor_t`) and assigning positive
        samples to the center grid cell plus its 2 nearest neighbor cells.
        """
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(
            7, device=targets.device
        )  # normalized to gridspace gain: [img_idx, cls, x, y, w, h, anchor_idx]
        ai = (
            torch.arange(na, device=targets.device)
            .float()
            .view(na, 1)
            .repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        targets = torch.cat(
            (targets.repeat(na, 1, 1), ai[..., None]), 2
        )  # append anchor indices: shape (na, nt, 7)

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i].to(targets.device)
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain: [nx, ny, nx, ny]

            # Match targets to anchors
            t = targets * gain  # shape(3, n, 7)
            if nt:
                # Matches via aspect-ratio threshold
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = (
                    torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                )  # compare
                t = t[j]  # filter: shape (N_matched, 7)

                # Offsets (neighboring grid cells)
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append(
                (
                    b,
                    a,
                    gj.clamp_(0, shape[2] - 1),
                    gi.clamp_(0, shape[3] - 1),
                )
            )  # image, anchor, gridy, gridx
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box offset & wh
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch


# Alias for naming consistency
YOLOv5DetectionLoss = ComputeLoss
