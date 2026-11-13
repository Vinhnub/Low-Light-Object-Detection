"""
Object Detection Loss Functions (`loss/losses.py`).
Supports YOLOv5, YOLOv11 (`v8DetectionLoss`), and YOLOv26 (`E2ELoss` / `E2EDetectLoss`).

Includes:
  - YOLOv5: `YOLOv5DetectionLoss`, `YOLOv5BoxLoss`, `YOLOv5ObjLoss`, `YOLOv5ClsLoss`
  - YOLOv11: `TaskAlignedAssigner`, `DFLoss`, `BboxLoss`, `YOLOv11DetectionLoss` (`v8DetectionLoss`)
  - YOLOv26: `YOLOv26DetectionLoss` (`E2ELoss`, `E2EDetectLoss`) with one2many/one2one decay schedule
  - Unified `ComputeLoss` auto-dispatching based on model head architecture
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.loss_utils import (
    BCEBlurWithLogitsLoss,
    FocalLoss,
    QFocalLoss,
    bbox_iou,
    smooth_BCE,
)
from net.Yolov11.utils import bbox2dist, dist2bbox, make_anchors, xywh2xyxy


# Default YOLOv5 hyperparameters (hyp.scratch-low.yaml)
DEFAULT_HYP: Dict[str, float] = {
    "box": 0.05,            # box loss gain
    "cls": 0.5,             # cls loss gain
    "cls_pw": 1.0,          # cls BCELoss positive_weight
    "obj": 1.0,             # obj loss gain (scale with pixels)
    "obj_pw": 1.0,          # obj BCELoss positive_weight
    "anchor_t": 4.0,        # anchor-multiple threshold
    "fl_gamma": 0.0,        # focal loss gamma
    "label_smoothing": 0.0, # label smoothing epsilon
}

# Default YOLOv11 / YOLOv26 hyperparameters (Ultralytics default.yaml)
DEFAULT_V11_HYP: Dict[str, float] = {
    "box": 7.5,             # box loss gain
    "cls": 0.5,             # cls loss gain
    "dfl": 1.5,             # dfl / l1 box regression loss gain
}


def _de_parallel(model: nn.Module) -> nn.Module:
    """De-parallelize a model: returns single-GPU model if model is DP or DDP."""
    return model.module if hasattr(model, "module") else model


def _get_detect_head(model: nn.Module) -> nn.Module:
    """Extract the Detect() head from either DetectionModel or modular YOLO."""
    m = _de_parallel(model)
    if hasattr(m, "head"):
        return m.head
    if hasattr(m, "model"):
        return m.model[-1]
    raise AttributeError(
        "Model must have either a '.head' attribute or '.model[-1]' (DetectionModel)."
    )


def _resolve_v11_hyp(hyp: Optional[Union[Dict[str, Any], Any]] = None) -> Dict[str, float]:
    """Extract YOLOv11/YOLOv26 loss gains from dict or argparse Namespace."""
    h = dict(DEFAULT_V11_HYP)
    if hyp is None:
        return h
    if isinstance(hyp, dict):
        for k in ("box", "cls", "dfl"):
            if hyp.get(k) is not None:
                h[k] = float(hyp[k])
        return h
    if getattr(hyp, "box_weight", None) is not None:
        h["box"] = float(hyp.box_weight)
    elif getattr(hyp, "box", None) is not None:
        h["box"] = float(hyp.box)
    if getattr(hyp, "cls_weight", None) is not None:
        h["cls"] = float(hyp.cls_weight)
    elif getattr(hyp, "cls", None) is not None:
        h["cls"] = float(hyp.cls)
    if getattr(hyp, "dfl_weight", None) is not None:
        h["dfl"] = float(hyp.dfl_weight)
    elif getattr(hyp, "dfl", None) is not None:
        h["dfl"] = float(hyp.dfl)
    return h


# ==================================================================================================
# 1. Standalone Modular YOLOv5 Loss Components
# ==================================================================================================

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


# ==================================================================================================
# 2. YOLOv5 Anchor-Based ComputeLoss
# ==================================================================================================

class YOLOv5DetectionLoss(nn.Module):
    """
    Official Ultralytics YOLOv5 ComputeLoss.
    Computes total loss and individual loss components (lbox, lobj, lcls)
    across all feature pyramid scales (P3/8, P4/16, P5/32).
    """

    sort_obj_iou: bool = False
    loss_names: Tuple[str, str, str] = ("box", "obj", "cls")

    def __init__(
        self,
        model: nn.Module,
        autobalance: bool = False,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        iou_type: str = "ciou",
    ):
        super().__init__()
        device = next(model.parameters()).device

        h = dict(DEFAULT_HYP)
        if hasattr(model, "hyp") and isinstance(model.hyp, dict):
            h.update(model.hyp)
        if hyp is not None:
            if isinstance(hyp, dict):
                h.update(hyp)
            else:
                if getattr(hyp, "box_weight", None) is not None:
                    h["box"] = float(hyp.box_weight)
                if getattr(hyp, "obj_weight", None) is not None:
                    h["obj"] = float(hyp.obj_weight)
                if getattr(hyp, "cls_weight", None) is not None:
                    h["cls"] = float(hyp.cls_weight)
                if getattr(hyp, "anchor_t", None) is not None:
                    h["anchor_t"] = float(hyp.anchor_t)
                if getattr(hyp, "fl_gamma", None) is not None:
                    h["fl_gamma"] = float(hyp.fl_gamma)
                if getattr(hyp, "label_smoothing", None) is not None:
                    h["label_smoothing"] = float(hyp.label_smoothing)
                if getattr(hyp, "iou_type", None) is not None:
                    iou_type = str(hyp.iou_type)

        m = _get_detect_head(model)

        h["box"] *= 3.0 / m.nl
        h["cls"] *= (m.nc / 80.0) * (3.0 / m.nl)
        h["obj"] *= 1.0

        self.hyp = h
        self.iou_type = iou_type.lower()

        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["obj_pw"]], device=device))

        self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))

        g = h["fl_gamma"]
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)

        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])
        self.ssi = list(m.stride).index(16) if autobalance else 0
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.gr = 1.0
        self.autobalance = autobalance

        self.na = m.na
        self.nc = m.nc
        self.nl = m.nl
        self.anchors = m.anchors
        self.device = device

    def __call__(
        self,
        p: List[torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(p, tuple) and len(p) == 2 and isinstance(p[1], list):
            p = p[1]

        device = targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)

        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=device)

            n = b.shape[0]
            if n:
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)

                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)
                iou = bbox_iou(
                    pbox,
                    tbox[i],
                    xywh=True,
                    CIoU=(self.iou_type == "ciou"),
                    DIoU=(self.iou_type == "diou"),
                    GIoU=(self.iou_type == "giou"),
                    EIoU=(self.iou_type == "eiou"),
                    SIoU=(self.iou_type == "siou"),
                ).squeeze(-1)
                lbox += (1.0 - iou).mean()

                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou

                if self.nc > 1:
                    t = torch.full_like(pcls, self.cn, device=device)
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]
            if self.autobalance:
                self.balance[i] = (
                    self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
                )

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]

        lbox *= self.hyp["box"]
        lobj *= self.hyp["obj"]
        lcls *= self.hyp["cls"]
        bs = p[0].shape[0]

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
        na, nt = self.na, targets.shape[0]
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=targets.device)
        ai = (
            torch.arange(na, device=targets.device)
            .float()
            .view(na, 1)
            .repeat(1, nt)
        )
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)

        g = 0.5
        off = (
            torch.tensor(
                [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
                device=targets.device,
            ).float()
            * g
        )

        for i in range(self.nl):
            anchors = self.anchors[i].to(targets.device)
            shape = p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]

            t = targets * gain
            if nt:
                r = t[..., 4:6] / anchors[:, None]
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp["anchor_t"]
                t = t[j]

                gxy = t[:, 2:4]
                gxi = gain[[2, 3]] - gxy
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            bc, gxy, gwh, a = t.chunk(4, 1)
            a, (b, c) = a.long().view(-1), bc.long().T
            gij = (gxy - offsets).long()
            gi, gj = gij.T

            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))
            anch.append(anchors[a])
            tcls.append(c)

        return tcls, tbox, indices, anch


# ==================================================================================================
# 3. Ultralytics TaskAlignedAssigner, DFLoss, BboxLoss (YOLOv11 & YOLOv26)
# ==================================================================================================

def select_candidates_in_gts(
    xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9
) -> torch.Tensor:
    """
    Select positive anchor centers within ground-truth bounding boxes.
    Args:
        xy_centers: (N_anchors, 2)
        gt_bboxes: (B, N_gt, 4) in xyxy format
    Returns:
        Boolean/float mask of shape (B, N_gt, N_anchors).
    """
    n_anchors = xy_centers.shape[0]
    bs, n_boxes, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(
        bs, n_boxes, n_anchors, -1
    )
    return bbox_deltas.amin(3).gt_(eps)


def select_highest_overlaps(
    mask_pos: torch.Tensor, overlaps: torch.Tensor, n_max_boxes: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Resolve anchors assigned to multiple ground-truth boxes by keeping highest IoU GT.
    Returns (target_gt_idx, fg_mask, mask_pos).
    """
    fg_mask = mask_pos.sum(-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)
        max_overlaps_idx = overlaps.argmax(1)
        is_max_overlaps = torch.zeros(
            mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device
        )
        is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()
        fg_mask = mask_pos.sum(-2)
    target_gt_idx = mask_pos.argmax(-2)
    return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner(nn.Module):
    """
    Task-Aligned One-Stage Object Detection label assigner (TOOD / YOLOv8 / YOLOv11 / YOLOv26).
    """

    def __init__(
        self,
        topk: int = 13,
        num_classes: int = 80,
        alpha: float = 1.0,
        beta: float = 6.0,
        eps: float = 1e-9,
        topk2: Optional[int] = None,
    ):
        super().__init__()
        self.topk = topk
        self.topk2 = topk2
        self.num_classes = num_classes
        self.bg_idx = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx, dtype=torch.long),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.bool),
                torch.zeros_like(pd_scores[..., 0], dtype=torch.long),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        # Soft score normalization
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (
            (align_metric * pos_overlaps / (pos_align_metrics + self.eps))
            .amax(-2)
            .unsqueeze(-1)
        )
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        is_in_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool()
        )
        if self.topk2 is not None:
            is_in_topk = is_in_topk & self.select_topk_candidates(
                align_metric,
                topk=self.topk2,
                topk_mask=mask_gt.expand(-1, -1, self.topk2).bool(),
            )
        mask_pos = is_in_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(
        self,
        pd_scores: torch.Tensor,
        pd_bboxes: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros(
            (self.bs, self.n_max_boxes, na), dtype=pd_bboxes.dtype, device=pd_bboxes.device
        )
        bbox_scores = torch.zeros(
            (self.bs, self.n_max_boxes, na), dtype=pd_scores.dtype, device=pd_scores.device
        )

        ind = torch.zeros((2, self.bs, self.n_max_boxes), dtype=torch.long, device=pd_scores.device)
        ind[0] = torch.arange(end=self.bs, device=pd_scores.device).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1).long().clamp(0, self.num_classes - 1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    @staticmethod
    def iou_calculation(gt_bboxes: torch.Tensor, pd_bboxes: torch.Tensor) -> torch.Tensor:
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(
        self,
        metrics: torch.Tensor,
        largest: bool = True,
        topk: Optional[int] = None,
        topk_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k = topk if topk is not None else self.topk
        topk_metrics, topk_idxs = torch.topk(metrics, k, dim=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        topk_idxs.masked_fill_(~topk_mask, 0)

        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for i in range(k):
            count_tensor.scatter_add_(-1, topk_idxs[:, :, i : i + 1], ones)
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype).bool()

    def get_targets(
        self,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        target_gt_idx: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_ind = torch.arange(
            end=self.bs, dtype=torch.int64, device=gt_labels.device
        )[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]

        target_labels.clamp_(0)
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)
        return target_labels, target_bboxes, target_scores


class DFLoss(nn.Module):
    """Distribution Focal Loss (DFL) for integral bounding box distribution learning."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
            F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
            + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """Bounding Box CIoU + DFL (YOLOv11) / normalized L1 (YOLOv26 `reg_max=1`) loss."""

    def __init__(self, reg_max: int = 16, iou_type: str = "ciou"):
        super().__init__()
        self.reg_max = reg_max
        self.iou_type = iou_type.lower()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_iou(
            pred_bboxes[fg_mask],
            target_bboxes[fg_mask],
            xywh=False,
            CIoU=(self.iou_type == "ciou"),
            DIoU=(self.iou_type == "diou"),
            GIoU=(self.iou_type == "giou"),
            EIoU=(self.iou_type == "eiou"),
            SIoU=(self.iou_type == "siou"),
        )
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss is not None:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                    target_ltrb[fg_mask],
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            # DFL-free normalized L1 loss (YOLOv26 `reg_max == 1`)
            target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_norm = pred_dist * stride
            pred_dist_norm[..., 0::2] /= imgsz[1]
            pred_dist_norm[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(
                    pred_dist_norm[fg_mask], target_ltrb[fg_mask], reduction="none"
                ).mean(-1, keepdim=True)
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


# ==================================================================================================
# 4. YOLOv11 Loss (`v8DetectionLoss`) & YOLOv26 End-to-End Loss (`E2ELoss`)
# ==================================================================================================

class YOLOv11DetectionLoss(nn.Module):
    """
    Ultralytics YOLOv8/YOLOv11 Detection Loss (`v8DetectionLoss`).
    Uses TaskAlignedAssigner + CIoU + DFL (or L1 when `reg_max == 1`) + BCE.
    """

    loss_names: Tuple[str, str, str] = ("box", "dfl", "cls")

    def __init__(
        self,
        model: nn.Module,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        tal_topk: int = 10,
        tal_topk2: Optional[int] = None,
        iou_type: str = "ciou",
    ):
        super().__init__()
        device = next(model.parameters()).device
        m = _get_detect_head(model)

        self.hyp = _resolve_v11_hyp(hyp)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.stride = m.stride.to(device)
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.device = device
        self.use_dfl = m.reg_max > 1
        if not self.use_dfl:
            self.loss_names = ("box", "l1", "cls")

        self.assigner = TaskAlignedAssigner(
            topk=tal_topk,
            num_classes=self.nc,
            alpha=0.5,
            beta=6.0,
            topk2=tal_topk2,
        )
        self.bbox_loss = BboxLoss(m.reg_max, iou_type=iou_type).to(device)
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(
        self,
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch_size: int,
        scale_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Convert either `[N, 6]` tensor or Ultralytics batch dict into `(B, max_gt, 5)` `[cls, x1, y1, x2, y2]`."""
        if isinstance(targets, dict):
            batch_idx = targets["batch_idx"].view(-1, 1)
            cls = targets["cls"].view(-1, 1)
            bboxes = targets["bboxes"].view(-1, 4)
            targets_t = torch.cat((batch_idx, cls, bboxes), dim=1).to(self.device)
        else:
            targets_t = targets

        nl, ne = targets_t.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, 5, device=self.device)

        i = targets_t[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, int(counts.max().item()), 5, device=self.device)
        for j in range(batch_size):
            matches = i == j
            if n := int(matches.sum().item()):
                out[j, :n] = targets_t[matches, 1:]
        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor) -> torch.Tensor:
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = (
                pred_dist.view(b, a, 4, c // 4)
                .softmax(3)
                .matmul(self.proj.to(pred_dist.device).type(pred_dist.dtype))
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def forward(
        self,
        preds: Union[Dict[str, Any], Tuple[Any, Dict[str, Any]]],
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(preds, tuple):
            preds = preds[1]

        feats = preds["feats"]
        pred_distri = preds["boxes"].permute(0, 2, 1).contiguous().float()
        pred_scores = preds["scores"].permute(0, 2, 1).contiguous().float()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        device = pred_scores.device
        stride = self.stride.to(device)
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * stride[0]
        anchor_points, stride_tensor = make_anchors(feats, stride, 0.5)

        targets = self.preprocess(batch, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        loss = torch.zeros(3, device=device)

        # Cls loss
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # Box & DFL/L1 loss
        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )

        loss[0] *= self.hyp["box"]
        loss[1] *= self.hyp["cls"]
        loss[2] *= self.hyp["dfl"]

        # Return (box, dfl_or_l1, cls) ordering so train.py logs [box, aux, cls] uniformly
        return loss.sum() * batch_size, torch.stack((loss[0], loss[2], loss[1])).detach()


v8DetectionLoss = YOLOv11DetectionLoss


class YOLOv26DetectionLoss(nn.Module):
    """
    Ultralytics YOLOv26 End-to-End Detection Loss (`E2ELoss` / `E2EDetectLoss`).
    Combines one-to-many auxiliary branch (`tal_topk=10`) with one-to-one NMS-free branch
    (`tal_topk=7, tal_topk2=1`), linearly annealing `o2m` from 0.8 -> 0.1 via `update()`.
    """

    loss_names: Tuple[str, str, str] = ("box", "l1", "cls")

    def __init__(
        self,
        model: nn.Module,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        epochs: int = 100,
        iou_type: str = "ciou",
    ):
        super().__init__()
        if hyp is not None and not isinstance(hyp, dict) and getattr(hyp, "nEpochs", None):
            epochs = int(hyp.nEpochs)

        self.one2many = YOLOv11DetectionLoss(model, hyp=hyp, tal_topk=10, iou_type=iou_type)
        self.one2one = YOLOv11DetectionLoss(
            model, hyp=hyp, tal_topk=7, tal_topk2=1, iou_type=iou_type
        )
        self.hyp = self.one2many.hyp
        self.loss_names = self.one2many.loss_names

        self.updates = 0
        self.initial_o2m = 0.8
        self.final_o2m = 0.1
        self.total_updates = max(epochs - 1, 1)
        self.o2m = self.initial_o2m
        self.o2o = 1.0 - self.o2m

    def update(self) -> None:
        """Advance decay schedule once per epoch (`o2m`: 0.8 -> 0.1, `o2o`: 0.2 -> 0.9)."""
        self.updates += 1
        progress = min(self.updates / self.total_updates, 1.0)
        self.o2m = self.initial_o2m - (self.initial_o2m - self.final_o2m) * progress
        self.o2o = 1.0 - self.o2m

    def sync_updates(self, completed_epochs: int) -> None:
        """Synchronize decay schedule when resuming from an existing epoch."""
        self.updates = max(int(completed_epochs), 0)
        progress = min(self.updates / self.total_updates, 1.0)
        self.o2m = self.initial_o2m - (self.initial_o2m - self.final_o2m) * progress
        self.o2o = 1.0 - self.o2m

    def forward(
        self,
        preds: Union[Dict[str, Any], Tuple[Any, Dict[str, Any]]],
        batch: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(preds, tuple):
            preds = preds[1]
        if "one2many" in preds and "one2one" in preds:
            loss_o2m, items_o2m = self.one2many(preds["one2many"], batch)
            loss_o2o, items_o2o = self.one2one(preds["one2one"], batch)
            total_loss = self.o2m * loss_o2m + self.o2o * loss_o2o
            total_items = self.o2m * items_o2m + self.o2o * items_o2o
            return total_loss, total_items
        return self.one2many(preds, batch)


E2ELoss = YOLOv26DetectionLoss
E2EDetectLoss = YOLOv26DetectionLoss


# ==================================================================================================
# 5. Unified ComputeLoss Dispatcher
# ==================================================================================================

class ComputeLoss(nn.Module):
    """
    Unified loss dispatcher for YOLOv5, YOLOv11, and YOLOv26.
    Automatically detects the Detect head type on `model` and instantiates:
      - `YOLOv5DetectionLoss`  (anchor-based YOLOv5 Detect head)
      - `YOLOv26DetectionLoss` (dual-branch end2end Detect head)
      - `YOLOv11DetectionLoss` (anchor-free decoupled Detect head)
    """

    def __init__(
        self,
        model: nn.Module,
        autobalance: bool = False,
        hyp: Optional[Union[Dict[str, Any], Any]] = None,
        iou_type: str = "ciou",
    ):
        super().__init__()
        head = _get_detect_head(model)
        if hasattr(head, "anchors") and getattr(head, "na", None) is not None and not hasattr(head, "cv2"):
            self.family = "yolov5"
            self.loss_fn = YOLOv5DetectionLoss(
                model, autobalance=autobalance, hyp=hyp, iou_type=iou_type
            )
        elif getattr(head, "one2one_cv2", None) is not None:
            self.family = "yolov26"
            epochs = int(getattr(hyp, "nEpochs", 100)) if hyp is not None and not isinstance(hyp, dict) else 100
            self.loss_fn = YOLOv26DetectionLoss(
                model, hyp=hyp, epochs=epochs, iou_type=iou_type
            )
        else:
            self.family = "yolov11"
            self.loss_fn = YOLOv11DetectionLoss(model, hyp=hyp, iou_type=iou_type)

        self.hyp = self.loss_fn.hyp
        self.loss_names = self.loss_fn.loss_names

    def update(self) -> None:
        if hasattr(self.loss_fn, "update"):
            self.loss_fn.update()

    def sync_updates(self, completed_epochs: int) -> None:
        if hasattr(self.loss_fn, "sync_updates"):
            self.loss_fn.sync_updates(completed_epochs)

    @property
    def updates(self) -> int:
        return getattr(self.loss_fn, "updates", 0)

    @updates.setter
    def updates(self, value: int) -> None:
        if hasattr(self.loss_fn, "sync_updates"):
            self.loss_fn.sync_updates(value)

    def forward(
        self,
        p: Any,
        targets: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.loss_fn(p, targets)
