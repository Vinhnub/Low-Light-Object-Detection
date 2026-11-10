import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import warnings
warnings.filterwarnings("ignore", message=".*Grad strides do not match bucket view strides.*")

import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.data import *
from data.options import option
from data.scheduler import *
from data.util import DARKFACE_CLASSES, EXDARK_CLASSES, xywhn2xyxy
from loss.losses import *
from measure import (
    VOC_CLASSES,
    ObjectDetectionEvaluator,
    print_academic_report,
)
from net.Yolov5 import (
    DetectionModel,
    YOLOv5,
    non_max_suppression,
    scale_boxes,
    yolov5l,
    yolov5m,
    yolov5n,
    yolov5s,
    yolov5x,
)

opt = option().parse_args()


def is_master() -> bool:
    """Kiểm tra xem process hiện tại có phải là rank 0 (master) không."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_model_module(m: torch.nn.Module) -> torch.nn.Module:
    """Lấy module gốc khi model được bọc bởi DDP hoặc DataParallel."""
    if isinstance(m, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
        return m.module
    return m


def seed_torch():
    # Thêm rank vào seed để mỗi GPU có chuỗi ngẫu nhiên riêng cho augmentation nhưng có thể tái lập
    seed = random.randint(1, 1000000) if is_master() else 0
    if dist.is_initialized():
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        seed_tensor = torch.tensor([seed], dtype=torch.long, device=device)
        dist.broadcast(seed_tensor, src=0)
        seed = int(seed_tensor.item()) + get_rank()
    else:
        seed = 42 + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_init() -> int:
    # Khởi tạo tiến trình phân tán (DDP)
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # Trên Windows (os.name == 'nt'), NCCL không được hỗ trợ chính thức nên tự động chuyển sang 'gloo'
        backend = "gloo" if os.name == "nt" else "nccl"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    seed_torch()
    cudnn.benchmark = False
    cudnn.deterministic = True
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode False")
    return local_rank


def get_class_names():
    ds = opt.dataset.lower()
    if ds == "exdark":
        return EXDARK_CLASSES
    elif ds == "darkface":
        return DARKFACE_CLASSES
    elif ds == "voc":
        return VOC_CLASSES
    else:
        return [str(i) for i in range(opt.num_classes)]


def save_training_visual_sample(img_tensor, targets_tensor, class_names, save_dir):
    """
    Lưu ảnh trực quan hóa Ground Truth (gt.png) và Dự đoán (test.png) trên tiến trình master (rank 0).
    """
    if not is_master():
        return
    os.makedirs(save_dir, exist_ok=True)
    img_np = (
        (img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)
        .clip(0, 255)
        .astype(np.uint8)
    )
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # 1. Vẽ Ground Truth boxes lên gt.png
    gt_img = img_bgr.copy()
    gt_b0 = targets_tensor[targets_tensor[:, 0] == 0]
    if len(gt_b0) > 0:
        gt_np = gt_b0[:, 1:].detach().cpu().numpy()
        boxes_xyxy = xywhn2xyxy(gt_np[:, 1:5], w=w, h=h)
        for cls_id, (x1, y1, x2, y2) in zip(gt_np[:, 0].astype(int), boxes_xyxy):
            cname = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(gt_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(
                gt_img,
                cname,
                (int(x1), max(15, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
    cv2.imwrite(os.path.join(save_dir, "gt.png"), gt_img)

    # 2. Vẽ Predicted boxes lên test.png bằng module gốc (unwrapped DDP)
    pred_img = img_bgr.copy()
    raw_model = get_model_module(model)
    was_training = raw_model.training
    raw_model.eval()
    with torch.no_grad():
        pred_out, _ = raw_model(img_tensor[0:1])
        dets = non_max_suppression(pred_out, conf_thres=0.25, iou_thres=0.45)[0]
        for *xyxy, conf, cls_id in dets.cpu().numpy():
            x1, y1, x2, y2 = xyxy
            cid = int(cls_id)
            cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            label_str = f"{cname} {conf:.2f}"
            cv2.rectangle(pred_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(
                pred_img,
                label_str,
                (int(x1), max(15, int(y1) - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
    if was_training:
        raw_model.train()
    cv2.imwrite(os.path.join(save_dir, "test.png"), pred_img)


def train(epoch: int, local_rank: int):
    model.train()
    loss_print = 0.0
    pic_cnt = 0
    loss_last_10 = 0.0
    pic_last_10 = 0

    lbox_sum = 0.0
    lobj_sum = 0.0
    lcls_sum = 0.0

    train_len = len(training_data_loader)
    iter_cnt = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    # Thiết lập sampler cho epoch hiện tại trong DDP để shuffle dữ liệu đúng cách giữa các GPU
    if (
        dist.is_initialized()
        and hasattr(training_data_loader, "sampler")
        and isinstance(training_data_loader.sampler, DistributedSampler)
    ):
        training_data_loader.sampler.set_epoch(epoch)

    # Chỉ hiển thị thanh progress bar tqdm trên tiến trình chính (rank 0)
    data_iterator = (
        tqdm(
            training_data_loader,
            desc=f"Epoch [{epoch}/{opt.nEpochs + opt.start_epoch}] (DDP x{get_world_size()})",
        )
        if is_master()
        else training_data_loader
    )

    for batch in data_iterator:
        imgs, targets, paths, shapes = batch[0], batch[1], batch[2], batch[3]
        if opt.gpu_mode:
            imgs = imgs.cuda(local_rank, non_blocking=True)
            targets = targets.cuda(local_rank, non_blocking=True)

        # Random gamma illumination curve
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            imgs = imgs.clamp(1e-6, 1.0) ** gamma

        optimizer.zero_grad()

        if opt.amp and opt.gpu_mode:
            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)
                # Nhân với world_size để bù lại việc DDP lấy trung bình gradient giữa các GPU
                loss = loss * get_world_size()
            scaler.scale(loss).backward()
            if opt.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss, loss_items = criterion(preds, targets)
            loss = loss * get_world_size()
            loss.backward()
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            optimizer.step()

        lbox_val = loss_items[0].item()
        lobj_val = loss_items[1].item()
        lcls_val = loss_items[2].item()
        total_item_loss = lbox_val + lobj_val + lcls_val

        lbox_sum += lbox_val
        lobj_sum += lobj_val
        lcls_sum += lcls_val

        loss_print += total_item_loss
        loss_last_10 += total_item_loss
        iter_cnt += 1
        pic_cnt += 1
        pic_last_10 += 1

        if is_master() and hasattr(data_iterator, "set_postfix"):
            data_iterator.set_postfix(
                {
                    "Total": f"{loss_print / pic_cnt:.4f}",
                    "Box": f"{lbox_sum / pic_cnt:.4f}",
                    "Obj": f"{lobj_sum / pic_cnt:.4f}",
                    "Cls": f"{lcls_sum / pic_cnt:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        if iter_cnt == train_len and is_master():
            print(
                "===> Epoch[{}]: Total Loss: {:.4f} || Box: {:.4f} | Obj: {:.4f} | Cls: {:.4f} || lr={:.8f}.".format(
                    epoch,
                    loss_last_10 / max(1, pic_last_10),
                    lbox_sum / max(1, pic_cnt),
                    lobj_sum / max(1, pic_cnt),
                    lcls_sum / max(1, pic_cnt),
                    optimizer.param_groups[0]["lr"],
                )
            )
            loss_last_10 = 0.0
            pic_last_10 = 0
            try:
                save_training_visual_sample(
                    imgs,
                    targets,
                    class_names,
                    os.path.join(opt.val_folder, "training"),
                )
            except Exception as e:
                print(f"[Warn] Could not save training preview image: {e}")

    detailed_losses = {
        "total": loss_print / pic_cnt if pic_cnt > 0 else 0.0,
        "box": lbox_sum / pic_cnt if pic_cnt > 0 else 0.0,
        "obj": lobj_sum / pic_cnt if pic_cnt > 0 else 0.0,
        "cls": lcls_sum / pic_cnt if pic_cnt > 0 else 0.0,
    }
    return loss_print, pic_cnt, detailed_losses


def evaluate_detection(model_eval, data_loader, class_names, pred_save_dir=None):
    """
    Chạy đánh giá trên tiến trình master (rank 0) dùng module gốc `get_model_module(model)`.
    Tính toán mAP@0.50, mAP@0.75, mAP@0.50:0.95 qua `ObjectDetectionEvaluator` từ `measure.py`.
    """
    model_eval.eval()
    evaluator = ObjectDetectionEvaluator(class_names=class_names, method="coco")

    if pred_save_dir:
        os.makedirs(pred_save_dir, exist_ok=True)

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating Validation Set (Rank 0)"):
            imgs, targets, paths, shapes = batch[0], batch[1], batch[2], batch[3]
            if opt.gpu_mode:
                imgs = imgs.cuda(non_blocking=True)

            pred_out, _ = model_eval(imgs)
            dets_batch = non_max_suppression(
                pred_out,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                max_det=opt.max_det,
            )

            for si, det in enumerate(dets_batch):
                img_path = paths[si]
                stem = Path(img_path).stem
                h0, w0 = shapes[si][0]
                ratio_pad = shapes[si][1]

                t_si = targets[targets[:, 0] == si]
                if len(t_si):
                    t_np = t_si[:, 1:].cpu().numpy()
                    t_xyxy = xywhn2xyxy(
                        t_np[:, 1:5],
                        w=imgs.shape[3],
                        h=imgs.shape[2],
                    )
                    t_xyxy_t = scale_boxes(
                        imgs.shape[2:],
                        torch.from_numpy(t_xyxy),
                        (h0, w0),
                        ratio_pad=ratio_pad,
                    ).numpy()
                    for cid, box in zip(t_np[:, 0].astype(int), t_xyxy_t):
                        evaluator.add_ground_truth(stem, int(cid), box.tolist())
                else:
                    evaluator.image_ids.add(stem)

                pred_lines = []
                if len(det):
                    det_clone = det.clone()
                    det_clone[:, :4] = scale_boxes(
                        imgs.shape[2:],
                        det_clone[:, :4],
                        (h0, w0),
                        ratio_pad=ratio_pad,
                    )
                    for *xyxy, conf, cls_id in det_clone.cpu().numpy():
                        cid = int(cls_id)
                        score = float(conf)
                        box = [float(v) for v in xyxy]
                        evaluator.add_prediction(stem, cid, score, box)
                        cname = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
                        pred_lines.append(
                            f"{cname} {score:.6f} {box[0]:.2f} {box[1]:.2f} {box[2]:.2f} {box[3]:.2f}\n"
                        )

                if pred_save_dir:
                    pred_txt_path = os.path.join(pred_save_dir, f"{stem}.txt")
                    with open(pred_txt_path, "w", encoding="utf-8") as pf:
                        pf.writelines(pred_lines)

    results = evaluator.evaluate(conf_threshold=opt.conf_thres)
    print_academic_report(results, class_names)
    return results


def checkpoint(epoch: int, lr=None, tag=None):
    filename = f"epoch_{epoch}.pth" if tag is None else f"epoch_{epoch}_{tag}.pth"
    model_out_path = os.path.join(opt.save_folder, filename)
    if not is_master():
        return model_out_path

    os.makedirs("./weights", exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    current_lr = lr if lr is not None else optimizer.param_groups[0]["lr"]
    checkpoint_dict = {
        "epoch": epoch,
        "model_name": opt.model,
        "num_classes": len(class_names),
        "state_dict": get_model_module(model).state_dict(),
        "lr": current_lr,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint_dict, model_out_path)
    print("Checkpoint saved to {} (lr: {:.8f})".format(model_out_path, current_lr))
    return model_out_path


def load_datasets():
    if is_master():
        print(f"===> Loading datasets: {opt.dataset}")

    if opt.dataset == "exdark":
        train_set = get_exdark_training_set(
            opt.data_train_exdark, size=opt.imgSize, hyp=opt
        )
        test_set = get_exdark_val_set(opt.data_val_exdark, size=opt.imgSize)
    elif opt.dataset == "darkface":
        train_set = get_darkface_training_set(
            opt.data_train_darkface, size=opt.imgSize, hyp=opt
        )
        test_set = get_darkface_val_set(opt.data_val_darkface, size=opt.imgSize)
    elif opt.dataset == "voc":
        train_set = get_training_set(
            opt.data_train_voc, size=opt.imgSize, dataset_name="voc", hyp=opt
        )
        test_set = get_eval_set(
            opt.data_val_voc, size=opt.imgSize, dataset_name="voc"
        )
    elif opt.dataset == "custom":
        train_set = get_training_set(
            opt.data_train_custom, size=opt.imgSize, dataset_name="custom", hyp=opt
        )
        test_set = get_eval_set(
            opt.data_val_custom, size=opt.imgSize, dataset_name="custom"
        )
    else:
        raise Exception("should choose a supported dataset (exdark, darkface, voc, custom)")

    pin_memory = opt.gpu_mode and torch.cuda.is_available()
    use_persistent = opt.threads > 0
    prefetch = 4 if opt.threads > 0 else None

    # Khi dùng DDP, sử dụng DistributedSampler để chia đều dataset cho các GPU
    if dist.is_initialized():
        train_sampler = DistributedSampler(train_set, shuffle=opt.shuffle)
        training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            sampler=train_sampler,
            shuffle=False,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )
    else:
        training_data_loader = DataLoader(
            dataset=train_set,
            num_workers=opt.threads,
            batch_size=opt.batchSize,
            shuffle=opt.shuffle,
            pin_memory=pin_memory,
            persistent_workers=use_persistent,
            prefetch_factor=prefetch,
            collate_fn=collate_fn,
        )

    testing_data_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=use_persistent,
        prefetch_factor=prefetch,
        collate_fn=collate_fn,
    )
    if is_master():
        print(
            f"===> Train samples: {len(train_set)} | Val samples: {len(test_set)} | World size: {get_world_size()}"
        )
    return training_data_loader, testing_data_loader


def build_model(local_rank: int, num_classes: int):
    if is_master():
        print(
            f"===> Building model: {opt.model} (num_classes={num_classes}, modular={opt.modular})"
        )
    model_builders = {
        "yolov5n": yolov5n,
        "yolov5s": yolov5s,
        "yolov5m": yolov5m,
        "yolov5l": yolov5l,
        "yolov5x": yolov5x,
    }
    if opt.cfg and os.path.exists(opt.cfg):
        net = DetectionModel(cfg=opt.cfg, ch=3, nc=num_classes)
    else:
        builder = model_builders.get(opt.model.lower(), yolov5s)
        net = builder(num_classes=num_classes, in_channels=3, modular=opt.modular)

    if opt.gpu_mode:
        net = net.cuda(local_rank)

    checkpoint_data = None
    ckpt_path = ""
    if opt.resume and os.path.exists(opt.resume):
        ckpt_path = opt.resume
    elif opt.start_epoch > 0:
        ckpt_path = os.path.join(opt.save_folder, f"epoch_{opt.start_epoch}.pth")
        if not os.path.exists(ckpt_path):
            alt_path = f"./weights/train/epoch_{opt.start_epoch}.pth"
            if os.path.exists(alt_path):
                ckpt_path = alt_path
            elif is_master():
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
    elif opt.pretrained and os.path.exists(opt.pretrained):
        ckpt_path = opt.pretrained

    if ckpt_path and os.path.exists(ckpt_path):
        checkpoint_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_state_dict = checkpoint_data
        if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data:
            raw_state_dict = checkpoint_data["state_dict"]
        elif isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
            raw_state_dict = (
                checkpoint_data["model"].state_dict()
                if hasattr(checkpoint_data["model"], "state_dict")
                else checkpoint_data["model"]
            )

        model_dict = net.state_dict()
        filtered_state_dict = {}
        for k, v in raw_state_dict.items():
            name = k[7:] if k.startswith("module.") else k
            if name in model_dict and model_dict[name].shape == v.shape:
                filtered_state_dict[name] = v
        net.load_state_dict(filtered_state_dict, strict=False)
        if is_master():
            print(
                f"Loaded {len(filtered_state_dict)}/{len(model_dict)} weight tensors from {ckpt_path}"
            )

    # Đồng bộ BatchNorm và bọc model bằng DistributedDataParallel khi chạy đa GPU
    if dist.is_initialized():
        if get_world_size() > 1:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net).to(local_rank)
        if is_master():
            print(f"===> Sử dụng DistributedDataParallel (DDP) trên {get_world_size()} GPUs!")
        net = DDP(
            net,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )

    return net, checkpoint_data


def make_scheduler(lr=None):
    train_lr = lr if lr is not None else opt.lr
    opt_type = opt.optimizer.upper()
    if opt_type == "ADAM":
        optimizer = optim.Adam(
            model.parameters(), lr=train_lr, weight_decay=opt.weight_decay
        )
    elif opt_type == "ADAMW":
        optimizer = optim.AdamW(
            model.parameters(), lr=train_lr, weight_decay=opt.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_lr,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
            nesterov=True,
        )

    if opt.cos_restart_cyclic:
        if opt.start_warmup and opt.start_epoch == 0:
            scheduler_step = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[
                    max(1, (opt.nEpochs // 4) - opt.warmup_epochs),
                    max(1, (opt.nEpochs * 3) // 4),
                ],
                restart_weights=[1, 1],
                eta_mins=[train_lr * 0.1, train_lr * 1e-4],
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
            )
        else:
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[max(1, opt.nEpochs // 4), max(1, (opt.nEpochs * 3) // 4)],
                restart_weights=[1, 1],
                eta_mins=[train_lr * 0.1, train_lr * 1e-4],
            )
    elif opt.cos_restart:
        if opt.start_warmup and opt.start_epoch == 0:
            scheduler_step = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[max(1, opt.nEpochs - opt.warmup_epochs)],
                restart_weights=[1],
                eta_min=train_lr * opt.lrf,
            )
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=opt.warmup_epochs,
                after_scheduler=scheduler_step,
            )
        else:
            remaining_epochs = max(1, opt.nEpochs - opt.start_epoch)
            scheduler = CosineAnnealingRestartLR(
                optimizer=optimizer,
                periods=[remaining_epochs],
                restart_weights=[1],
                eta_min=train_lr * opt.lrf,
            )
    else:
        raise Exception("should choose a scheduler (--cos_restart or --cos_restart_cyclic)")

    return optimizer, scheduler


def init_loss():
    criterion_fn = ComputeLoss(model, hyp=opt, iou_type=opt.iou_type)
    return criterion_fn


if __name__ == "__main__":
    local_rank = train_init()
    class_names = get_class_names()
    opt.num_classes = len(class_names)

    training_data_loader, testing_data_loader = load_datasets()
    model, checkpoint_data = build_model(local_rank, num_classes=len(class_names))

    # Xác định learning rate khi bắt đầu / tiếp tục train
    resume_lr = None
    if opt.start_epoch > 0 or opt.resume:
        if checkpoint_data is not None and isinstance(checkpoint_data, dict):
            resume_lr = checkpoint_data.get("lr", None)
            if opt.start_epoch == 0 and "epoch" in checkpoint_data:
                opt.start_epoch = int(checkpoint_data["epoch"])

        if resume_lr is not None:
            if is_master():
                print(
                    f"===> [Resume] Đã tải learning rate từ checkpoint epoch_{opt.start_epoch}: {resume_lr:.8f}"
                )
        else:
            resume_lr = opt.lr
            if is_master():
                print(
                    f"===> [Resume] Không tìm thấy lr trong checkpoint. Tự động sử dụng lr mặc định: {opt.lr}"
                )
    else:
        if is_master():
            print(f"===> Khởi đầu huấn luyện từ epoch 1 với lr: {opt.lr}")

    optimizer, scheduler = make_scheduler(lr=resume_lr)

    if (
        checkpoint_data is not None
        and isinstance(checkpoint_data, dict)
        and "optimizer" in checkpoint_data
        and (opt.start_epoch > 0 or opt.resume)
    ):
        try:
            optimizer.load_state_dict(checkpoint_data["optimizer"])
            target_lr = resume_lr if resume_lr is not None else opt.lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = target_lr
            if is_master():
                print("===> Đã khôi phục trạng thái optimizer từ checkpoint")
        except Exception as e:
            if is_master():
                print(f"===> Không thể nạp trạng thái optimizer ({e}), sử dụng optimizer mới")

    criterion = init_loss()
    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.gpu_mode))

    map50_history = []
    map75_history = []
    map50_95_history = []
    best_map50 = 0.0
    best_map50_95 = 0.0

    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 0
    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    metrics_md_path = f"./results/training/metrics{now}.md"

    if is_master():
        os.makedirs(opt.val_folder, exist_ok=True)
        os.makedirs("./results/training", exist_ok=True)
        os.makedirs(opt.save_folder, exist_ok=True)

        with open(metrics_md_path, "w", encoding="utf-8") as f:
            f.write(f"dataset: {opt.dataset}\n")
            f.write(f"model: {opt.model} (modular={opt.modular}, world_size={get_world_size()})\n")
            f.write(f"num_classes: {len(class_names)}\n")
            f.write(f"optimizer: {opt.optimizer}\n")
            f.write(f"lr: {optimizer.param_groups[0]['lr']}\n")
            f.write(f"batch size (per GPU): {opt.batchSize}\n")
            f.write(f"img size: {opt.imgSize}\n")
            f.write(f"iou_type: {opt.iou_type}\n")
            f.write(f"box_weight: {opt.box_weight}\n")
            f.write(f"obj_weight: {opt.obj_weight}\n")
            f.write(f"cls_weight: {opt.cls_weight}\n")
            f.write(f"mosaic: {opt.mosaic} | gamma: {opt.gamma}\n\n")
            f.write(
                "| Epochs | Total Loss | Box | Obj | Cls | Precision | Recall | mAP@0.50 | mAP@0.75 | mAP@50:95 | lr | Note |\n"
            )
            f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    for epoch in range(start_epoch + 1, opt.nEpochs + start_epoch + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_loss, pic_num, detailed_losses = train(epoch, local_rank)
        scheduler.step()

        if epoch % opt.snapshots == 0 or epoch == (opt.nEpochs + start_epoch):
            model_out_path = checkpoint(epoch, lr=current_lr)

            # Chỉ rank 0 thực hiện đánh giá (eval) và ghi kết quả
            if is_master():
                output_folder = f"{opt.dataset}_preds/"
                pred_save_dir = os.path.join(opt.val_folder, output_folder)

                results = evaluate_detection(
                    get_model_module(model),
                    testing_data_loader,
                    class_names=class_names,
                    pred_save_dir=pred_save_dir,
                )

                cur_map50 = results["mAP50"]
                cur_map75 = results["mAP75"]
                cur_map50_95 = results["mAP50_95"]
                cur_p = results["mean_precision"]
                cur_r = results["mean_recall"]

                print("===> Avg.Precision@0.50: {:.4f}".format(cur_p))
                print("===> Avg.Recall@0.50:    {:.4f}".format(cur_r))
                print("===> mAP@0.50:           {:.4f} ({:.2f}%)".format(cur_map50, cur_map50 * 100))
                print("===> mAP@0.75:           {:.4f} ({:.2f}%)".format(cur_map75, cur_map75 * 100))
                print(
                    "===> mAP@0.50:0.95:      {:.4f} ({:.2f}%)".format(
                        cur_map50_95, cur_map50_95 * 100
                    )
                )

                map50_history.append(cur_map50)
                map75_history.append(cur_map75)
                map50_95_history.append(cur_map50_95)

                note_list = []
                if cur_map50 > best_map50:
                    best_map50 = cur_map50
                    checkpoint(epoch, lr=current_lr, tag="best_mAP50")
                    note_list.append("Best mAP@0.50")
                if cur_map50_95 > best_map50_95:
                    best_map50_95 = cur_map50_95
                    checkpoint(epoch, lr=current_lr, tag="best_mAP50_95")
                    note_list.append("Best mAP@50:95")
                note_str = ", ".join(note_list) if note_list else "-"

                with open(metrics_md_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"| {epoch} | {detailed_losses['total']:.4f} | {detailed_losses['box']:.4f} | "
                        f"{detailed_losses['obj']:.4f} | {detailed_losses['cls']:.4f} | "
                        f"{cur_p:.4f} | {cur_r:.4f} | {cur_map50:.4f} | {cur_map75:.4f} | "
                        f"{cur_map50_95:.4f} | {current_lr:.8f} | {note_str} |\n"
                    )

        # Đảm bảo tất cả các GPU đợi GPU 0 hoàn thành eval xong rồi mới chuyển sang epoch tiếp theo
        if epoch % opt.snapshots == 0 or epoch == (opt.nEpochs + start_epoch):
            if dist.is_initialized():
                dist.barrier()
            if opt.gpu_mode:
                torch.cuda.empty_cache()

    if dist.is_initialized():
        dist.destroy_process_group()
