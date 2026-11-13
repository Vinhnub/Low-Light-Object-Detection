import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data import *
from data.options import option
from data.scheduler import *
from data.util import DARKFACE_CLASSES, EXDARK_CLASSES, xywhn2xyxy
from eval import eval
from loss.losses import *
from measure import VOC_CLASSES, metrics
from net import (
    ModelEMA,
    build_yolo_model,
    get_model_family,
    load_yolo_checkpoint,
    non_max_suppression,
    smart_optimizer,
)

opt = option().parse_args()
if opt.box_weight is None:
    _family = get_model_family(opt.cfg if opt.cfg else opt.model)
    opt.box_weight = 7.5 if _family in {"yolov11", "yolov26"} else 0.05


def seed_torch(seed: int = None):
    if seed is None:
        seed = random.randint(1, 1000000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train_init():
    seed_torch()
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cuda = opt.gpu_mode
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --gpu_mode False")


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


def save_training_visual_sample(img_tensor, targets_tensor, preds_raw, class_names, save_dir):
    """
    Save visual comparison of Ground Truth (gt.png) and Model Predictions (test.png)
    from the last training batch into `results/training/`, matching IGMamba-EdgeLCA-LLIE.
    """
    os.makedirs(save_dir, exist_ok=True)
    # Convert first image in batch from RGB float [0, 1] CHW to BGR uint8 HWC
    img_np = (img_tensor[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    h, w = img_bgr.shape[:2]

    # 1. Draw Ground Truth boxes on gt.png
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

    # 2. Draw Predicted boxes on test.png (run quick eval forward on first image)
    pred_img = img_bgr.copy()
    vis_model = ema.ema if (ema is not None) else model
    was_training = vis_model.training
    vis_model.eval()
    with torch.no_grad():
        pred_out, _ = vis_model(img_tensor[0:1])
        dets = non_max_suppression(
            pred_out,
            conf_thres=0.25,
            iou_thres=0.45,
            end2end=getattr(vis_model, "end2end", False),
        )[0]
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
        vis_model.train()
    cv2.imwrite(os.path.join(save_dir, "test.png"), pred_img)


def train(epoch):
    model.train()
    # Disable mosaic in the final 15 epochs to align training with real un-mosaicked validation images
    if hasattr(training_data_loader, "dataset") and hasattr(training_data_loader.dataset, "mosaic"):
        if epoch > max(1, opt.nEpochs - 15) and training_data_loader.dataset.mosaic:
            print(f"===> [Epoch {epoch}] Closing Mosaic augmentation for final fine-tuning.")
            training_data_loader.dataset.mosaic = False

    loss_print = 0.0
    pic_cnt = 0
    loss_last_10 = 0.0
    pic_last_10 = 0

    # Accumulators for individual YOLOv5 detection losses
    lbox_sum = 0.0
    lobj_sum = 0.0
    lcls_sum = 0.0

    train_len = len(training_data_loader)
    iter_cnt = 0
    torch.autograd.set_detect_anomaly(opt.grad_detect)

    pbar = tqdm(training_data_loader, desc=f"Epoch [{epoch}/{opt.nEpochs}]")
    for batch in pbar:
        imgs, targets, paths, shapes = batch[0], batch[1], batch[2], batch[3]
        if opt.gpu_mode:
            imgs = imgs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        # Optional random gamma illumination curve on tensor (matching IGMamba-EdgeLCA-LLIE)
        if opt.gamma:
            gamma = random.randint(opt.start_gamma, opt.end_gamma) / 100.0
            imgs = imgs.clamp(1e-6, 1.0) ** gamma

        optimizer.zero_grad()

        if opt.amp and opt.gpu_mode:
            with torch.amp.autocast("cuda"):
                preds = model(imgs)
                loss, loss_items = criterion(preds, targets)
            scaler.scale(loss).backward()
            if opt.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(imgs)
            loss, loss_items = criterion(preds, targets)
            loss.backward()
            if opt.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, norm_type=2)
            optimizer.step()

        if ema is not None:
            ema.update(model)

        # loss_items = [lbox, lobj, lcls]
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

        aux_label = getattr(criterion, "loss_names", ("box", "obj", "cls"))[1].capitalize()
        pbar.set_postfix(
            {
                "Total": f"{loss_print / pic_cnt:.4f}",
                "Box": f"{lbox_sum / pic_cnt:.4f}",
                aux_label: f"{lobj_sum / pic_cnt:.4f}",
                "Cls": f"{lcls_sum / pic_cnt:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            }
        )

        if iter_cnt == train_len:
            print(
                "===> Epoch[{}]: Total Loss: {:.4f} || Box: {:.4f} | {}: {:.4f} | Cls: {:.4f} || lr={:.8f}.".format(
                    epoch,
                    loss_last_10 / max(1, pic_last_10),
                    lbox_sum / max(1, pic_cnt),
                    aux_label,
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
                    preds,
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


def checkpoint(epoch, lr=None, tag=None):
    os.makedirs("./weights", exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    filename = f"epoch_{epoch}.pth" if tag is None else f"{tag}.pth"
    model_out_path = os.path.join(opt.save_folder, filename)
    current_lr = lr if lr is not None else optimizer.param_groups[0]["lr"]
    eval_model = ema.ema if (ema is not None) else model
    checkpoint_dict = {
        "epoch": epoch,
        "model_name": opt.model,
        "num_classes": len(class_names),
        "state_dict": eval_model.state_dict(),
        "model_state_dict": model.state_dict(),
        "ema_updates": ema.updates if (ema is not None) else 0,
        "lr": current_lr,
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint_dict, model_out_path)
    print(
        "Checkpoint saved to {} (epoch: {}, lr: {:.8f})".format(
            model_out_path, epoch, current_lr
        )
    )
    return model_out_path


def load_datasets():
    print(f"===> Loading datasets: {opt.dataset}")
    if opt.dataset == "exdark":
        train_set = get_exdark_training_set(
            opt.data_train_exdark, size=opt.imgSize, hyp=opt
        )
        val_set = get_exdark_val_set(opt.data_val_exdark, size=opt.imgSize)
        test_set = (
            get_exdark_test_set(opt.data_test_exdark, size=opt.imgSize)
            if os.path.exists(opt.data_test_exdark)
            else val_set
        )
    elif opt.dataset == "darkface":
        train_set = get_darkface_training_set(
            opt.data_train_darkface, size=opt.imgSize, hyp=opt
        )
        val_set = get_darkface_val_set(opt.data_val_darkface, size=opt.imgSize)
        test_set = val_set
    elif opt.dataset == "voc":
        train_set = get_training_set(
            opt.data_train_voc, size=opt.imgSize, dataset_name="voc", hyp=opt
        )
        val_set = get_eval_set(
            opt.data_val_voc, size=opt.imgSize, dataset_name="voc"
        )
        test_set = val_set
    elif opt.dataset == "custom":
        train_set = get_training_set(
            opt.data_train_custom, size=opt.imgSize, dataset_name="custom", hyp=opt
        )
        val_set = get_eval_set(
            opt.data_val_custom, size=opt.imgSize, dataset_name="custom"
        )
        test_set = val_set
    else:
        raise Exception("should choose a supported dataset (exdark, darkface, voc, custom)")

    pin_memory = opt.gpu_mode and torch.cuda.is_available()
    use_persistent = opt.threads > 0
    prefetch = 4 if opt.threads > 0 else None
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
        dataset=val_set,
        num_workers=0,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    final_test_data_loader = DataLoader(
        dataset=test_set,
        num_workers=0,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    print(
        f"===> Train samples: {len(train_set)} | Val samples: {len(val_set)} | Test samples: {len(test_set)}"
    )
    return training_data_loader, testing_data_loader, final_test_data_loader


def build_model(num_classes: int):
    print(f"===> Building model: {opt.model} (num_classes={num_classes}, modular={opt.modular})")
    cfg_path = opt.cfg if (opt.cfg and os.path.exists(opt.cfg)) else None
    net = build_yolo_model(
        model_name=opt.model,
        num_classes=num_classes,
        in_channels=3,
        cfg=cfg_path,
        modular=opt.modular,
        end2end=opt.end2end,
    )

    if opt.gpu_mode:
        net = net.cuda()

    checkpoint_data = None
    ckpt_path = ""
    is_resume = False
    if opt.resume and os.path.exists(opt.resume):
        ckpt_path = opt.resume
        is_resume = True
    elif opt.start_epoch > 0:
        ckpt_path = os.path.join(opt.save_folder, f"epoch_{opt.start_epoch}.pth")
        if not os.path.exists(ckpt_path):
            alt_path = f"./weights/train/epoch_{opt.start_epoch}.pth"
            if os.path.exists(alt_path):
                ckpt_path = alt_path
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")
        is_resume = True
    elif opt.pretrained and str(opt.pretrained).lower() not in {"none", "false", "no"}:
        ckpt_path = opt.pretrained

    if ckpt_path:
        loaded_ckpt = load_yolo_checkpoint(
            net, ckpt_path, model_name=opt.model, verbose=True
        )
        if is_resume:
            checkpoint_data = loaded_ckpt

    return net, checkpoint_data


def make_scheduler(lr=None):
    train_lr = lr if lr is not None else opt.lr
    optimizer = smart_optimizer(
        model,
        name=opt.optimizer,
        lr=train_lr,
        momentum=opt.momentum,
        decay=opt.weight_decay,
        verbose=True,
    )

    if opt.cos_restart_cyclic:
        if opt.start_warmup and opt.warmup_epochs > 0 and opt.start_epoch == 0:
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
            remaining_epochs = max(1, opt.nEpochs - opt.start_epoch)
            scheduler = CosineAnnealingRestartCyclicLR(
                optimizer=optimizer,
                periods=[
                    max(1, remaining_epochs // 4),
                    max(1, remaining_epochs - (remaining_epochs // 4)),
                ],
                restart_weights=[1, 1],
                eta_mins=[train_lr * 0.1, train_lr * 1e-4],
            )
    elif opt.cos_restart:
        if opt.start_warmup and opt.warmup_epochs > 0 and opt.start_epoch == 0:
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
    """
    1. Preparation
    """
    train_init()
    class_names = get_class_names()
    opt.num_classes = len(class_names)

    training_data_loader, testing_data_loader, final_test_data_loader = load_datasets()
    model, checkpoint_data = build_model(num_classes=len(class_names))

    # Determine learning rate on start / resume
    resume_lr = None
    if opt.start_epoch > 0 or opt.resume:
        if checkpoint_data is not None and isinstance(checkpoint_data, dict):
            resume_lr = checkpoint_data.get("lr", None)
            if opt.start_epoch == 0 and "epoch" in checkpoint_data:
                opt.start_epoch = int(checkpoint_data["epoch"])

        if resume_lr is not None:
            print(
                f"===> [Resume] Đã tải learning rate từ checkpoint epoch_{opt.start_epoch}: {resume_lr:.8f}"
            )
        else:
            resume_lr = opt.lr
            print(
                f"===> [Resume] Không tìm thấy lr trong checkpoint. Tự động sử dụng lr mặc định: {opt.lr}"
            )
    else:
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
            print("===> Đã khôi phục trạng thái optimizer từ checkpoint")
        except Exception as e:
            print(f"===> Không thể nạp trạng thái optimizer ({e}), sử dụng optimizer mới")

    criterion = init_loss()
    scaler = torch.amp.GradScaler("cuda", enabled=(opt.amp and opt.gpu_mode))
    ema = ModelEMA(model) if opt.ema else None
    if (
        ema is not None
        and checkpoint_data is not None
        and isinstance(checkpoint_data, dict)
        and (opt.start_epoch > 0 or opt.resume)
    ):
        if "state_dict" in checkpoint_data:
            ema.ema.load_state_dict(checkpoint_data["state_dict"], strict=False)
        if "model_state_dict" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
        ema.updates = int(checkpoint_data.get("ema_updates", 0))
        print(f"===> Đã khôi phục trạng thái ModelEMA (updates={ema.updates})")

    """
    2. Training & Evaluation Loop
    """
    map50_history = []
    map75_history = []
    map50_95_history = []
    best_precision = 0.0
    best_recall = 0.0
    best_map50 = 0.0
    best_map75 = 0.0
    best_map50_95 = 0.0
    best_model_path = None
    last_model_path = None

    start_epoch = opt.start_epoch if opt.start_epoch > 0 else 0
    if start_epoch > 0 and hasattr(criterion, "sync_updates"):
        criterion.sync_updates(start_epoch)

    os.makedirs(opt.val_folder, exist_ok=True)
    os.makedirs("./results/training", exist_ok=True)
    os.makedirs(opt.save_folder, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    metrics_md_path = f"./results/training/metrics{now}.md"
    aux_header = getattr(criterion, "loss_names", ("box", "obj", "cls"))[1].capitalize()
    init_lr = resume_lr if resume_lr is not None else opt.lr
    with open(metrics_md_path, "w", encoding="utf-8") as f:
        f.write(f"dataset: {opt.dataset}\n")
        f.write(f"model: {opt.model} (modular={opt.modular})\n")
        f.write(f"num_classes: {len(class_names)}\n")
        f.write(f"optimizer: {opt.optimizer}\n")
        f.write(f"lr: {init_lr}\n")
        f.write(f"batch size: {opt.batchSize}\n")
        f.write(f"img size: {opt.imgSize}\n")
        f.write(f"iou_type: {opt.iou_type}\n")
        f.write(f"box_weight: {opt.box_weight}\n")
        f.write(f"obj_weight: {opt.obj_weight} | dfl_weight: {opt.dfl_weight}\n")
        f.write(f"cls_weight: {opt.cls_weight}\n")
        f.write(f"mosaic: {opt.mosaic} | gamma: {opt.gamma}\n\n")
        f.write(
            f"| Epochs | Total Loss | Box | {aux_header} | Cls | Precision | Recall | mAP@0.50 | mAP@0.75 | mAP@50:95 | lr | Note |\n"
        )
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")

    eval_device = "cuda" if opt.gpu_mode else "cpu"
    if opt.dataset == "exdark":
        val_label_dir = opt.data_valgt_exdark
        test_label_dir = (
            opt.data_testgt_exdark
            if os.path.exists(opt.data_testgt_exdark)
            else opt.data_valgt_exdark
        )
    elif opt.dataset == "darkface":
        val_label_dir = opt.data_valgt_darkface
        test_label_dir = opt.data_valgt_darkface
    elif opt.dataset == "voc":
        val_label_dir = opt.data_valgt_voc
        test_label_dir = opt.data_valgt_voc
    else:
        val_label_dir = opt.data_valgt_custom
        test_label_dir = opt.data_valgt_custom

    for epoch in range(start_epoch + 1, opt.nEpochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_loss, pic_num, detailed_losses = train(epoch)
        scheduler.step()
        if hasattr(criterion, "update"):
            criterion.update()

        if epoch % opt.snapshots == 0 or epoch == opt.nEpochs:
            model_out_path = checkpoint(epoch, lr=current_lr)
            last_model_path = model_out_path

            output_folder = f"{opt.dataset}_preds/"
            pred_save_dir = os.path.join(opt.val_folder, output_folder)
            eval_net = ema.ema if (ema is not None) else model

            # 1. Chạy suy luận & lưu dự đoán trong RAM qua eval.py (sử dụng trọng số EMA nếu bật)
            val_preds = eval(
                model=eval_net,
                testing_data_loader=testing_data_loader,
                model_path=None,
                output_folder=pred_save_dir,
                class_names=class_names,
                device=eval_device,
                conf_thres=opt.conf_thres,
                iou_thres=opt.iou_thres,
                max_det=opt.max_det,
                half=(opt.amp and opt.gpu_mode),
                save_txt=False,
                return_preds=True,
                desc="Evaluating Validation Set (EMA)" if (ema is not None) else "Evaluating Validation Set",
            )

            # 2. Tính toán và hiển thị các chỉ số đánh giá qua measure.py
            results = metrics(
                pred_dir=pred_save_dir,
                label_dir=val_label_dir,
                class_names=class_names,
                dataset_type=opt.dataset,
                conf_thres=opt.conf_thres,
                in_memory_preds=val_preds,
            )

            cur_map50 = results["mAP50"]
            cur_map75 = results["mAP75"]
            cur_map50_95 = results["mAP50_95"]
            cur_p = results["mean_precision"]
            cur_r = results["mean_recall"]

            map50_history.append(cur_map50)
            map75_history.append(cur_map75)
            map50_95_history.append(cur_map50_95)

            note_list = []
            if cur_p > best_precision:
                best_precision = cur_p
                checkpoint(epoch, lr=current_lr, tag="best_Precision")
                note_list.append("Best Precision")
            if cur_r > best_recall:
                best_recall = cur_r
                checkpoint(epoch, lr=current_lr, tag="best_Recall")
                note_list.append("Best Recall")
            if cur_map50 > best_map50:
                best_map50 = cur_map50
                best_model_path = checkpoint(epoch, lr=current_lr, tag="best_mAP50")
                note_list.append("Best mAP@0.50")
            if cur_map75 > best_map75:
                best_map75 = cur_map75
                checkpoint(epoch, lr=current_lr, tag="best_mAP75")
                note_list.append("Best mAP@0.75")
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

            if opt.gpu_mode:
                torch.cuda.empty_cache()

    """
    3. Automatic Final Testing on Test Split with Best Checkpoint
    """
    final_ckpt = best_model_path if (best_model_path and os.path.exists(best_model_path)) else last_model_path
    if final_ckpt and os.path.exists(final_ckpt):
        print("\n" + "=" * 95)
        print(f"===> [Final Test] Tự động đánh giá tập TEST với weight tốt nhất: {final_ckpt}")
        print("=" * 95)
        test_pred_dir = os.path.join(opt.val_folder, f"{opt.dataset}_test_preds")

        # 1. Xuất dự đoán trên tập Test qua eval.py
        test_preds = eval(
            model=model,
            testing_data_loader=final_test_data_loader,
            model_path=final_ckpt,
            output_folder=test_pred_dir,
            class_names=class_names,
            device=eval_device,
            conf_thres=opt.conf_thres,
            iou_thres=opt.iou_thres,
            max_det=opt.max_det,
            half=(opt.amp and opt.gpu_mode),
            save_txt=True,
            return_preds=True,
            desc="Testing on Test Set (Best Weight)",
        )

        # 2. Tính toán metrics trên tập Test qua measure.py
        test_results = metrics(
            pred_dir=test_pred_dir,
            label_dir=test_label_dir,
            class_names=class_names,
            dataset_type=opt.dataset,
            conf_thres=opt.conf_thres,
            in_memory_preds=test_preds,
        )
        t_map50 = test_results["mAP50"]
        t_map75 = test_results["mAP75"]
        t_map50_95 = test_results["mAP50_95"]
        t_p = test_results["mean_precision"]
        t_r = test_results["mean_recall"]
        with open(metrics_md_path, "a", encoding="utf-8") as f:
            f.write(
                f"| **Test (Best)** | - | - | - | - | "
                f"**{t_p:.4f}** | **{t_r:.4f}** | **{t_map50:.4f}** | **{t_map75:.4f}** | "
                f"**{t_map50_95:.4f}** | - | `{os.path.basename(final_ckpt)}` |\n"
            )
