import argparse
import copy
import time

import torch
from thop import profile

from net import SUPPORTED_MODELS, build_yolo_model


def _format_secondary_output(secondary):
    if isinstance(secondary, dict):
        if "one2many" in secondary and isinstance(secondary["one2many"], dict):
            return {
                k: [tuple(f.shape) for f in v.get("feats", [])]
                for k, v in secondary.items()
                if isinstance(v, dict)
            }
        if "feats" in secondary:
            return [tuple(f.shape) for f in secondary["feats"]]
        return list(secondary.keys())
    if isinstance(secondary, (list, tuple)):
        return [tuple(f.shape) for f in secondary]
    return type(secondary).__name__


def benchmark_model(
    model_name: str = "yolov5n",
    num_classes: int = 12,
    modular: bool = False,
    img_size: int = 640,
    warmup: int = 50,
    iters: int = 100,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_yolo_model(
        model_name=model_name,
        num_classes=num_classes,
        in_channels=3,
        modular=modular,
    ).to(device)
    x = torch.rand(1, 3, img_size, img_size, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()
    time_start = time.time()
    with torch.no_grad():
        for _ in range(iters):
            out = model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    time_end = time.time()

    time_avg = (time_end - time_start) / max(iters, 1)
    fps = 1.0 / time_avg if time_avg > 0 else 0.0
    pred_main = out[0] if isinstance(out, tuple) else out
    pred_sec = out[1] if isinstance(out, tuple) and len(out) > 1 else None

    n_param = sum(p.nelement() for p in model.parameters())
    model_profile = copy.deepcopy(model).eval()
    macs, _ = profile(model_profile, inputs=(x,), verbose=False)

    print(f"=== {model_name} (nc={num_classes}, modular={modular}, device={device}) ===")
    print(f"Input shape: {tuple(x.shape)}")
    print(
        f"Pred output shape: {tuple(pred_main.shape)} | Feature maps: {_format_secondary_output(pred_sec)}"
    )
    print(f"Average Time over {iters} iterations: {time_avg:.6f} s ({fps:.2f} FPS)")
    print(f"n_paras: {n_param / (2**20):.4f}M ({n_param / 1e6:.4f}M params)")
    print(f"FLOPs: {macs / (2**30):.4f}G (GFLOPs @ 2*MACs: {(2 * macs) / 1e9:.4f}G)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark YOLO model latency, params, and FLOPs")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        choices=["all"] + list(SUPPORTED_MODELS),
        help="Model to benchmark ('all' tests yolov5n, yolov11n, yolov26n)",
    )
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--modular", action="store_true")
    parser.add_argument("--imgSize", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    models_to_run = (
        ["yolov5n", "yolov11n", "yolov26n"] if args.model == "all" else [args.model]
    )
    for m_name in models_to_run:
        benchmark_model(
            model_name=m_name,
            num_classes=args.num_classes,
            modular=args.modular,
            img_size=args.imgSize,
            warmup=args.warmup,
            iters=args.iters,
        )
