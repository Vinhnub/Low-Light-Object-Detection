from thop import profile
import copy
import time
import torch
from net.Yolov5 import yolov5n, yolov5s, yolov5m, yolov5l, yolov5x, YOLOv5


device = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo model (12 classes cho bộ dữ liệu ExDark, có thể đổi sang yolov5n/s/m/l/x)
model = yolov5n(num_classes=12, modular=False).to(device)
input = torch.rand(1, 3, 640, 640).to(device)
model.eval()

# Warmup to initialize CUDA and avoid cold start overhead
with torch.no_grad():
    for _ in range(50):
        _ = model(input)

# Measure time
if device == "cuda":
    torch.cuda.synchronize()
time_start = time.time()
num_iterations = 100
with torch.no_grad():
    for _ in range(num_iterations):
        out = model(input)
if device == "cuda":
    torch.cuda.synchronize()
time_end = time.time()

time_avg = (time_end - time_start) / num_iterations
fps = 1.0 / time_avg if time_avg > 0 else 0.0
print(f"Device: {device}")
print(f"Input shape: {tuple(input.shape)}")
print(f"Pred output shape: {tuple(out[0].shape)} | Feature maps: {[tuple(f.shape) for f in out[1]]}")
print(f"Average Time over {num_iterations} iterations: {time_avg:.6f} s ({fps:.2f} FPS)")

n_param = sum([p.nelement() for p in model.parameters()])
n_paras = f"n_paras: {(n_param / 2**20):.4f}M ({n_param / 1e6:.4f}M params)\n"
print(n_paras)

# Profile MACs / FLOPs on a clean copy so thop hooks do not pollute the model
model_profile = copy.deepcopy(model).eval()
macs, params = profile(model_profile, inputs=(input,), verbose=False)
print(f"FLOPs:{macs / (2**30):.4f}G (GFLOPs @ 2*MACs: {(2 * macs) / 1e9:.4f}G)")
