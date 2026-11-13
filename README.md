# Low-Light-Object-Detection

## Cài đặt môi trường

```bash
# 1. Tạo môi trường conda với Python 3.10
conda create -n llod python=3.10 -y

# 2. Kích hoạt môi trường
conda activate llod

# 3. Cài đặt PyTorch (CUDA 12.8)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu128

# 4. Cài đặt các thư viện phụ trợ
pip install opencv-python tqdm
```
