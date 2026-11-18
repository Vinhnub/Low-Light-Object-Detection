"""
Common modules and building blocks for YOLOv5 architecture.
Based on Ultralytics YOLOv5 (models/common.py).
Includes:
  - Conv, DWConv, DWConvTranspose2d, ConvTranspose, Focus, GhostConv
  - Bottleneck, BottleneckCSP, CrossConv, GhostBottleneck
  - C3, C3x, C3TR, C3SPP, C3Ghost (CSPNet blocks with 3 convolutions)
  - SPP, SPPF (Spatial Pyramid Pooling - Fast)
  - TransformerLayer, TransformerBlock
  - Contract, Expand, Concat, Proto
"""

import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(
    k: Union[int, List[int], Tuple[int, ...]],
    p: Optional[Union[int, List[int], Tuple[int, ...]]] = None,
    d: int = 1,
) -> Union[int, List[int]]:
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = (
            d * (k - 1) + 1
            if isinstance(k, int)
            else [d * (x - 1) + 1 for x in k]
        )  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Union[int, Tuple[int, int]] = 1,
        s: Union[int, Tuple[int, int]] = 1,
        p: Optional[Union[int, Tuple[int, int]]] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p1: int = 0,
        p2: int = 0,
    ):
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 2,
        s: int = 2,
        p: int = 0,
        bn: bool = True,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = (
            self.default_act
            if act is True
            else act
            if isinstance(act, nn.Module)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv_transpose(x))


class TransformerLayer(nn.Module):
    """
    Transformer layer https://arxiv.org/abs/2010.11929
    (LayerNorm layers removed for better performance).
    """

    def __init__(self, c: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    """Vision Transformer https://arxiv.org/abs/2010.11929."""

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(
            *(TransformerLayer(c2, num_heads) for _ in range(num_layers))
        )
        self.c2 = c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """
    CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    (Used in YOLOv5 v1.0 - v3.0).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    """Cross Convolution Downsample."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        g: int = 1,
        e: float = 1.0,
        shortcut: bool = False,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions (Primary building block in YOLOv5 v5.0+ / v6.0+ / v7.0)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n))
        )


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Tuple[int, ...] = (5, 9, 13),
    ):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher.
    Mathematically equivalent to SPP(k=(5, 9, 13)) when k=5, but >2x faster.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C3SPP(C3):
    """C3 module with SPP()."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: Tuple[int, ...] = (5, 9, 13),
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        g: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
    ):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class Focus(nn.Module):
    """
    Focus wh information into c-space (Used in YOLOv5 v1.0 - v5.0,
    replaced by 6x6 stride-2 Conv in YOLOv5 v6.0+).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                (
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ),
                1,
            )
        )


class Contract(nn.Module):
    """Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)."""

    def __init__(self, gain: int = 2):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    """Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)."""

    def __init__(self, gain: int = 2):
        super().__init__()
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        return torch.cat(x, self.d)


class Proto(nn.Module):
    """YOLOv5 mask Proto module for instance segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class EdgeExtractor(nn.Module):
    """
    Extracts a structural edge priority map from the input image (inspired by IGMamba-EdgeLCA).
    Pipeline:
      1. Channel reduction to 1-channel luminance/intensity.
      2. Gaussian blur pre-filtering to suppress low-light shot/ISO noise.
      3. Multi-directional spatial gradients (Sobel X, Sobel Y, Laplacian) fused via 1x1 Conv.
      4. Per-image Min-Max normalization and learnable affine + sigmoid priority mapping.
    """

    def __init__(
        self,
        in_channels: int = 3,
        rgb_to_gray: bool = True,
        use_blur: bool = True,
        blur_kernel_size: int = 3,
        blur_sigma: float = 1.0,
        weber_log: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.rgb_to_gray = rgb_to_gray and (in_channels == 3)
        self.use_blur = use_blur
        self.blur_kernel_size = blur_kernel_size
        self.weber_log = weber_log

        if self.rgb_to_gray:
            self.register_buffer(
                "rgb_weight",
                torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1),
            )

        if self.use_blur:
            self.register_buffer(
                "gaussian_kernel",
                self._create_gaussian_kernel(blur_kernel_size, blur_sigma),
            )

        self.register_buffer(
            "sobel_x",
            torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "sobel_y",
            torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )
        self.register_buffer(
            "laplacian",
            torch.tensor(
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

        self.channel_fusion = nn.Conv2d(3, 1, kernel_size=1, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.offset = nn.Parameter(torch.tensor(-5.0))
        if self.weber_log:
            self.weber_alpha = nn.Parameter(torch.tensor(0.7))

    @staticmethod
    def _create_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        variance = sigma ** 2.0
        gaussian = torch.exp(-(grid_x**2 + grid_y**2) / (2.0 * variance))
        gaussian = gaussian / torch.sum(gaussian)
        return gaussian.view(1, 1, kernel_size, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(
                f"EdgeExtractor initialized with in_channels={self.in_channels}, got {c}."
            )

        if c > 1:
            if self.rgb_to_gray and c == 3:
                target = torch.sum(
                    x * self.rgb_weight.to(dtype=x.dtype), dim=1, keepdim=True
                )
            else:
                target = x.mean(dim=1, keepdim=True)
        else:
            target = x

        if self.use_blur:
            pad = self.blur_kernel_size // 2
            target_blur_pad = F.pad(target, (pad, pad, pad, pad), mode="replicate")
            target = F.conv2d(target_blur_pad, self.gaussian_kernel.to(dtype=target.dtype))

        if getattr(self, "weber_log", True):
            eps = 1e-3
            log_min = math.log(eps)
            t_fp32 = target.float()
            target_log = (torch.log(t_fp32.clamp(min=eps)) - log_min) / (-log_min)
            alpha = self.weber_alpha.float().clamp(0.0, 1.0)
            target = (alpha * target_log + (1.0 - alpha) * t_fp32).to(dtype=x.dtype)

        target_padded = F.pad(target, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(target_padded, self.sobel_x.to(dtype=target.dtype))
        grad_y = F.conv2d(target_padded, self.sobel_y.to(dtype=target.dtype))
        grad_lap = F.conv2d(target_padded, self.laplacian.to(dtype=target.dtype))

        grad_stack = torch.cat([grad_x, grad_y, grad_lap], dim=1)
        gradient_magnitude = torch.abs(self.channel_fusion(grad_stack))

        # Normalize and map priority in FP32 for numerical stability under AMP/FP16
        gradient_flat = gradient_magnitude.float().view(b, -1)
        grad_min = gradient_flat.min(dim=1, keepdim=True)[0]
        grad_max = gradient_flat.max(dim=1, keepdim=True)[0]
        grad_norm = (gradient_flat - grad_min) / (grad_max - grad_min + 1e-6)
        grad_norm = grad_norm.view(b, 1, h, w)

        priority_score = self.scale.float() * grad_norm + self.offset.float()
        return torch.sigmoid(priority_score).to(dtype=x.dtype)


class MultiScaleEdgeGate(nn.Module):
    """Multi-scale spatial edge gating combining 3x3 sharp local and dilated 3x3 wide edge contexts."""

    def __init__(self, dim: int):
        super().__init__()
        self.local_conv = nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True)
        self.wide_dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=True
        )
        self.act = nn.Sigmoid()

    def forward(self, edge: torch.Tensor) -> torch.Tensor:
        feat = self.local_conv(edge)
        return self.act(feat + self.wide_dwconv(feat))


class EdgeSAB(nn.Module):
    """
    Pre-Head Edge-Guided Transposed Channel Self-Attention Block (MDTA-style)
    with residual gating (`x + gamma * out`). Supports Self-Attention (`y=None`)
    and Cross-Attention (`y` or `[x, y]`).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = False,
        reduction: int = 2,
        kv_dim: Optional[int] = None,
    ):
        super().__init__()
        inner_dim = max(dim // max(reduction, 1), num_heads)
        inner_dim = max(num_heads, (inner_dim // num_heads) * num_heads)
        assert inner_dim % num_heads == 0, (
            f"inner_dim ({inner_dim}) must be divisible by num_heads ({num_heads})"
        )

        kv_in = kv_dim or dim
        self.dim = dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = inner_dim // num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), 0.0))
        self.edge_scale = nn.Parameter(torch.tensor(1.0))

        self.q = nn.Conv2d(dim, inner_dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(
            inner_dim, inner_dim, kernel_size=3, stride=1, padding=1, groups=inner_dim, bias=bias
        )
        self.kv = nn.Conv2d(kv_in, inner_dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(
            inner_dim * 2,
            inner_dim * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=inner_dim * 2,
            bias=bias,
        )
        self.edge_gate = MultiScaleEdgeGate(dim)
        self.ffn_gate = nn.Sequential(
            nn.Conv2d(
                inner_dim, inner_dim, kernel_size=3, stride=1, padding=1, groups=inner_dim, bias=False
            ),
            nn.SiLU(inplace=True),
        )
        self.project_out = nn.Conv2d(inner_dim, dim, kernel_size=1, bias=bias)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...]],
        y: Optional[torch.Tensor] = None,
        edge_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(x, (list, tuple)):
            if y is None and len(x) >= 2:
                x, y = x[0], x[1]
            else:
                x = x[0]
        if y is None:
            y = x

        b, _, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        edge_scaled: Optional[torch.Tensor] = None
        if edge_map is not None:
            if edge_map.shape[-2:] != (h, w):
                edge_bil = F.interpolate(
                    edge_map, size=(h, w), mode="bilinear", align_corners=False
                )
                edge_max = F.adaptive_max_pool2d(edge_map, output_size=(h, w))
                edge_scaled = 0.5 * edge_bil + 0.5 * edge_max
            else:
                edge_scaled = edge_map
            weight = (1.0 + self.edge_scale * edge_scaled).to(q.dtype)
            q = q * weight
            k = k * weight

        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        # Normalize in FP32 to prevent FP16 overflow when summing squares over H*W
        q_norm = F.normalize(q.float(), dim=-1, eps=1e-6)
        k_norm = F.normalize(k.float(), dim=-1, eps=1e-6)

        attn = (q_norm @ k_norm.transpose(-2, -1)) * self.temperature.float()
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v.float()).to(x.dtype).reshape(b, self.inner_dim, h, w)
        out = out * self.ffn_gate(out)
        out = self.project_out(out)

        if edge_scaled is not None:
            out = out * self.edge_gate(edge_scaled.to(x.dtype))

        return x + self.gamma * out


SAB = EdgeSAB
