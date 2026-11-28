"""
Common modules and building blocks for YOLOv11 architecture (`net.Yolov11.common`).
Based on Ultralytics (`ultralytics/nn/modules/conv.py` and `ultralytics/nn/modules/block.py`).

Includes:
  - autopad, Conv, Conv2, LightConv, DWConv, DWConvTranspose2d, ConvTranspose, Focus, GhostConv
  - ChannelAttention, SpatialAttention, CBAM, Concat
  - Bottleneck, BottleneckCSP, C1, C2, C2f, C3, C3x, C3TR, C3SPP, C3Ghost, GhostBottleneck
  - C3k, C3k2 (Primary CSP blocks in YOLOv11)
  - SPP, SPPF, SCDown, DFL, Proto
  - Attention, PSABlock, PSA, C2PSA (Position-Sensitive Attention in YOLOv11)
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

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


class Conv2(Conv):
    """Simplified RepConv module with parallel 1x1 branch and Conv fusing."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        g: int = 1,
        d: int = 1,
        act: Union[bool, nn.Module] = True,
    ):
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(
            c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def fuse_convs(self) -> None:
        """Fold the 1x1 branch into the main convolution."""
        if not hasattr(self, "cv2"):
            return
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = super().forward


class LightConv(nn.Module):
    """Light convolution module with 1x1 pointwise and depthwise convolutions."""

    def __init__(self, c1: int, c2: int, k: int = 1, act: Optional[nn.Module] = None):
        super().__init__()
        act = nn.ReLU() if act is None else act
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x))


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

    default_act = nn.SiLU()

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


class Focus(nn.Module):
    """Focus wh information into c-space."""

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


class GhostConv(nn.Module):
    """Ghost Convolution."""

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
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(
            self.cv1(
                torch.cat(
                    [torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]],
                    1,
                )
            )
        )


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1: int, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension: int = 1):
        super().__init__()
        self.d = dimension

    def forward(self, x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        return torch.cat(x, self.d)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).
    Proposed in Generalized Focal Loss https://arxiv.org/abs/2006.04388
    """

    def __init__(self, c1: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """Ultralytics YOLO models mask Proto module for segmentation models."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32):
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck."""

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
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1: int, c2: int, n: int = 1):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

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
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

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
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        g: int = 1,
        e: float = 0.5,
        k: int = 3,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
        )


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
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n))
        )


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),
            GhostConv(c_, c2, 1, 1, act=False),
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
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer."""

    def __init__(self, c1: int, c2: int, k: Tuple[int, ...] = (5, 9, 13)):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


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


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 / YOLOv11."""

    export = False

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 5,
        n: int = 3,
        shortcut: bool = False,
        act: Optional[Union[bool, nn.Module]] = None,
    ):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        if act is None:
            act = not shortcut
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = [self.cv1(x)]
        k = self.m.kernel_size
        separable = not self.export and x.device.type == "cpu"
        for _ in range(getattr(self, "n", 3)):
            y.append(
                F.max_pool2d(
                    F.max_pool2d(y[-1], (1, k), 1, (0, k // 2)), (k, 1), 1, (k // 2, 0)
                )
                if separable
                else self.m(y[-1])
            )
        y = self.cv2(torch.cat(y, 1))
        return y + x if getattr(self, "add", False) else y


class Attention(nn.Module):
    """Multi-head self-attention module with depthwise positional encoding."""

    format = None

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)

        if self.format == "coreml" and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(
                q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
            )
            x = x.transpose(-2, -1).reshape(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        else:
            attn = ((q * self.scale).transpose(-2, -1) @ k).softmax(dim=-1)
            x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(
                v.reshape(B, C, H, W)
            )
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """Position-Sensitive Attention block with Attention + FFN."""

    def __init__(
        self,
        c: int,
        attn_ratio: float = 0.5,
        num_heads: int = 4,
        shortcut: bool = True,
    ) -> None:
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """Position-Sensitive Attention (PSA) module."""

    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
        self.ffn = nn.Sequential(
            Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """C2PSA module with stacked PSABlock modules (Primary attention block in YOLOv11)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
                for _ in range(n)
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class SAB(nn.Module):
    """
    Transposed Channel Self-Attention Block (MDTA-style)
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
        gamma_init: float = 0.1,
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
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))

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

        return x + self.gamma * out


class EdgeExtractor(nn.Module):
    """
    Extracts a structural edge priority map from the input image (inspired by IGMamba-EdgeLCA).
    Pipeline:
      1. Channel reduction to 1-channel luminance/intensity.
      2. Gaussian blur pre-filtering to suppress low-light shot/ISO noise.
      3. Weber-Fechner logarithmic illumination compression for dark/backlit robustness.
      4. Multi-directional spatial gradients (Sobel X, Sobel Y, Laplacian) fused via 1x1 Conv.
      5. Per-image Min-Max normalization and learnable affine + sigmoid priority mapping.
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
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.offset = nn.Parameter(torch.tensor(-1.5))
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
    Edge-Guided Transposed Channel Self-Attention Block (MDTA-style)
    with residual gating (`x + gamma * out`).
    Configured with `gamma_init=0.1` and `edge_scale_init=2.0` for early Backbone P2 (160x160).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = False,
        reduction: int = 2,
        kv_dim: Optional[int] = None,
        gamma_init: float = 0.1,
        edge_scale_init: float = 2.0,
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
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))
        self.edge_scale = nn.Parameter(torch.tensor(float(edge_scale_init)))

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


class MultiPriorIllumExtractor(nn.Module):
    """
    Parameter-free Multi-Prior Analytic Luminance Stack inspired by Multinex (CVPR 2026),
    enhanced with Weber-Fechner / mu-law logarithmic dark-region expansion (PA1-v2).

    Computes the 4 complementary illumination descriptors S_L(I) in FP32 for AMP safety:
      1. Y_Rec.709   = 0.2126*R + 0.7152*G + 0.0722*B        (ITU-R BT.709 perceptual brightness)
      2. Y_vmax      = max(R, G, B)                           (Max-RGB / highlight energy proxy)
      3. Y_lightness = 0.5 * (max(R, G, B) + min(R, G, B))   (HSL lightness / contrast regularization)
      4. Y_L2        = sqrt((R^2 + G^2 + B^2) / 3 + eps)     (Normalized RGB vector energy in [0, 1])

    Applies:
      - Stride-4 area/average downsampling before analytic extraction to suppress high-ISO
        dark-region shot noise (~16x variance reduction) while cutting FLOPs by 16x for P2/4.
      - Optional mu-law log compression (`log(1 + mu * S_L) / log(1 + mu)`, default `log_mu=20.0`)
        to expand dark-region resolution (~4x in [0, 0.15]) so CWA & FiLM gates distinguish dark
        objects from background rather than being dominated by bright glare pixels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        downsample_stride: int = 4,
        eps: float = 1e-6,
        log_mu: float = 20.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.downsample_stride = int(downsample_stride)
        self.eps = float(eps)
        self.log_mu = float(log_mu)

    def extract_linear(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw linear 4-channel S_L in FP32 [0, 1] at stride-4 resolution."""
        _, c, h, w = x.shape
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            x_fp32 = x.float().clamp(0.0, 1.0)
            s = self.downsample_stride
            if s > 1 and h >= s and w >= s:
                if (h % s == 0) and (w % s == 0):
                    x_ds = F.avg_pool2d(x_fp32, kernel_size=s, stride=s)
                else:
                    x_ds = F.interpolate(
                        x_fp32,
                        size=(max(1, h // s), max(1, w // s)),
                        mode="area",
                    )
            else:
                x_ds = x_fp32

            if c == 3:
                r = x_ds[:, 0:1]
                g = x_ds[:, 1:2]
                b_ch = x_ds[:, 2:3]
                y_rec709 = 0.2126 * r + 0.7152 * g + 0.0722 * b_ch
                y_vmax = x_ds.amax(dim=1, keepdim=True)
                y_vmin = x_ds.amin(dim=1, keepdim=True)
                y_lightness = 0.5 * (y_vmax + y_vmin)
                y_l2 = torch.sqrt((r * r + g * g + b_ch * b_ch) * (1.0 / 3.0) + self.eps)
            else:
                y_rec709 = x_ds.mean(dim=1, keepdim=True)
                y_vmax = x_ds.amax(dim=1, keepdim=True)
                y_vmin = x_ds.amin(dim=1, keepdim=True)
                y_lightness = 0.5 * (y_vmax + y_vmin)
                y_l2 = torch.sqrt((x_ds * x_ds).mean(dim=1, keepdim=True) + self.eps)

            return torch.cat([y_rec709, y_vmax, y_lightness, y_l2], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            illum_stack = self.extract_linear(x)
            if self.log_mu > 0.0:
                illum_stack = torch.log1p(self.log_mu * illum_stack) / math.log1p(self.log_mu)
        return illum_stack.to(dtype=x.dtype)


class ComponentWiseAttention(nn.Module):
    """
    Multinex Component-wise Attention (CWA, Eq. 16) with BatchNorm dynamic-range unlocking (PA1-v2):
      A = sigmoid(Conv1x1(BN(DWConv7x7(S))))

    Preserves per-descriptor independence via grouped 7x7 spatial filtering, then normalizes
    each descriptor across the dataset distribution via BatchNorm2d before 1x1 projection so
    sigmoid activations stay well-spread across dark, mid, and bright regimes (avoiding the
    [0.47..0.50] saturation trap on low-light scenes). Set `use_bn=False` for legacy PA1-v1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        use_bn: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.use_bn = bool(use_bn)
        self.padding_mode = padding_mode
        padding = kernel_size // 2
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
            groups=in_channels,
            bias=not self.use_bn,
        )
        self.bn = nn.BatchNorm2d(in_channels) if self.use_bn else nn.Identity()
        self.pwconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.use_bn,
        )
        self.act = nn.Sigmoid()

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.act(self.pwconv(self.bn(self.dwconv(s))))


class IllumRetinexGate(nn.Module):
    """
    Post-Attention Feature-Space Retinex-Delta FiLM Gate conditioned on S_L via Multinex CWA (PA1-v2).

    Improvements over PA1-v1:
      - BatchNorm2d after `proj` and `refine_pw` unlocks spatial contrast in `scale_head` so
        FiLM scale actively adapts between dark and bright regions instead of collapsing flat.
      - Darkness-weighted additive shift (`(1 - Y_Rec709_log) * shift_head(feat)`) focuses
        additive shadow-lifting on under-exposed regions and vanishes cleanly on bright glare.
    """

    def __init__(
        self,
        illum_channels: int,
        dim: int,
        kernel_size: int = 7,
        use_bn: bool = True,
        darkness_weighted_shift: bool = True,
    ):
        super().__init__()
        self.use_bn = bool(use_bn)
        self.darkness_weighted_shift = bool(darkness_weighted_shift)
        hidden = max(dim // 4, 16)

        self.proj = nn.Conv2d(illum_channels, hidden, kernel_size=1, bias=not self.use_bn)
        self.proj_bn = nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity()
        self.proj_act = nn.SiLU()
        self.cwa = ComponentWiseAttention(
            illum_channels, hidden, kernel_size=kernel_size, use_bn=self.use_bn
        )
        self.refine_dw = nn.Conv2d(
            hidden, hidden, kernel_size=3, stride=1, padding=1, groups=hidden, bias=True
        )
        self.refine_pw = nn.Conv2d(hidden, hidden, kernel_size=1, bias=not self.use_bn)
        self.refine_bn = nn.BatchNorm2d(hidden) if self.use_bn else nn.Identity()
        self.refine_act = nn.SiLU()
        self.scale_head = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)
        self.shift_head = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

        # Near-identity init with strong step-1 gradient flow (2 * sigmoid(0) = 1.0, shift ≈ 0.0)
        nn.init.normal_(self.scale_head.weight, mean=0.0, std=0.08 if self.use_bn else 0.05)
        nn.init.zeros_(self.scale_head.bias)
        nn.init.normal_(self.shift_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.shift_head.bias)

    def forward(self, illum: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.proj_act(self.proj_bn(self.proj(illum)))
        feat = feat * self.cwa(illum)
        feat = self.refine_act(self.refine_bn(self.refine_pw(self.refine_dw(feat))))
        scale = 2.0 * torch.sigmoid(self.scale_head(feat))
        shift = self.shift_head(feat)
        if self.darkness_weighted_shift:
            dark_mask = (1.0 - illum[:, 0:1]).clamp(0.0, 1.0)
            shift = dark_mask * shift
        return scale, shift


class IllumSAB(nn.Module):
    """
    PA1-v2: Multi-Prior CWA Illumination-Guided Transposed Channel Self-Attention Block (IllumSAB).

    Enhances Backbone P2 (160x160) SAB with Multinex's mu-law log-expanded 4-channel luminance stack:
      1. Pre-Attention Centered CWA Q/K Re-weighting (with BN-unlocked CWA dynamic range):
         Modulates Q and K by `1 + 0.9 * tanh(illum_scale) * (2 * CWA(S_L) - 1)` in (0.1, 1.9),
         giving strong spatial contrast to simultaneously boost dark-region correlations and
         suppress glaring lights before spatial L2 normalization (`illum_scale_init=0.3`).
      2. Post-Attention Darkness-Weighted Retinex-Delta FiLM Gate (`IllumRetinexGate`):
         Applies spatially-adaptive per-channel `(scale * out + (1 - Y_log) * shift)` before
         residual fusion `x + gamma * out` (`gamma_init=0.1` for smooth COCO pretrained adaptation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        bias: bool = False,
        reduction: int = 2,
        kv_dim: Optional[int] = None,
        gamma_init: float = 0.1,
        illum_channels: int = 4,
        illum_scale_init: float = 0.3,
        use_bn: bool = True,
        darkness_weighted_shift: bool = True,
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
        self.illum_channels = illum_channels
        self.use_bn = bool(use_bn)
        self.darkness_weighted_shift = bool(darkness_weighted_shift)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.gamma = nn.Parameter(torch.full((1, dim, 1, 1), float(gamma_init)))
        self.illum_scale = nn.Parameter(torch.full((1, inner_dim, 1, 1), float(illum_scale_init)))

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

        self.illum_qk_cwa = ComponentWiseAttention(
            illum_channels, inner_dim, kernel_size=7, use_bn=self.use_bn
        )
        self.illum_gate = IllumRetinexGate(
            illum_channels,
            dim,
            kernel_size=7,
            use_bn=self.use_bn,
            darkness_weighted_shift=self.darkness_weighted_shift,
        )

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
        illum_map: Optional[torch.Tensor] = None,
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

        illum_scaled: Optional[torch.Tensor] = None
        if illum_map is not None:
            if illum_map.shape[-2:] != (h, w):
                illum_scaled = F.interpolate(
                    illum_map, size=(h, w), mode="bilinear", align_corners=False
                )
            else:
                illum_scaled = illum_map
            illum_scaled = illum_scaled.to(dtype=x.dtype)

            qk_attn = self.illum_qk_cwa(illum_scaled)
            qk_scale = 0.9 * torch.tanh(self.illum_scale).to(dtype=q.dtype)
            qk_weight = 1.0 + qk_scale * (2.0 * qk_attn - 1.0)
            q = q * qk_weight
            k = k * qk_weight

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

        if illum_scaled is not None:
            scale, shift = self.illum_gate(illum_scaled)
            out = out * scale + shift

        return x + self.gamma * out


class IllumConcat(nn.Module):
    """
    IG-PAN: Illumination-Guided Complementary Neck Fusion (`IllumConcat`).

    Replaces static `Concat` at YOLOv11 FPN+PAN junctions (L13, L16, L19, L22) while keeping
    all 25 layer indices and `C3k2`/`Detect` channel dimensions 100% compatible with pretrained
    COCO (`yolo11*.pt`) and `P2-IllumSAB` checkpoints.

    At each multi-scale Neck junction `(80x80, 40x40, 20x20)`, `IllumConcat` fuses a Deep Semantic
    Context branch (`f_ctx`) and a Shallow Spatial Detail branch (`f_det`) conditioned on the
    anti-aliased 4-prior luminance stack `S_L = [Y_Rec709, V_MaxRGB, L_Lightness, E_EnergyL2]`:
      1. Reflect-padded BN-unlocked Multinex CWA Illumination Encoder (`S_L -> h_il`), preserving
         dataset-wide brightness awareness without zero-padding border artifacts.
      2. Complementary Context-Detail Spatial Routing (`d_bal in (-1, +1)`):
         - `M_ctx = 1 - alpha_ctx * d_bal` (amplifies top-down/deep semantic context in dark shadows)
         - `M_det = 1 + alpha_det * d_bal - beta_glare * Y_glare * g_glare` (amplifies crisp shallow
           edges in well-lit zones while damping streetlamp/headlight blooming halos on `f_det`).
      3. Global Exposure-Aware Channel Recalibration (`W_ctx, W_det`), conditioned on global scene
         exposure statistics and detached branch RMS balance without any feature shortcut collapse.
    """

    def __init__(
        self,
        c0: int,
        c1: int,
        d: int = 1,
        deep_first: bool = True,
        illum_channels: int = 4,
        hidden_dim: int = 16,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.c0 = int(c0)
        self.c1 = int(c1)
        self.d = int(d)
        self.deep_first = bool(deep_first)
        self.illum_channels = int(illum_channels)
        self.hidden_dim = max(int(hidden_dim), 8)
        self.kernel_size = int(kernel_size)

        self.c_ctx = self.c0 if self.deep_first else self.c1
        self.c_det = self.c1 if self.deep_first else self.c0
        c_total = self.c_ctx + self.c_det
        gap_dim = max(self.hidden_dim // 2, 4)

        # 1. Reflect-padded BN-unlocked CWA Illumination Encoder
        self.proj = nn.Conv2d(self.illum_channels, self.hidden_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(self.hidden_dim)
        self.proj_act = nn.SiLU()
        self.cwa = ComponentWiseAttention(
            self.illum_channels,
            self.hidden_dim,
            kernel_size=self.kernel_size,
            use_bn=True,
            padding_mode="reflect",
        )
        self.refine_dw = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            groups=self.hidden_dim,
            bias=True,
        )
        self.refine_pw = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False)
        self.refine_bn = nn.BatchNorm2d(self.hidden_dim)
        self.refine_act = nn.SiLU()

        # 2. Spatial Routing & Glare Suppression Heads
        self.spatial_heads = nn.Conv2d(self.hidden_dim, 2, kernel_size=1, bias=True)
        self.route_ctx = nn.Parameter(torch.full((1, self.c_ctx, 1, 1), 0.25))
        self.route_det = nn.Parameter(torch.full((1, self.c_det, 1, 1), 0.25))
        self.glare_det = nn.Parameter(torch.full((1, self.c_det, 1, 1), -1.5))

        # 3. Global Exposure-Aware Channel Recalibration
        self.global_squeeze = nn.Sequential(
            nn.Conv2d(self.hidden_dim, gap_dim, kernel_size=1, bias=True),
            nn.SiLU(),
        )
        self.channel_head = nn.Conv2d(
            gap_dim + self.illum_channels + 1, c_total, kernel_size=1, bias=True
        )
        self.gamma_cali = nn.Parameter(torch.full((1, c_total, 1, 1), 0.20))

        # Near-identity initialization (<1% perturbation at Step 0 with active Step-1 gradients)
        nn.init.normal_(self.spatial_heads.weight, mean=0.0, std=0.04)
        with torch.no_grad():
            self.spatial_heads.bias.copy_(torch.tensor([0.0, -1.5]))
        nn.init.normal_(self.channel_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.channel_head.bias)

    @staticmethod
    def _pool_illum(illum_map: torch.Tensor, h: int, w: int, dtype: torch.dtype) -> torch.Tensor:
        h0, w0 = illum_map.shape[-2:]
        if (h0, w0) == (h, w):
            s = illum_map
        elif h0 >= h and w0 >= w and (h0 % h == 0) and (w0 % w == 0):
            kh, kw = h0 // h, w0 // w
            s = F.avg_pool2d(illum_map, kernel_size=(kh, kw), stride=(kh, kw))
        else:
            s = F.interpolate(illum_map, size=(h, w), mode="area")
        return s.to(dtype=dtype)

    @staticmethod
    def _glare_prior(s: torch.Tensor) -> torch.Tensor:
        # Combines high-intensity luminance energy (V_MaxRGB * E_EnergyL2) and saturated neon/brake
        # chromatic highlight gap (V_MaxRGB * max(V_MaxRGB - L_Lightness, 0)); ~0 in dark shadows.
        v_max = s[:, 1:2]
        l_light = s[:, 2:3]
        e_l2 = s[:, 3:4] if s.shape[1] >= 4 else v_max
        chroma_gap = (v_max - l_light).clamp(0.0, 1.0)
        return (0.70 * (v_max * e_l2) + 0.60 * (v_max * chroma_gap)).clamp(0.0, 1.0)

    def _compute_modulation(
        self,
        f_ctx: torch.Tensor,
        f_det: torch.Tensor,
        illum_map: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        dtype = f_ctx.dtype
        h, w = f_ctx.shape[-2:]
        s = self._pool_illum(illum_map, h, w, dtype)

        a_cwa = self.cwa(s)
        feat = self.proj_act(self.proj_bn(self.proj(s)))
        feat = feat * a_cwa
        h_il = self.refine_act(self.refine_bn(self.refine_pw(self.refine_dw(feat))))

        z_route, z_glare = self.spatial_heads(h_il).chunk(2, dim=1)
        d_bal = 2.0 * torch.sigmoid(z_route) - 1.0
        g_glare = torch.sigmoid(z_glare)
        y_glare = self._glare_prior(s)
        eff_glare = y_glare * g_glare

        alpha_ctx = 0.8 * torch.tanh(self.route_ctx).to(dtype=dtype)
        alpha_det = 0.8 * torch.tanh(self.route_det).to(dtype=dtype)
        beta_glare = 0.8 * torch.sigmoid(self.glare_det).to(dtype=dtype)

        m_ctx = 1.0 - alpha_ctx * d_bal
        m_det = (1.0 + alpha_det * d_bal - beta_glare * eff_glare).clamp(min=0.05)

        il_gap = self.global_squeeze(h_il.mean(dim=(-2, -1), keepdim=True))
        with torch.no_grad():
            rms_ctx = torch.sqrt(f_ctx.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
            rms_det = torch.sqrt(f_det.float().pow(2).mean(dim=(1, 2, 3), keepdim=True) + 1e-6)
            bal_ratio = (rms_ctx / (rms_ctx + rms_det + 1e-6)).to(dtype=dtype)
            s_gap = s.mean(dim=(-2, -1), keepdim=True)

        g_vec = torch.cat([il_gap, s_gap, bal_ratio], dim=1)
        ch_mod = torch.tanh(self.channel_head(g_vec))
        gamma_c = 0.5 * torch.tanh(self.gamma_cali).to(dtype=dtype)
        w_all = 1.0 + gamma_c * ch_mod
        w_ctx, w_det = torch.split(w_all, [self.c_ctx, self.c_det], dim=1)

        return {
            "s": s,
            "a_cwa": a_cwa,
            "h_il": h_il,
            "d_bal": d_bal,
            "g_glare": g_glare,
            "y_glare": y_glare,
            "eff_glare": eff_glare,
            "alpha_ctx": alpha_ctx,
            "alpha_det": alpha_det,
            "beta_glare": beta_glare,
            "gamma_c": gamma_c,
            "m_ctx": m_ctx,
            "m_det": m_det,
            "w_ctx": w_ctx,
            "w_det": w_det,
        }

    def forward_with_diagnostics(
        self,
        x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        illum_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if illum_map is None or len(x) != 2:
            return torch.cat(list(x), self.d), None

        if self.deep_first:
            f_ctx, f_det = x[0], x[1]
        else:
            f_det, f_ctx = x[0], x[1]

        mod = self._compute_modulation(f_ctx, f_det, illum_map)
        f_ctx_out = f_ctx * mod["m_ctx"] * mod["w_ctx"]
        f_det_out = f_det * mod["m_det"] * mod["w_det"]
        out = (
            torch.cat([f_ctx_out, f_det_out], self.d)
            if self.deep_first
            else torch.cat([f_det_out, f_ctx_out], self.d)
        )
        mod["f_ctx_in"] = f_ctx
        mod["f_det_in"] = f_det
        mod["f_ctx_out"] = f_ctx_out
        mod["f_det_out"] = f_det_out
        mod["out"] = out
        return out, mod

    def forward(
        self,
        x: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]],
        illum_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if illum_map is None or len(x) != 2:
            return torch.cat(list(x), self.d)

        if self.deep_first:
            f_ctx, f_det = x[0], x[1]
        else:
            f_det, f_ctx = x[0], x[1]

        mod = self._compute_modulation(f_ctx, f_det, illum_map)
        f_ctx_out = f_ctx * mod["m_ctx"] * mod["w_ctx"]
        f_det_out = f_det * mod["m_det"] * mod["w_det"]
        if self.deep_first:
            return torch.cat([f_ctx_out, f_det_out], self.d)
        return torch.cat([f_det_out, f_ctx_out], self.d)


class C3k2(C2f):
    """
    C2f variant whose inner blocks are Bottleneck, C3k, or Bottleneck + PSABlock attention modules.
    Primary building block in YOLOv11.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
    ):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g),
                PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1)),
            )
            if attn
            else C3k(self.c, self.c, 2, shortcut, g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class SCDown(nn.Module):
    """SCDown module for downsampling with separable convolutions."""

    def __init__(self, c1: int, c2: int, k: int, s: int):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv2(self.cv1(x))


class TransformerLayer(nn.Module):
    """Transformer layer (LayerNorm removed)."""

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
    """Vision Transformer block."""

    def __init__(self, c1: int, c2: int, num_heads: int, num_layers: int):
        super().__init__()
        self.conv = Conv(c1, c2) if c1 != c2 else None
        self.linear = nn.Linear(c2, c2)
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
