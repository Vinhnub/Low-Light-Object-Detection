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
    ):
        super().__init__()
        self.in_channels = in_channels
        self.rgb_to_gray = rgb_to_gray and (in_channels == 3)
        self.use_blur = use_blur
        self.blur_kernel_size = blur_kernel_size

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


class EdgeCAB(nn.Module):
    """
    Pre-Head Edge-Guided Transposed Channel Attention Block (MDTA-style)
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
        # bias=True lets large-object interior regions at P5 learn an open gate bias
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
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
        out = self.project_out(out)

        if edge_scaled is not None:
            out = out * self.edge_gate(edge_scaled.to(x.dtype))

        return x + self.gamma * out


CAB = EdgeCAB


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
