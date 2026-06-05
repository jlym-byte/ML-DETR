# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv
from .transformer import TransformerBlock

__all__ = (
    'DFL',
    'HGBlock',
    'HGStem',
    'SPP',
    'SPPF',
    'C1',
    'C2',
    'C3',
    'C2f',
    'C3x',
    'C3TR',
    'C3Ghost',
    'GhostBottleneck',
    'Bottleneck',
    'BottleneckCSP',
    'Proto',
    'RepC3',
    'ConvNormLayer',
    'BasicBlock',
    'BottleNeck',
    'Blocks',
    'ContextBlock',
    'TOGLSRConvBranch',
    'TOGLSR',
    'MambaScan2D',
    'SSMamba',
    'LayerNorm',
    'CGLU',
    'PBDConv',
    'PTAttention',
    'DCConv',
    'DAHB',
    'EncoderBlock',
    'ECIE',
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))

################################### RT-DETR PResnet ###################################
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 

class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))
    
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        self.shortcut = shortcut

        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out, 1, stride)

        self.branch2a = ConvNormLayer(ch_in, ch_out, 3, stride, act=act)
        self.branch2b = ConvNormLayer(ch_out, ch_out, 3, 1, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 


    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        if self.shortcut:
            short = x
        else:
            short = self.short(x)
        
        out = out + short
        out = self.act(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, ch_in, ch_out, stride, shortcut, act='relu', variant='d'):
        super().__init__()

        if variant == 'a':
            stride1, stride2 = stride, 1
        else:
            stride1, stride2 = 1, stride

        width = ch_out 

        self.branch2a = ConvNormLayer(ch_in, width, 1, stride1, act=act)
        self.branch2b = ConvNormLayer(width, width, 3, stride2, act=act)
        self.branch2c = ConvNormLayer(width, ch_out * self.expansion, 1, 1)

        self.shortcut = shortcut
        if not shortcut:
            if variant == 'd' and stride == 2:
                self.short = nn.Sequential(OrderedDict([
                    ('pool', nn.AvgPool2d(2, 2, 0, ceil_mode=True)),
                    ('conv', ConvNormLayer(ch_in, ch_out * self.expansion, 1, 1))
                ]))
            else:
                self.short = ConvNormLayer(ch_in, ch_out * self.expansion, 1, stride)

        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        out = self.branch2a(x)
        out = self.branch2b(out)
        out = self.branch2c(out)

        if self.shortcut:
            short = x
        else:
            short = self.short(x)

        out = out + short
        out = self.act(out)

        return out


class Blocks(nn.Module):
    def __init__(self, ch_in, ch_out, block, count, stage_num, act='relu', input_resolution=None, sr_ratio=None, kernel_size=None, kan_name=None, variant='d'):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(count):
            if input_resolution is not None and sr_ratio is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        input_resolution=input_resolution,
                        sr_ratio=sr_ratio)
                )
            elif kernel_size is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kernel_size=kernel_size)
                )
            elif kan_name is not None:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act,
                        kan_name=kan_name)
                )
            else:
                self.blocks.append(
                    block(
                        ch_in, 
                        ch_out,
                        stride=2 if i == 0 and stage_num != 2 else 1, 
                        shortcut=False if i == 0 else True,
                        variant=variant,
                        act=act)
                )
            if i == 0:
                ch_in = ch_out * block.expansion

    def forward(self, x):
        out = x
        for block in self.blocks:
            out = block(out)
        return out


# ML-DETR attention and state-space blocks.
class ContextBlock(nn.Module):
    def __init__(self, inplanes, ratio, pooling_type="att", fusion_types=("channel_mul",)):
        super().__init__()
        assert pooling_type in ["avg", "att"]
        assert isinstance(fusion_types, (list, tuple))
        assert len(fusion_types) > 0
        valid_fusion_types = ["channel_add", "channel_mul"]
        assert all(f in valid_fusion_types for f in fusion_types)

        self.inplanes = inplanes
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == "att":
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_add_conv = None
        if "channel_add" in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
            )

        self.channel_mul_conv = None
        if "channel_mul" in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1),
            )

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == "att":
            input_x = x.view(batch, channel, height * width).unsqueeze(1)
            context_mask = self.conv_mask(x).view(batch, 1, height * width)
            context_mask = self.softmax(context_mask).unsqueeze(-1)
            context = torch.matmul(input_x, context_mask).view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)
        out = x
        if self.channel_mul_conv is not None:
            out = out + out * torch.sigmoid(self.channel_mul_conv(context))
        if self.channel_add_conv is not None:
            out = out + self.channel_add_conv(context)
        return out


class TOGLSRConvBranch(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = Conv(in_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv2 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv3 = Conv(hidden_features, hidden_features, 1, act=nn.ReLU(inplace=True))
        self.conv4 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv5 = Conv(hidden_features, hidden_features, 1, act=nn.SiLU(inplace=True))
        self.conv6 = Conv(hidden_features, hidden_features, 3, g=hidden_features, act=nn.ReLU(inplace=True))
        self.conv7 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1, bias=False), nn.ReLU(inplace=True))
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        res1 = x
        res2 = x
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = self.conv3(x)
        x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)
        res1 = res1 * self.sigmoid_spatial(x)
        return res2 + res1


class TOGLSR(nn.Module):
    """Topology-guided Global-local Spatial Recalibration.

    TOGLSR models sparse same-scale target topology: it selects object-aware tokens,
    propagates their spatial relations, and writes the enhanced token context back
    to the feature map.
    """

    def __init__(self, input_dim=512, embed_dim=32, inner_dim=None, token_ratio=0.03, max_tokens=64, min_tokens=8):
        super().__init__()
        self.local_in_dim = input_dim // 2
        self.global_in_dim = input_dim - self.local_in_dim
        self.embed_dim = inner_dim or embed_dim
        self.out_dim = embed_dim
        self.token_ratio = token_ratio
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        hidden_dim = max(self.embed_dim // 4, 1)

        self.local_11conv = nn.Conv2d(self.local_in_dim, self.embed_dim, 1)
        self.local = TOGLSRConvBranch(in_features=self.embed_dim, hidden_features=self.embed_dim, out_features=self.embed_dim)
        self.global_11conv = nn.Conv2d(self.global_in_dim, self.embed_dim, 1)

        self.token_score = nn.Sequential(
            Conv(self.embed_dim, hidden_dim, 1, act=nn.ReLU(inplace=True)),
            nn.Conv2d(hidden_dim, 1, 3, padding=1, bias=True),
            nn.Sigmoid(),
        )
        self.pos_embed = nn.Sequential(
            nn.Linear(2, self.embed_dim),
            nn.SiLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )
        self.token_norm = nn.LayerNorm(self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.token_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.topology_scale = nn.Parameter(torch.tensor(1.0))
        self.context_refine = nn.Sequential(
            DWConv(self.embed_dim, self.embed_dim, 3, 1),
            Conv(self.embed_dim, self.embed_dim, 1, act=nn.SiLU(inplace=True)),
        )
        self.fusion_gate = nn.Sequential(
            Conv(self.embed_dim * 2 + 1, self.embed_dim, 1, act=nn.SiLU(inplace=True)),
            nn.Conv2d(self.embed_dim, self.embed_dim, 1, bias=True),
            nn.Sigmoid(),
        )
        self.context_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.score_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.out_proj = Conv(self.embed_dim, self.out_dim, 1)

    def _token_count(self, num_points):
        k = max(self.min_tokens, int(num_points * self.token_ratio))
        return min(self.max_tokens, num_points, k)

    def _select_tokens(self, feat, score):
        b, c, h, w = feat.shape
        n = h * w
        k = self._token_count(n)
        flat_score = score.flatten(2).squeeze(1)
        token_score, token_idx = flat_score.topk(k, dim=-1)
        flat_feat = feat.flatten(2).transpose(1, 2)
        gather_idx = token_idx.unsqueeze(-1).expand(-1, -1, c)
        tokens = flat_feat.gather(1, gather_idx) * token_score.unsqueeze(-1)

        y = (token_idx // w).to(feat.dtype)
        x = (token_idx % w).to(feat.dtype)
        if h > 1:
            y = y / (h - 1) * 2.0 - 1.0
        else:
            y = y * 0.0
        if w > 1:
            x = x / (w - 1) * 2.0 - 1.0
        else:
            x = x * 0.0
        coords = torch.stack((x, y), dim=-1)
        return tokens, token_idx, token_score, coords

    def _scatter_tokens(self, tokens, token_idx, token_score, height, width):
        b, k, c = tokens.shape
        context = tokens.new_zeros(b, c, height * width)
        weights = tokens.new_zeros(b, 1, height * width)
        scatter_idx = token_idx.unsqueeze(1).expand(-1, c, -1)
        context.scatter_add_(2, scatter_idx, tokens.transpose(1, 2))
        weights.scatter_add_(2, token_idx.unsqueeze(1), token_score.unsqueeze(1))
        context = context / weights.clamp_min(1e-6)
        context = context.view(b, c, height, width)
        return self.context_refine(context)

    def forward(self, x):
        b, _, h, w = x.shape
        x_local, x_global = torch.split(x, [self.local_in_dim, self.global_in_dim], dim=1)
        local = self.local(self.local_11conv(x_local))
        global_feat = self.global_11conv(x_global)

        score = self.token_score(global_feat)
        tokens, token_idx, token_score, coords = self._select_tokens(global_feat, score)
        tokens = tokens + self.pos_embed(coords)

        norm_tokens = self.token_norm(tokens)
        q = self.q_proj(norm_tokens)
        k = self.k_proj(norm_tokens)
        v = self.v_proj(norm_tokens)
        relation = torch.matmul(q, k.transpose(-2, -1)) * (self.embed_dim ** -0.5)
        topology_bias = -torch.cdist(coords, coords, p=1) * self.topology_scale.abs()
        relation = (relation + topology_bias).softmax(dim=-1)
        tokens = tokens + self.token_proj(torch.matmul(relation, v))

        topo_context = self._scatter_tokens(tokens, token_idx, token_score, h, w)
        gate = self.fusion_gate(torch.cat([local, topo_context, score], dim=1))
        fused = local + self.context_scale * gate * topo_context
        return self.out_proj(fused * (1.0 + self.score_scale * score))

class MambaScan2D(nn.Module):
    """2D scan branch built on the real mamba-ssm Mamba block.

    The innovation of SSMamba is not a hand-written pseudo-Mamba recurrence. This
    branch imports the official mamba-ssm implementation and applies it to four
    image sequences: width, reversed width, height, and reversed height.
    """

    def __init__(self, channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:
            try:
                from mamba_ssm.modules.mamba_simple import Mamba
            except Exception as fallback_exc:
                raise ImportError(
                    "SSMamba requires the real mamba-ssm package. Install it in the torch_gpu "
                    "environment, for example: pip install causal-conv1d mamba-ssm"
                ) from fallback_exc

        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out_proj = Conv(channels, channels, 1, act=False)

    def _scan_width(self, x, reverse=False):
        bs, c, h, w = x.shape
        seq = x.permute(0, 2, 3, 1).reshape(bs * h, w, c).contiguous()
        if reverse:
            seq = seq.flip(1).contiguous()
        seq = self.mamba(self.norm(seq))
        if reverse:
            seq = seq.flip(1).contiguous()
        return seq.reshape(bs, h, w, c).permute(0, 3, 1, 2).contiguous()

    def _scan_height(self, x, reverse=False):
        bs, c, h, w = x.shape
        seq = x.permute(0, 3, 2, 1).reshape(bs * w, h, c).contiguous()
        if reverse:
            seq = seq.flip(1).contiguous()
        seq = self.mamba(self.norm(seq))
        if reverse:
            seq = seq.flip(1).contiguous()
        return seq.reshape(bs, w, h, c).permute(0, 3, 2, 1).contiguous()

    def forward(self, x):
        y = (
            self._scan_width(x, reverse=False)
            + self._scan_width(x, reverse=True)
            + self._scan_height(x, reverse=False)
            + self._scan_height(x, reverse=True)
        ) * 0.25
        return self.out_proj(y)


class SSMamba(nn.Module):
    """Scale-Selective State-Space Mamba aggregation before the detection head.

    TOGLSR focuses on object-level topology recalibration, while SSMamba focuses on
    scale-level state propagation after cross-scale fusion. It receives the final
    multi-scale features, applies shared local/state branches, and uses a global
    scale gate to calibrate Mamba-style state features for each scale.
    """

    def __init__(
        self, channels, out_channels=256, d_state=16, d_conv=4, expand=2, reduction=4, state_channels=None, res_init=0.1
    ):
        super().__init__()
        channels = channels if isinstance(channels, (list, tuple)) else [channels]
        self.num_scales = len(channels)
        self.out_channels = out_channels
        self.state_channels = state_channels or out_channels
        hidden = max(out_channels // reduction, 16)

        self.proj = nn.ModuleList(nn.Identity() if c == out_channels else Conv(c, out_channels, 1) for c in channels)
        self.local_dw = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.local_scale = nn.Parameter(torch.zeros(1))
        self.state_norm = LayerNorm(out_channels)
        self.state_in = Conv(out_channels, self.state_channels, 1, act=False)
        self.state_branch = MambaScan2D(self.state_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.state_out = Conv(self.state_channels, out_channels, 1, act=False)
        self.scale_gate = nn.Sequential(
            nn.Conv2d(out_channels * self.num_scales, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, self.state_channels * self.num_scales, 1, bias=True),
        )
        self.res_scale = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.scale_gate[-1].weight)
        nn.init.zeros_(self.scale_gate[-1].bias)

    def forward(self, x):
        x = x if isinstance(x, (list, tuple)) else [x]
        feats = [proj(feat) for proj, feat in zip(self.proj, x)]
        context = torch.cat([F.adaptive_avg_pool2d(feat, 1) for feat in feats], dim=1)
        gates = self.scale_gate(context).view(context.shape[0], self.num_scales, self.state_channels, 1, 1)
        gates = gates.softmax(dim=1).unbind(dim=1)

        outs = []
        for feat, gate in zip(feats, gates):
            local = feat + self.local_scale * self.local_dw(feat)
            state_input = self.state_in(self.state_norm(local))
            state = self.state_out(self.state_branch(state_input) * gate)
            outs.append(local + self.res_scale * state)
        return outs


class LayerNorm(nn.Module):
    def __init__(self, dim, layer_norm_type="BiasFree"):
        super().__init__()
        self.layer_norm_type = layer_norm_type
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if layer_norm_type != "BiasFree" else None

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        if self.layer_norm_type == "BiasFree":
            sigma = x.var(-1, keepdim=True, unbiased=False)
            x = x / torch.sqrt(sigma + 1e-5)
            x = x * self.weight
        else:
            mu = x.mean(-1, keepdim=True)
            sigma = x.var(-1, keepdim=True, unbiased=False)
            x = (x - mu) / torch.sqrt(sigma + 1e-5)
            x = x * self.weight + self.bias
        return x.permute(0, 3, 1, 2)


class CGLU(nn.Module):
    """Convolutional Gated Linear Unit (CGLU) in the ECIE encoder block."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer(),
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x


class PBDConv(nn.Module):
    """Prior-guided Boundary Detail Convolution branch in DAHB."""

    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=(1, 3), padding=(0, 1), groups=dim, bias=False)
        self.conv_v = nn.Conv2d(dim, dim, kernel_size=(3, 1), padding=(1, 0), groups=dim, bias=False)
        self.conv_s = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.fuse = nn.Conv2d(dim * 3, dim, kernel_size=1, bias=False)
        self.prior_proj = nn.Conv2d(1, dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        gray = x.mean(dim=1, keepdim=True)
        grad_x = F.pad((gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs(), (0, 1, 0, 0))
        grad_y = F.pad((gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs(), (0, 0, 0, 1))
        boundary_prior = torch.sigmoid(self.prior_proj(grad_x + grad_y))

        horizontal_detail = self.conv_h(x)
        vertical_detail = self.conv_v(x)
        spatial_detail = self.conv_s(x)
        detail = self.act(self.fuse(torch.cat([horizontal_detail, vertical_detail, spatial_detail], dim=1)))
        return detail * (1.0 + boundary_prior)


class PTAttention(nn.Module):
    """Pooling Transposition Attention branch in DAHB."""

    def __init__(self, dim, heads=4):
        super().__init__()
        if heads < 4:
            raise ValueError("PTAttention requires heads >= 4 to generate Q, K, V, and Z features.")
        self.heads = heads
        self.qkvz = nn.Conv2d(dim // 2, (dim // 4) * self.heads, 1, padding=0)
        self.pool_q = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.pool_k = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        b, c, h, w = x.shape
        qkvz = self.act(self.qkvz(x))
        qkvz = qkvz.reshape(b, self.heads, c // 2, h, w)

        q = torch.sum(qkvz[:, :-3, :, :, :], dim=1)
        k = qkvz[:, -3, :, :, :]
        q = self.pool_q(q)
        k = self.pool_k(k)

        v = qkvz[:, -2, :, :, :].flatten(2)
        z = qkvz[:, -1, :, :, :]
        qk = torch.matmul(q.flatten(2), k.flatten(2).transpose(1, 2))
        qk = torch.softmax(qk, dim=1).transpose(1, 2)
        attention = torch.matmul(qk, v).reshape(b, c // 2, h, w)
        return attention, z


class DCConv(nn.Module):
    """Directional Context Convolution branch that produces the Z feature in DAHB."""

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.horizontal_stripe = nn.Conv2d(
            dim, dim, kernel_size=(1, kernel_size), padding=(0, padding), groups=dim, bias=False
        )
        self.vertical_stripe = nn.Conv2d(
            dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim, bias=False
        )
        self.gate = nn.Sequential(nn.Conv2d(dim * 2, dim, kernel_size=1), nn.Sigmoid())
        self.fuse = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        horizontal_context = self.horizontal_stripe(x)
        vertical_context = self.vertical_stripe(x)
        context = torch.cat([horizontal_context, vertical_context], dim=1)
        anchor_gate = self.gate(context)
        return self.act(self.fuse(context)) * anchor_gate


class DAHB(nn.Module):
    """Dual-Path Attention Hybrid Block with PBDConv, PTAttention, and DCConv."""

    def __init__(self, dim, heads=4, stripe_kernel=7):
        super().__init__()
        self.pbdconv = PBDConv(dim // 2)
        self.pt_attention = PTAttention(dim, heads=heads)
        self.dcconv = DCConv(dim // 4, kernel_size=stripe_kernel)

    def forward(self, x):
        c = x.shape[1]
        x_lo, x_gl = torch.split(x, [c // 2, c // 2], dim=1)
        x_lo = self.pbdconv(x_lo)
        a_att, z = self.pt_attention(x_gl)
        z = self.dcconv(z)
        return torch.cat([x_lo, z, a_att], dim=1)


class EncoderBlock(nn.Module):
    """Encoder block composed of DAHB and CGLU, matching the ECIE network diagram."""

    def __init__(self, dim, drop_path=0.1, mlp_ratio=4, heads=4):
        super().__init__()
        self.layer_norm1 = LayerNorm(dim, "BiasFree")
        self.layer_norm2 = LayerNorm(dim, "BiasFree")
        self.dahb = DAHB(dim, heads=heads)
        self.cglu = CGLU(dim)

    def forward(self, x):
        inp_copy = x
        x = self.layer_norm1(inp_copy)
        x = self.dahb(x)
        out = x + inp_copy
        x = self.layer_norm2(out)
        x = self.cglu(x)
        return out + x


class ECIE(C2f):
    """ECIE module: C2f-style encoder with repeated Encoder Blocks."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(EncoderBlock(self.c) for _ in range(n))
