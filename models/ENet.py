from audioop import bias
from turtle import down
from cv2 import dilate
import torch
import torch.nn as nn
from .enet_modules import InitialBlock, DDBottleNeck, ABottleNeck, UBottleNeck

class ENet(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.C = C  # number of classes

        # The Initial block
        self.initial = InitialBlock()

        # The first bottleneck
        self.b10 = DDBottleNeck(in_channels=16, out_channels=64, dilation=1, downsampling=True,  p=0.01)
        self.b11 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b12 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b13 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)
        self.b14 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False, p=0.01)

        # The second bottleneck
        self.b20 = DDBottleNeck(in_channels=64, out_channels=128, dilation=1, downsampling=True)
        self.b21 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b22 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b23 = ABottleNeck(in_channels=128, out_channels=128)

        self.b24 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b25 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b26 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b27 = ABottleNeck(in_channels=128, out_channels=128)
        self.b28 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The third bottleneck
        self.b31 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b32 = DDBottleNeck(in_channels=128, out_channels=128, dilation=2, downsampling=False)
        self.b33 = ABottleNeck(in_channels=128, out_channels=128)

        self.b34 = DDBottleNeck(in_channels=128, out_channels=128, dilation=4, downsampling=False)
        self.b35 = DDBottleNeck(in_channels=128, out_channels=128, dilation=1, downsampling=False)
        self.b36 = DDBottleNeck(in_channels=128, out_channels=128, dilation=8, downsampling=False)
        self.b37 = ABottleNeck(in_channels=128, out_channels=128)
        self.b38 = DDBottleNeck(in_channels=128, out_channels=128, dilation=16, downsampling=False)

        # The fourth bottleneck
        self.b40 = UBottleNeck(in_channels=128, out_channels=64)
        self.b41 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b42 = DDBottleNeck(in_channels=64, out_channels=64, dilation=1, downsampling=False)
        self.b50 = UBottleNeck(in_channels=64, out_channels=16)
        self.b51 = DDBottleNeck(in_channels=16, out_channels=16, dilation=1, downsampling=False)

        self.fullconv = nn.ConvTranspose2d(in_channels=16, out_channels=self.C, kernel_size=3, \
                                            stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        # The initial block
        x = self.initial(x)

        # The first bottleneck
        x, i1 = self.b10(x)
        x = self.b11(x)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)

        # The second bottleneck
        x, i2 = self.b20(x)
        x = self.b21(x)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)

        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)

        # The fourth bottleneck
        x = self.b40(x, i2)
        x = self.b41(x)
        x = self.b42(x)

        # The fifth bottleneck
        x = self.b50(x, i1)
        x = self.b51(x)

        # Final ConvTranspose Layer
        x = self.fullconv(x)

        return x
        