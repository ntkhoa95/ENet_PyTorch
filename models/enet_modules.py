from audioop import bias
from numpy import indices
import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_channels=3, out_channels=13):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.activation = nn.PReLU(16)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.norm(main)
        
        side = self.max_pool(x)

        x = torch.cat((main, side), dim=1)
        x = self.activation(x)
        return x

class DDBottleNeck(nn.Module):
    """
    Bottleneck 2.x for dowmsampling and dilated types
    """
    def __init__(self, in_channels, out_channels, dilation, downsampling, activation='prelu', down_ratio=4, p=0.1):
        super().__init__()

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.dilation     = dilation
        self.downsampling = downsampling

        if self.downsampling: # if bottleneck 2.x with downsampling
            self.stride = 2
            self.down_channels = int(in_channels // down_ratio)
        else:
            self.stride = 1
            self.down_channels = int(out_channels // down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        # Main MaxPooling
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        
        # Side Conv 1x1
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, stride=1, \
                                padding=0, bias=False, dilation=1)
        self.activation1 = self.activation
        self.conv2 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3, stride=self.stride, \
                                padding=self.dilation, bias=True, dilation=self.dilation)
        self.activation2 = self.activation
        self.conv3 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, stride=1, \
                                padding=0, bias=False, dilation=1)
        self.activation3 = self.activation
        self.norm1 = nn.BatchNorm2d(self.down_channels)
        self.norm2 = nn.BatchNorm2d(self.down_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        self.dropout = nn.Dropout2d(p=p)

    def forward(self, x):
        batch_size = x.size()[0]
        x_main = x

        # Side branch
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.dropout(x)

        # Main branch
        if self.downsampling:
            x_main, indices = self.max_pool(x_main)

        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                temp = temp.cuda()
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x = x + x_main
        x = self.activation3(x)

        if self.downsampling:
            return x, indices
        else:
            return x

class ABottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='prelu', down_ratio=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.down_channels = int(self.in_channels / down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, stride=1,\
                                padding=0, bias=False)
        self.activation1 = self.activation

        self.conv21 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(1, 5), stride=1,\
                                padding=(0, 2), bias=False)
        self.conv22 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=(5, 1), stride=1,\
                                padding=(2, 0), bias=False)
        self.activation2 = self.activation

        self.conv3 = nn.Conv2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, stride=1,\
                                padding=0, bias=False)
        self.activation3 = self.activation

        self.norm1 = nn.BatchNorm2d(self.down_channels)
        self.norm2 = nn.BatchNorm2d(self.down_channels)
        self.norm3 = nn.BatchNorm2d(self.out_channels)

        self.drop_out = nn.Dropout2d(p=0.1)

    def forward(self, x):
        batch_size = x.size()[0]
        x_main = x

        # Side branch
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.conv21(x)
        x = self.conv22(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.conv3(x)
        
        x = self.drop_out(x)
        x = self.norm3(x)

        # Main branch
        if self.in_channels != self.out_channels:
            out_shape = self.out_channels - self.in_channels
            temp = torch.zeros((batch_size, out_shape, x.shape[2], x.shape[3]))
            if torch.cuda.is_available():
                temp = temp.cuda()
            x_main = torch.cat((x_main, temp), dim=1)

        # Side + Main
        x += x_main
        x = self.activation3(x)
        return x

class UBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, activation='prelu', down_ratio=4):
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.down_channels = int(self.in_channels / down_ratio)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'mish':
            self.activation = nn.Mish()

        self.main_conv   = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        
        self.convt1      = nn.ConvTranspose2d(in_channels=self.in_channels, out_channels=self.down_channels, kernel_size=1, padding=0, bias=False)
        self.activation1 = self.activation
        self.convt2      = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.down_channels, kernel_size=3, stride=2, padding=1,\
                                            output_padding=1, bias=False)
        self.activation2 = self.activation
        self.convt3      = nn.ConvTranspose2d(in_channels=self.down_channels, out_channels=self.out_channels, kernel_size=1, padding=0, bias=False)
        self.activation3 = self.activation

        self.norm1       = nn.BatchNorm2d(self.down_channels)
        self.norm2       = nn.BatchNorm2d(self.down_channels)
        self.norm3       = nn.BatchNorm2d(self.out_channels)

        self.unpool     = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.drop_out    = nn.Dropout2d(p=0.1)

    def forward(self, x, indices):
        x_main = x

        # Side branch
        x = self.convt1(x)
        x = self.norm1(x)
        x = self.activation1(x)

        x = self.convt2(x)
        x = self.norm2(x)
        x = self.activation2(x)

        x = self.convt3(x)
        x = self.norm3(x)

        x = self.drop_out(x)

        # Main branch
        x_main = self.main_conv(x_main)
        x_main = self.unpool(x_main, indices, output_size=x.size())

        # Concatenate
        x = x + x_main
        x  = self.activation3(x)

        return x