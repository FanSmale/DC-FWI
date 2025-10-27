# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@author: Zeng Zifei
"""

import torch
import torch.nn.functional as F
from BaseClass import *


class UnetUp(nn.Module):
    def __init__(self, in_channel, out_channel, is_deconv=True, active_func=nn.ReLU(inplace=True)):
        '''
        UpSampling Unit
        [Affiliated with FCNVMB]

        :param in_channel:      Number of channels of input
        :param out_channel:     Number of channels of output
        :param is_deconv:    Whether to use deconvolution
        '''
        super(UnetUp, self).__init__()
        self.conv = Conv2(in_channel, out_channel, True, active_func)
        # Transposed convolution
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2):
        """

        :param inputs1:      Layer of the selected coding area via skip connection
        :param inputs2:      Current network layer based on network flows
        :return:
        """
        outputs2 = self.up(inputs2)
        offset1 = (outputs2.size()[2] - inputs1.size()[2])
        offset2 = (outputs2.size()[3] - inputs1.size()[3])
        padding = [offset2 // 2, (offset2 + 1) // 2, offset1 // 2, (offset1 + 1) // 2]

        # Skip and concatenate
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class FCNVMB(nn.Module):
    def __init__(self, n_classes, in_channels, is_deconv, is_bn):
        '''
        Network architecture of FCNVMB

        :param n_classes:       Number of channels of output (any single decoder)
        :param in_channels:     Number of channels of network input
        :param is_deconv:       Whether to use deconvolution
        :param is_bn:    Whether to use BN
        '''
        super(FCNVMB, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.is_deconv = is_deconv
        self.is_bn = is_bn

        filters = [64, 128, 256, 512, 1024]

        self.down1 = DownSampling(self.in_channels, filters[0], self.is_bn)
        self.down2 = DownSampling(filters[0], filters[1], self.is_bn)
        self.down3 = DownSampling(filters[1], filters[2], self.is_bn)
        self.down4 = DownSampling(filters[2], filters[3], self.is_bn)
        self.center = Conv2(filters[3], filters[4], self.is_bn)
        self.up4 = UnetUp(filters[4], filters[3], self.is_deconv)
        self.up3 = UnetUp(filters[3], filters[2], self.is_deconv)
        self.up2 = UnetUp(filters[2], filters[1], self.is_deconv)
        self.up1 = UnetUp(filters[1], filters[0], self.is_deconv)
        self.final = nn.Conv2d(filters[0], self.n_classes, 1)

    def forward(self, inputs):
        """

        :param inputs:          Input Image
        :param label_dsp_dim:   Size of the network output image (velocity model size)
        :return:
        """
        label_dsp_dim = [201, 301]
        down1 = self.down1(inputs)  # Tensor:{x,64,200,151}
        down2 = self.down2(down1)  # Tensor:{x,128,100,76}
        down3 = self.down3(down2)  # Tensor:{x,256,50,38}
        down4 = self.down4(down3)  # Tensor:{x,512,25,19}
        center = self.center(down4)  # Tensor:{x,1024,25,19}
        up4 = self.up4(down4, center)  # Tensor:{x,512,50,38}
        up3 = self.up3(down3, up4)  # Tensor:{x,256,100,76}
        up2 = self.up2(down2, up3)  # Tensor:{x,128,200,152}
        up1 = self.up1(down1, up2)  # Tensor:{x,64,400,304}
        up1 = up1[:, :, 1:1 + label_dsp_dim[0], 1:1 + label_dsp_dim[1]].contiguous()  # Tensor:{x,64,201,301}

        return self.final(up1)


if __name__ == '__main__':

    # FCNVMB
    x = torch.zeros((5, 29, 400, 301))
    model = FCNVMB(n_classes=1, in_channels=29, is_deconv=True, is_bn=True)
    out = model(x)
    print("out: ", out.size())

    # model = FCNVMB(n_classes=1, in_channels=29, is_deconv=True, is_bn=True)
    # cuda_available = torch.cuda.is_available()
    # print(cuda_available)
    # device = torch.device("cuda" if cuda_available else "cpu")
    # model.to(device)
    #
    # from torchsummary import summary
    # summary(model, input_size=[(29, 400, 301)])