# -*- coding: utf-8 -*-
"""
@Time: 2024/6/28

@Author : Zeng Zifei
"""

"""
改动如下：基于DenseNetwork70_2
1、2、3是DC_Net70_2_v2的修改内容
1.底层降维借鉴kaggle竞赛的网络ConvNeXt修改部分：Stem：两个卷积，尺寸到72*72，这个会导致flops上升
2.编码器还是到4*4
3.将解码器的反卷积+卷积改为：上采样（用的包） + 卷积，为了降低flops
4.bottleneck层中采用深度可分离卷积去替代3×3普通卷积
参数量下降2.9M左右，flops下降了0.6左右：params: 4.152 M，flops: 3.390 G
"""

import torch
import torch.nn as nn
from math import ceil
import torch.nn.functional as F
from thop import profile
from monai.networks.blocks import UpSample


class Stem(nn.Module):
    def __init__(self, in_fea):
        super(Stem,self).__init__()

        self.conv1 = nn.Conv2d(in_fea, 32, kernel_size=4, stride=(4, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=(4, 1), padding=(0, 1))
        # 组合原始stem和新conv层
        self.stem= nn.Sequential(
            nn.ReflectionPad2d((1, 1, 80, 80)),
            self.conv1,
            self.conv2,
        )


    def forward(self, x):
        return self.stem(x)


# 对于一次卷积操作进行封装:conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
class ConvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1, dropout=None):
        super(ConvBlock,self).__init__()
        # 卷积层，卷积：有输入信号、滤波器即提取有意义特征的作用。
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]
        # 如果网络输入具有零均值、单位方差和去相关性，则深层网络的收敛速度会加快。
        # 所以 使中间层的输出具有这些属性也是有利的。
        # 批量归一化层 用于在每次迭代时，对送到网络中的中间层的数据子集在输出时进行归一化。

        layers.append(nn.BatchNorm2d(out_fea))
        # 激活函数层，LeaklyReLU通过将x的非常小的线性分量 给予 负输入αx来调整负值的零梯度问题，此外也可以扩大函数y的范围。
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout:
            # 赋值对象是彩色的图像数据（N,C,H,W）的一个通道里的每一个数据，
            # 即输入为 Input: (N,C,H,W) 时，对每一个通道维度C按概率赋值为0。输出和输入的形状一致
            # 0.8：元素置零的概率
            layers.append(nn.Dropout2d(0.8))
        # 输入也可以是list,然后输入的时候用*来引用，否则会报错 TypeError: list is not a Module subclass
        # *作用在形参上，代表这个位置接收任意多个非关键字参数，转化成元组方式；
        # *作用在实参上，代表的是将输入迭代器拆成一个个元素。
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x：输入图像
        return self.layers(x)


# 反卷积->归一化->激活函数LeakyReLU
class DeconvBlock(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=2, stride=2, padding=0, output_padding=0):
        super(DeconvBlock, self).__init__()
        # 反卷积操作
        layers = [nn.ConvTranspose2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)]
        # layers = nn.UpsamplingBilinear2d(scale_factor=2)

        layers.append(nn.BatchNorm2d(out_fea))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# 卷积->归一化->激活函数Tanh()
class ConvBlock_Tanh(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_Tanh, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]

        layers.append(nn.BatchNorm2d(out_fea))
        # （双曲正切）对数据进行激活，将元素调整到区间(-1,1)内，因此将强负输入映射为负值。
        # 仅将接近零的值映射到接近零的输出，在某种程度上解决了“vanishing gradients”问题。
        layers.append(nn.Tanh())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvBlock_Sigmoid(nn.Module):
    def __init__(self, in_fea, out_fea, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_Sigmoid, self).__init__()
        layers = [nn.Conv2d(in_channels=in_fea, out_channels=out_fea, kernel_size=kernel_size, stride=stride, padding=padding)]

        layers.append(nn.BatchNorm2d(out_fea))
        # （双曲正切）对数据进行激活，将元素调整到区间(-1,1)内，因此将强负输入映射为负值。
        # 仅将接近零的值映射到接近零的输出，在某种程度上解决了“vanishing gradients”问题。
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


'''-------------定义Dense Block模块-----------------------------'''

'''---（1）构造Dense Block内部结构---'''


# bottleneck：BN+ReLU+1x1Conv+BN+ReLU+3x3Conv
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate=0):
        """

        :param num_input_features: 输入通道数
        :param growth_rate: 增长率k，设为32
        :param bn_size:
        :param drop_rate:
        """
        super(_DenseLayer, self).__init__()
        self.drop_rate = drop_rate
        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),# inplace=True:不创建新的对象.真接对原始对象进行修改;
            # growth_rate：增长率。一层产生多少个特征图
            nn.Conv2d(in_channels=num_input_features, out_channels=bn_size * growth_rate,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            # 深度可分离卷积替代3×3普通卷积
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=bn_size * growth_rate,
                      kernel_size=3, stride=1, padding=1, groups=bn_size * growth_rate, bias=False),  # 深度卷积
            nn.Conv2d(in_channels=bn_size * growth_rate, out_channels=growth_rate,
                      kernel_size=1, stride=1, padding=0, bias=False),  # 逐点卷积
        )
        self.dropout = nn.Dropout(p=self.drop_rate)

    def forward(self, x):
        new_features = self.dense_layer(x)
        if self.drop_rate > 0:
            new_features = self.dropout(new_features)
        return torch.cat([x, new_features], 1)

'''---（2）构造Dense Block模块---'''

class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate=0):
        super(DenseBlock, self).__init__()
        layers = []
        # 随着layer层数的增加，每增加一层，输入的特征图就增加一倍growth_rate
        for i in range(num_layers):
            layers.append(_DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

'''-------------构造Transition层-----------------------------'''


# 过渡层：BN+ReLU+1×1Conv+2×2AveragePooling，减少通道和尺寸
class _TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_TransitionLayer, self).__init__()
        self.transition_layer = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_input_features, out_channels=num_output_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.transition_layer(x)


# ------------------------#
# CBAM模块的Pytorch实现
# ------------------------#

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        # reduction代表缩放的比例，因为第一次全连接神经元个数较少
        mid_channel = channel // reduction

        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # w、h输出尺寸1*1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        # self.avg_pool(x)得到(b,c,1,1)
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out


class DenseNet(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 8], scale_factor: int = 2, upsample_mode: str = "deconv"):
        """
        k=32, blocks=[6, 12, 8]
        降维代码使用自己的，conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
        密集连接使用ReLU
        编码器仅下采样到4*4，没有到1*1
        解码器包含一次反卷积（反卷积->归一化->LeakyReLU）和一次卷积（conv2d -> nn.BatchNorm2d -> nn.LeakyReLU）
        最后使用Tanh，输入维度是32。

        :param init_channels: 初始通道数
        :param growth_rate: 增长率，即K=32
        :param blocks: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet, self).__init__()
        bn_size = 4
        # 修改dropout
        drop_rate = 0.2

        # ( , 64, 72, 72)
        self.convblock1 = Stem(5)

        blocks * 3
        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有8个DenseLayer, 执行DenseBlock（8,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（512,256）
        self.transition3 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))

        self.convblock5 = ConvBlock(256, 512, stride=2)

        self.upsample1 = UpSample(spatial_dims=2, in_channels=512, out_channels=256, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv1 = ConvBlock(256, 256)
        self.CBAM1 = CBAM(256)
        self.upsample2 = UpSample(spatial_dims=2, in_channels=256, out_channels=128, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv2 = ConvBlock(128, 128)
        self.upsample3 = UpSample(spatial_dims=2, in_channels=128, out_channels=64, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv3 = ConvBlock(64, 64)
        self.upsample4 = UpSample(spatial_dims=2, in_channels=64, out_channels=32, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv4 = ConvBlock(32, 32)

        # 裁剪输出层
        # self.deconv5 = ConvBlock_Tanh(32, 1)
        self.deconv5 = ConvBlock_Sigmoid(32, 1)


    def forward(self, x):
        x = self.convblock1(x)  # (None, 64, 72, 72)
        x = self.layer1(x)  # {Tensor:(None,256,72,72)}
        x = self.transition1(x)  # {Tensor:(None,128,36,36)}
        x = self.layer2(x)  # {Tensor:(None,512,36,36)}
        x = self.transition2(x)  # {Tensor:(None,256,18,18)}
        x = self.layer3(x)  # {Tensor:(None,512,18,18)}
        x = self.transition3(x)  # {Tensor:(None,256,9,9)}

        x = self.convblock5(x)  # (None, 512, 5, 5)

        # Decoder Part

        x = self.upsample1(x)  # (None, 256, 10, 10)
        x = self.deconv1(x)  # (None, 256, 10, 10)
        x = self.CBAM1(x)  # (None, 256, 10, 10)
        x = self.upsample2(x)  # (None, 128, 20, 20)
        x = self.deconv2(x)  # (None, 128, 20, 20)
        x = self.upsample3(x)  # (None, 64, 40, 40)
        x = self.deconv3(x)  # (None, 64, 40, 40)
        x = self.upsample4(x)  # (None, 32, 80, 80)
        x = self.deconv4(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70)
        x = self.deconv5(x)  # (None, 1, 70, 70)
        return x


class DenseNet_2(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 8], scale_factor: int = 2, upsample_mode: str = "deconv"):
        """
        k=32, blocks=[6, 12, 8]
        降维代码使用自己的，conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
        密集连接使用ReLU
        编码器仅下采样到4*4，没有到1*1
        解码器包含一次反卷积（反卷积->归一化->LeakyReLU）和一次卷积（conv2d -> nn.BatchNorm2d -> nn.LeakyReLU）
        最后使用Tanh，输入维度是32。

        :param init_channels: 初始通道数
        :param growth_rate: 增长率，即K=32
        :param blocks: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet_2, self).__init__()
        bn_size = 4
        # 修改dropout
        drop_rate = 0.2

        # ( , 64, 72, 72)
        self.convblock1 = Stem(5)

        blocks * 3
        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有8个DenseLayer, 执行DenseBlock（8,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（512,256）
        self.transition3 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))

        self.convblock5 = ConvBlock(256, 512, stride=2)

        self.upsample1 = UpSample(spatial_dims=2, in_channels=512, out_channels=256, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv1 = ConvBlock(256, 256)
        self.CBAM1 = CBAM(256)
        self.upsample2 = UpSample(spatial_dims=2, in_channels=256, out_channels=128, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv2 = ConvBlock(128, 128)
        self.upsample3 = UpSample(spatial_dims=2, in_channels=128, out_channels=64, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv3 = ConvBlock(64, 64)
        self.upsample4 = UpSample(spatial_dims=2, in_channels=64, out_channels=32, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv4 = ConvBlock(32, 32)

        # 裁剪输出层
        # self.deconv5 = ConvBlock_Tanh(32, 1)
        self.deconv5 = ConvBlock_Sigmoid(32, 1)


    def forward(self, x):
        x1 = self.convblock1(x)  # (None, 64, 72, 72)
        x2 = self.layer1(x1)  # {Tensor:(None,256,72,72)}
        x3 = self.transition1(x2)  # {Tensor:(None,128,36,36)}
        x4 = self.layer2(x3)  # {Tensor:(None,512,36,36)}
        x5 = self.transition2(x4)  # {Tensor:(None,256,18,18)}
        x6 = self.layer3(x5)  # {Tensor:(None,512,18,18)}
        x7 = self.transition3(x6)  # {Tensor:(None,256,9,9)}

        x8 = self.convblock5(x7)  # (None, 512, 5, 5)

        # Decoder Part

        y1 = self.upsample1(x8)  # (None, 256, 10, 10)
        y2 = self.deconv1(y1)  # (None, 256, 10, 10)
        y3 = self.CBAM1(y2)  # (None, 256, 10, 10)
        y4 = self.upsample2(y3)  # (None, 128, 20, 20)
        y5 = self.deconv2(y4)  # (None, 128, 20, 20)
        y6 = self.upsample3(y5)  # (None, 64, 40, 40)
        y7 = self.deconv3(y6)  # (None, 64, 40, 40)
        z1 = y7
        y8 = self.upsample4(y7)  # (None, 32, 80, 80)
        y9 = self.deconv4(y8)  # (None, 32, 80, 80)
        z2 = y9
        y10 = F.pad(y9, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70)
        y11 = self.deconv5(y10)  # (None, 1, 70, 70)
        # return x
        return z1, z2


class DenseNet_conv(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 8], scale_factor: int = 2, upsample_mode: str = "deconv"):
        """
        k=32, blocks=[6, 12, 8]
        降维代码使用自己的，conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
        密集连接使用ReLU
        编码器仅下采样到4*4，没有到1*1
        解码器包含一次反卷积（反卷积->归一化->LeakyReLU）和一次卷积（conv2d -> nn.BatchNorm2d -> nn.LeakyReLU）
        最后使用Tanh，输入维度是32。

        :param init_channels: 初始通道数
        :param growth_rate: 增长率，即K=32
        :param blocks: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet_conv, self).__init__()
        bn_size = 4
        # 修改dropout
        drop_rate = 0.2

        # ( , 64, 72, 72)
        self.convblock1 = Stem(5)

        blocks * 3
        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        self.convblock2_1 = ConvBlock(64, 128, stride=2)
        self.convblock2_2 = ConvBlock(128, 128)
        self.convblock3_1 = ConvBlock(128, 256, stride=2)
        self.convblock3_2 = ConvBlock(256, 256)
        self.convblock4_1 = ConvBlock(256, 512, stride=2)
        self.convblock4_2 = ConvBlock(512, 512)


        self.convblock5 = ConvBlock(512, 512, stride=2)

        self.upsample1 = UpSample(spatial_dims=2, in_channels=512, out_channels=256, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv1 = ConvBlock(256, 256)
        self.CBAM1 = CBAM(256)
        self.upsample2 = UpSample(spatial_dims=2, in_channels=256, out_channels=128, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv2 = ConvBlock(128, 128)
        self.upsample3 = UpSample(spatial_dims=2, in_channels=128, out_channels=64, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv3 = ConvBlock(64, 64)
        self.upsample4 = UpSample(spatial_dims=2, in_channels=64, out_channels=32, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv4 = ConvBlock(32, 32)

        # 裁剪输出层
        # self.deconv5 = ConvBlock_Tanh(32, 1)
        self.deconv5 = ConvBlock_Sigmoid(32, 1)


    def forward(self, x):
        x = self.convblock1(x)  # (None, 64, 72, 72)
        x = self.convblock2_1(x)  # {Tensor:(None,128,36,36)}
        x = self.convblock2_2(x)  # {Tensor:(None,128,36,36)}
        x = self.convblock3_1(x)  # {Tensor:(None,256,18,18)}
        x = self.convblock3_2(x)  # {Tensor:(None,256,18,18)}
        x = self.convblock4_1(x)  # {Tensor:(None,512,9,9)}
        x = self.convblock4_2(x)  # {Tensor:(None,512,9,9)}

        x = self.convblock5(x)  # (None, 512, 5, 5)

        # Decoder Part

        x = self.upsample1(x)  # (None, 256, 10, 10)
        x = self.deconv1(x)  # (None, 256, 10, 10)
        x = self.CBAM1(x)  # (None, 256, 10, 10)
        x = self.upsample2(x)  # (None, 128, 20, 20)
        x = self.deconv2(x)  # (None, 128, 20, 20)
        x = self.upsample3(x)  # (None, 64, 40, 40)
        x = self.deconv3(x)  # (None, 64, 40, 40)
        x = self.upsample4(x)  # (None, 32, 80, 80)
        x = self.deconv4(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70)
        x = self.deconv5(x)  # (None, 1, 70, 70)
        return x


class DenseNet_conv_2(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 8], scale_factor: int = 2, upsample_mode: str = "deconv"):
        """
        k=32, blocks=[6, 12, 8]
        降维代码使用自己的，conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
        密集连接使用ReLU
        编码器仅下采样到4*4，没有到1*1
        解码器包含一次反卷积（反卷积->归一化->LeakyReLU）和一次卷积（conv2d -> nn.BatchNorm2d -> nn.LeakyReLU）
        最后使用Tanh，输入维度是32。

        :param init_channels: 初始通道数
        :param growth_rate: 增长率，即K=32
        :param blocks: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet_conv_2, self).__init__()
        bn_size = 4
        # 修改dropout
        drop_rate = 0.2

        # ( , 64, 72, 72)
        self.convblock1 = Stem(5)

        blocks * 3
        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        self.convblock2_1 = ConvBlock(64, 128, stride=2)
        self.convblock2_2 = ConvBlock(128, 128)
        self.convblock3_1 = ConvBlock(128, 256, stride=2)
        self.convblock3_2 = ConvBlock(256, 256)
        self.convblock4_1 = ConvBlock(256, 512, stride=2)
        self.convblock4_2 = ConvBlock(512, 512)


        self.convblock5 = ConvBlock(512, 512, stride=2)

        self.upsample1 = UpSample(spatial_dims=2, in_channels=512, out_channels=256, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv1 = ConvBlock(256, 256)
        self.CBAM1 = CBAM(256)
        self.upsample2 = UpSample(spatial_dims=2, in_channels=256, out_channels=128, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv2 = ConvBlock(128, 128)
        self.upsample3 = UpSample(spatial_dims=2, in_channels=128, out_channels=64, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv3 = ConvBlock(64, 64)
        self.upsample4 = UpSample(spatial_dims=2, in_channels=64, out_channels=32, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv4 = ConvBlock(32, 32)

        # 裁剪输出层
        # self.deconv5 = ConvBlock_Tanh(32, 1)
        self.deconv5 = ConvBlock_Sigmoid(32, 1)


    def forward(self, x):
        x = self.convblock1(x)  # (None, 64, 72, 72)
        x = self.convblock2_1(x)  # {Tensor:(None,128,36,36)}
        x = self.convblock2_2(x)  # {Tensor:(None,128,36,36)}
        x = self.convblock3_1(x)  # {Tensor:(None,256,18,18)}
        x = self.convblock3_2(x)  # {Tensor:(None,256,18,18)}
        x = self.convblock4_1(x)  # {Tensor:(None,512,9,9)}
        x = self.convblock4_2(x)  # {Tensor:(None,512,9,9)}

        x8 = self.convblock5(x)  # (None, 512, 5, 5)

        # Decoder Part

        y1 = self.upsample1(x8)  # (None, 256, 10, 10)
        y2 = self.deconv1(y1)  # (None, 256, 10, 10)
        y3 = self.CBAM1(y2)  # (None, 256, 10, 10)
        y4 = self.upsample2(y3)  # (None, 128, 20, 20)
        y5 = self.deconv2(y4)  # (None, 128, 20, 20)
        y6 = self.upsample3(y5)  # (None, 64, 40, 40)
        y7 = self.deconv3(y6)  # (None, 64, 40, 40)
        z1 = y7
        y8 = self.upsample4(y7)  # (None, 32, 80, 80)
        y9 = self.deconv4(y8)  # (None, 32, 80, 80)
        z2 = y9
        y10 = F.pad(y9, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70)
        y11 = self.deconv5(y10)  # (None, 1, 70, 70)
        # return x
        return z1, z2


class DenseNet_withoutCBAM(nn.Module):
    def __init__(self, init_channels=64, growth_rate=32, blocks=[6, 12, 8], scale_factor: int = 2, upsample_mode: str = "deconv"):
        """
        k=32, blocks=[6, 12, 8]
        降维代码使用自己的，conv2d -> nn.BatchNorm2d -> nn.LeakyReLU
        密集连接使用ReLU
        编码器仅下采样到4*4，没有到1*1
        解码器包含一次反卷积（反卷积->归一化->LeakyReLU）和一次卷积（conv2d -> nn.BatchNorm2d -> nn.LeakyReLU）
        最后使用Tanh，输入维度是32。

        :param init_channels: 初始通道数
        :param growth_rate: 增长率，即K=32
        :param blocks: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet_withoutCBAM, self).__init__()
        bn_size = 4
        # 修改dropout
        drop_rate = 0.2

        # ( , 64, 72, 72)
        self.convblock1 = Stem(5)

        blocks * 3
        # 第一次执行特征的维度来自于前面的特征提取
        num_features = init_channels

        # 第1个DenseBlock有6个DenseLayer, 执行DenseBlock（6,64,32,4）
        self.layer1 = DenseBlock(num_layers=blocks[0], num_input_features=num_features, growth_rate=growth_rate, bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[0] * growth_rate
        # 第1个transition 执行 _TransitionLayer（256,128）
        self.transition1 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第1回合之后，第2个DenseBlock的输入的feature应该是：num_features = 128
        num_features = num_features // 2

        # 第2个DenseBlock有12个DenseLayer, 执行DenseBlock（12,128,32,4）
        self.layer2 = DenseBlock(num_layers=blocks[1], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[1] * growth_rate
        # 第2个transition 执行 _TransitionLayer（512,256）
        self.transition2 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))
        # num_features减少为原来的一半，执行第2回合之后，第3个DenseBlock的输入的feature应该是：num_features = 256
        num_features = num_features // 2

        # 第3个DenseBlock有8个DenseLayer, 执行DenseBlock（8,256,32,4）
        self.layer3 = DenseBlock(num_layers=blocks[2], num_input_features=num_features, growth_rate=growth_rate,
                                 bn_size=bn_size,
                                 drop_rate=drop_rate)
        num_features = num_features + blocks[2] * growth_rate
        # 第3个transition 执行 _TransitionLayer（512,256）
        self.transition3 = _TransitionLayer(num_input_features=int(num_features), num_output_features=int(num_features // 2))

        self.convblock5 = ConvBlock(256, 512, stride=2)

        self.upsample1 = UpSample(spatial_dims=2, in_channels=512, out_channels=256, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv1 = ConvBlock(256, 256)
        # self.CBAM1 = CBAM(256)
        self.upsample2 = UpSample(spatial_dims=2, in_channels=256, out_channels=128, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv2 = ConvBlock(128, 128)
        self.upsample3 = UpSample(spatial_dims=2, in_channels=128, out_channels=64, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv3 = ConvBlock(64, 64)
        self.upsample4 = UpSample(spatial_dims=2, in_channels=64, out_channels=32, scale_factor=scale_factor, mode=upsample_mode)
        self.deconv4 = ConvBlock(32, 32)

        # 裁剪输出层
        # self.deconv5 = ConvBlock_Tanh(32, 1)
        self.deconv5 = ConvBlock_Sigmoid(32, 1)


    def forward(self, x):
        x = self.convblock1(x)  # (None, 64, 72, 72)
        x = self.layer1(x)  # {Tensor:(None,256,72,72)}
        x = self.transition1(x)  # {Tensor:(None,128,36,36)}
        x = self.layer2(x)  # {Tensor:(None,512,36,36)}
        x = self.transition2(x)  # {Tensor:(None,256,18,18)}
        x = self.layer3(x)  # {Tensor:(None,512,18,18)}
        x = self.transition3(x)  # {Tensor:(None,256,9,9)}

        x = self.convblock5(x)  # (None, 512, 5, 5)

        # Decoder Part

        x = self.upsample1(x)  # (None, 256, 10, 10)
        x = self.deconv1(x)  # (None, 256, 10, 10)
        # x = self.CBAM1(x)  # (None, 256, 10, 10)
        x = self.upsample2(x)  # (None, 128, 20, 20)
        x = self.deconv2(x)  # (None, 128, 20, 20)
        x = self.upsample3(x)  # (None, 64, 40, 40)
        x = self.deconv3(x)  # (None, 64, 40, 40)
        x = self.upsample4(x)  # (None, 32, 80, 80)
        x = self.deconv4(x)  # (None, 32, 80, 80)
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)  # (None, 32, 70, 70)
        x = self.deconv5(x)  # (None, 1, 70, 70)
        return x

if __name__=='__main__':
    model = DenseNet()
    input = torch.zeros((1, 5, 1000, 70))
    out = model(input)
    print(out.shape)  # torch.Size([10, 1, 70, 70])
    # flops, params = profile(model, (input,))
    # print('flops:', flops, 'params:', params)
    # print('flops: %.3f M, params: %.3f M' % (flops / 1000000.0, params / 1000000.0))
    # print('flops: %.3f G, params: %.3f G' % (flops / 1000000000.0, params / 1000000000.0))