import os

import torch
import torch.nn as nn
import spikingjelly
from spikingjelly.activation_based import layer, neuron, functional, base


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class AdaptiveMaxPool2d(nn.AdaptiveMaxPool2d, base.StepModule):
    def __init__(self, output_size, step_mode='m') -> None:
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class CBAM(nn.Module):
    def __init__(self, in_channel):
        super(CBAM, self).__init__()

        self.Cam = ChannelAttentionModul(in_channel=in_channel)  # 通道注意力模块
        self.Sam = SpatialAttentionModul(in_channel=in_channel)  # 空间注意力模块

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        functional.reset_net(self)
        # x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
        x = self.Cam(x)
        x = self.Sam(x)
        return x


class ChannelAttentionModul(nn.Module):
    # 通道注意力模块
    def __init__(self, in_channel, r=0.5):  # channel为输入的维度, r为全连接层缩放比例->控制中间层个数
        super(ChannelAttentionModul, self).__init__()
        # 全局最大池化
        self.MaxPool = AdaptiveMaxPool2d(1)

        self.fc_MaxPool = nn.Sequential(
            layer.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            neuron.IFNode(),
            layer.Linear(int(in_channel * r), in_channel),
            neuron.IFNode(),
        )

        # 全局均值池化
        self.AvgPool = layer.AdaptiveAvgPool2d(1)

        self.fc_AvgPool = nn.Sequential(
            layer.Linear(in_channel, int(in_channel * r)),  # int(channel * r)取整数, 中间层神经元数至少为1, 如有必要可设为向上取整
            neuron.IFNode(),
            layer.Linear(int(in_channel * r), in_channel),
            neuron.IFNode(),
        )

        # 激活函数
        self.sigmoid = neuron.IFNode()

    def forward(self, x):
        # 1.最大池化分支
        max_branch = self.MaxPool(x)
        # 送入MLP全连接神经网络, 得到权重
        max_branch_shape = max_branch.shape
        max_in = max_branch.view(*max_branch_shape[:-3], -1)
        # print(max_in.shape)
        max_weight = self.fc_MaxPool(max_in)
        # print(max_weight.shape)

        # 2.全局池化分支
        avg_branch = self.AvgPool(x)
        # 送入MLP全连接神经网络, 得到权重
        avg_branch_shape = avg_branch.shape
        avg_in = avg_branch.view(*avg_branch_shape[:-3], -1)
        # print(avg_in.shape)
        avg_weight = self.fc_AvgPool(avg_in)
        # print(avg_weight.shape)

        # MaxPool + AvgPool 激活后得到权重weight
        weight = max_weight + avg_weight
        # print(weight.shape)
        weight = self.sigmoid(weight)
        # print('weight.shape:',weight.shape)
        # print('x.shape:',x.shape)

        # 将维度为b, c的weight, reshape成b, c, 1, 1 与 输入x 相乘
        t, h, w = weight.shape
        # 通道注意力Mc
        Mc = torch.reshape(weight, (t, h, w, 1, 1))

        # 乘积获得结果
        x = Mc * x
        # print(x.shape)

        return x


class SpatialAttentionModul(nn.Module):  # 空间注意力模块
    def __init__(self, in_channel):
        super(SpatialAttentionModul, self).__init__()
        self.conv = layer.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = neuron.IFNode()

    def forward(self, x):
        # x维度为 [N, C, H, W] 沿着维度C进行操作, 所以dim=1, 结果为[N, H, W]
        MaxPool = torch.max(x, dim=-3).values  # torch.max 返回的是索引和value， 要用.values去访问数值
        AvgPool = torch.mean(x, dim=-3)
        # print('MaxPool:',MaxPool.shape)
        # print('AvgPool:',AvgPool.shape)
        # print(MaxPool)
        # print(AvgPool)

        # 增加维度, 变成 [N, 1, H, W]
        MaxPool = torch.unsqueeze(MaxPool, dim=-3)
        AvgPool = torch.unsqueeze(AvgPool, dim=-3)
        # print('MaxPool:',MaxPool.shape)
        # print('AvgPool:',AvgPool.shape)

        # 维度拼接 [N, 2, H, W]
        x_cat = torch.cat((MaxPool, AvgPool), dim=2)  # 获得特征图
        # print(x_cat.shape)

        # 卷积操作得到空间注意力结果

        x_out = self.conv(x_cat)
        Ms = self.sigmoid(x_out)

        # 与原图通道进行乘积
        x = Ms * x

        return x


# 测试
if __name__ == '__main__':
    inputs = torch.randn(4, 2, 3, 64, 64)
    # model = CBAM(in_channel=3)  # CBAM模块, 可以插入CNN及任意网络中, 输入特征图in_channel的维度
    # model = ChannelAttentionModul(in_channel=3)
    model = SpatialAttentionModul(in_channel=3)
    # print(model)
    outputs = model(inputs)
    print("输入维度:", inputs.shape)
    print("输出维度:", outputs.shape)
