import torch
import torch.nn as nn
from einops import rearrange
from spikingjelly.activation_based import layer,neuron,functional,base
import torch.nn.functional as F

class spLayerNorm(nn.LayerNorm, base.StepModule):
    def __init__(self, out_channels, step_mode='m') -> None:
        super().__init__(out_channels)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, :], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, in_features, num_heads):
        """
        初始化函数
        :param in_features: 输入特征的维度
        :param num_heads: 多头注意力的头数
        """
        super(MultiHeadAttention, self).__init__()

        self.in_features = in_features
        self.num_heads = num_heads

        # 定义 Q、K、V 的线性变换
        self.q_linear = layer.Linear(in_features, in_features)
        self.k_linear = layer.Linear(in_features, in_features)
        self.v_linear = layer.Linear(in_features, in_features)

        # 定义输出的线性变换
        self.out_linear = layer.Linear(in_features, in_features)

    def forward(self, x):
        """
        前向传播函数
        :param x: 输入张量，形状为 [batch_size, seq_len, in_features]
        :return: 多头注意力机制的输出，形状同输入张量
        """
        t , batch_size, seq_len, in_features = x.size()

        # 将输入张量通过线性变换得到 Q、K、V
        q = self.q_linear(x)
        # print(q.shape)
        k = self.k_linear(x)
        v = self.v_linear(x)

        # 将 Q、K、V 分别拆分成多个头
        q = q.view(t*batch_size * self.num_heads, seq_len, in_features // self.num_heads)
        k = k.view(t*batch_size * self.num_heads, seq_len, in_features // self.num_heads)
        v = v.view(t*batch_size * self.num_heads, seq_len, in_features // self.num_heads)

        # 计算注意力得分
        scores = torch.bmm(q, k.transpose(1, 2)) / (in_features // self.num_heads) ** 0.5
        attn_weights = F.softmax(scores, dim=-1)

        # 对 V 应用注意力权重
        attn_output = torch.bmm(attn_weights, v)
        attn_output = attn_output.view(t, batch_size, seq_len, in_features)

        # 对输出进行线性变换
        output = self.out_linear(attn_output)

        return output

class ViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, num_heads, mlp_ratio):
        super(ViTBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        # 定义3个线性层，用于将输入的序列进行线性变换
        self.linear1 = layer.Linear(in_channels*self.patch_size*self.patch_size, out_channels)
        self.linear2 = layer.Linear(out_channels, out_channels)
        self.linear3 = layer.Linear(out_channels, out_channels)

        # 定义一个多头注意力层
        self.self_attention = MultiHeadAttention(in_features=out_channels, num_heads=num_heads)

        # 定义一个MLP层
        hidden_dim = int(out_channels * mlp_ratio)
        self.mlp = nn.Sequential(
            layer.Linear(out_channels, hidden_dim),
            neuron.IFNode(),
            layer.Linear(hidden_dim, out_channels)
        )

        self.layer_norm1 = spLayerNorm(out_channels)
        self.layer_norm2 = spLayerNorm(out_channels)

        # 将图像分为若干个大小为patch_size * patch_size的小块，并将每个小块展开成一个序列

        self.seq_length = 0

    def forward(self, x):
        x_shape = x.shape
        self.seq_length = (x_shape[-1] // self.patch_size) ** 2
        # 将输入的图像分为若干个大小为patch_size * patch_size的小块，并将每个小块展开成一个序列
        # print(x.shape)
        x = rearrange(x, 't b c (h p1) (w p2) ->t b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)
        # print(x.shape)

        # 将序列进行线性变换，并添加位置编码
        x = self.linear1(x)
        # print(x.shape)
        x = x + self.positional_encoding(x)
        x = self.layer_norm1(x)

        # 进行多头注意力
        attn_output = self.self_attention(x)
        x = x + attn_output
        x = self.layer_norm2(x)

        # 进行MLP
        mlp_output = self.mlp(x)
        x = x + mlp_output

        # 将序列重新组合成图像形状
        x = rearrange(x, 't b (h w) c ->t b c h w', h=int(x_shape[-1] / self.patch_size))

        return x

    def positional_encoding(self, x):
        # 创建表示位置的张量，形状为 [seq_length]
        pos = torch.arange(0, self.seq_length, device=x.device).float()

        # 创建表示不同维度的张量，形状为 [out_channels // 2]
        dim = torch.arange(0, self.out_channels, 2, device=x.device).float() / self.out_channels

        # 创建一个零张量，形状为 [seq_length, out_channels]
        pos_enc = torch.zeros(self.seq_length, self.out_channels, device=x.device)

        # 根据位置和维度计算正弦和余弦值
        pos_enc[:, 0::2] = torch.sin(pos[:, None] / (10000 ** dim[None, :]))
        pos_enc[:, 1::2] = torch.cos(pos[:, None] / (10000 ** dim[None, :]))

        # 将位置编码张量重复 batch_size 次，并添加一个维度
        return pos_enc[None, :, :].repeat(x.shape[1], 1, 1)


if __name__ == '__main__':
    # 定义一个随机张量作为输入
    x = torch.randn(4, 1, 128, 64, 64)

    # 定义一个ViTBlock模块
    vit_block = ViTBlock(in_channels=128, out_channels=64, patch_size=16, num_heads=8, mlp_ratio=2)
    functional.set_step_mode(vit_block, step_mode='m')

    # 进行前向传递
    output = vit_block(x)

    # 打印输出张量的形状
    print(output.shape)