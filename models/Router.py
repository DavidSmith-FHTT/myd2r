import torch
import torch.nn as nn
import torch.nn.functional as F


def activateFunc(x):
    """
    Tanh  ->  ReLU
    """
    x = torch.tanh(x)
    return F.relu(x)


# class Router(nn.Module):
#     """
#     return: 路径概率向量 p_m^(n-1)
#     """
#     def __init__(self, num_out_path, embed_size, hid):
#         """
#         Args:
#             num_out_path: 路由的输出维度
#         """
#         super(Router, self).__init__()
#         self.num_out_path = num_out_path
#         self.mlp = nn.Sequential(nn.Linear(embed_size, hid),
#                                  nn.ReLU(True),
#                                  nn.Linear(hid, num_out_path))
#         self.init_weights()
#
#     def init_weights(self):
#         """
#         将MLP中第三层（self.mlp[2]）的偏置值全部设置为1.5。
#         """
#         self.mlp[2].bias.data.fill_(1.5)
#
#     def forward(self, x):  # (bsz, L, 768) -> (bsz, 4)
#         """
#         输入x首先通过mean(-2)操作在第二维度（通常是序列的长度）上求均值，然后通过MLP进行转换，最后通过activateFunc函数应用激活。
#         """
#         x = x.mean(-2)
#         x = self.mlp(x)
#         soft_g = activateFunc(x)
#         return soft_g

class Router(nn.Module):
    def __init__(self, num_out_path, embed_size, hidden_size):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads=8)
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_out_path)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [bsz, L, dim]
        x_t = x.transpose(0, 1)  # [L, bsz, dim]
        attn_out, _ = self.attention(x_t, x_t, x_t)
        attn_out = attn_out.transpose(0, 1)  # [bsz, L, dim]

        # 全局池化
        pooled = torch.mean(attn_out, dim=1)  # [bsz, dim]

        hidden = F.relu(self.fc1(pooled))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)

        return F.softmax(logits, dim=-1)
