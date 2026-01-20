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
    """
    return: 路径概率向量 p_m^(n-1)
    """

    def __init__(self, num_out_path, embed_size, hid, temperature=1.0, top_k=None):
        """
        Args:
            num_out_path: 路由的输出维度
            temperature: softmax 温度系数
            top_k: 可选，top-k 稀疏化路径选择
        """
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.temperature = float(temperature)
        self.top_k = top_k
        self.pool = nn.Linear(embed_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, hid),
            nn.ReLU(True),
            nn.Linear(hid, num_out_path),
        )
        self.init_weights()

    def init_weights(self):
        """
        初始化路由层参数，保持 softmax 路由的可控性。
        """
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, x):  # (bsz, L, 768) -> (bsz, 4)
        """
        输入x通过注意力池化得到全局表示，然后经 MLP 转换为路径 logits，
        最后通过 softmax 得到可解释的路径概率。
        """
        attn_scores = self.pool(x).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=-2)
        logits = self.mlp(pooled)
        logits = logits / max(self.temperature, 1e-6)

        if self.top_k is not None and self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, topk_idx, topk_vals)

        return torch.softmax(logits, dim=-1)
