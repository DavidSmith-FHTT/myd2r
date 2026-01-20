import torch
import torch.nn as nn


class Router(nn.Module):
    """
    return: 路径概率向量 p_m^(n-1)
    """

    def __init__(self, num_out_path, embed_size, hid, temperature=1.0):
        """
        Args:
            num_out_path: 路由的输出维度
            temperature: softmax 温度系数
        """
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.temperature = float(temperature)
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

        return torch.softmax(logits, dim=-1)
