import torch
import torch.nn as nn


# ============================================================================
# Sentiment-Aware Router (已注释,用于消融实验对比)
# ============================================================================
class Router(nn.Module):
    """
    Attention-Weighted Router(注意力加权路由器) for Dynamic Path Selection.
    Uses attention pooling to weight sequence importance for routing decisions.

    Sentiment-Aware Router.
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
        输入 x 通过注意力池化得到全局表示，然后经 MLP 转换为路径 logits，
        最后通过 softmax 得到可解释的路径概率。
        核心目的:
        1. 信息聚合
        将序列中所有位置的信息压缩成一个固定长度的向量
        2. 重要性加权
        不是简单平均，而是根据注意力权重进行加权聚合：重要信息贡献更大,冗余信息贡献较小
        3. 维度统一
        为后续的MLP分类器提供固定维度的输入

        直观理解
        想象一个句子："这部电影非常精彩"
        传统平均：每个词权重相同
        注意力池化："精彩"权重最大，"非常"次之，"这部电影"权重较小
        这样得到的特征向量更能抓住句子的核心语义，为路由决策提供更准确的依据。
        """
        # 序列到标量的映射
        attn_scores = self.pool(x).squeeze(-1) 
        # 归一化为权重
        attn_weights = torch.softmax(attn_scores, dim=-1)  
        # 加权求和得到全局表示。将所有位置的特征聚合成一个向量
        pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=-2)
        # 路径 logits
        logits = self.mlp(pooled)
        logits = logits / max(self.temperature, 1e-6)

        return torch.softmax(logits, dim=-1)


# ============================================================================
# Soft Router (消融实验版本 - 已注释)
# ============================================================================
# import torch.nn.functional as F
#
#
# def activateFunc(x):
#     """
#     Tanh  ->  ReLU
#     """
#     x = torch.tanh(x)
#     return F.relu(x)
#
#
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


# ============================================================================
# Random Router (消融实验版本 - 已注释)
# ============================================================================
# class Router(nn.Module):
#     """
#     Random Routing (用于消融实验)
#     随机生成路径概率分布,不依赖输入特征
#     return: 路径概率向量 p_m^(n-1)
#     """
#
#     def __init__(self, num_out_path, embed_size, hid, temperature=1.0):
#         """
#         Args:
#             num_out_path: 路由的输出维度
#             embed_size: 嵌入维度 (保持接口一致性,但不使用)
#             hid: 隐藏层维度 (保持接口一致性,但不使用)
#             temperature: softmax 温度系数 (保持接口一致性,但不使用)
#         """
#         super(Router, self).__init__()
#         self.num_out_path = num_out_path
#         # Random Router 不需要任何可学习参数
#
#     def forward(self, x):  # (bsz, L, 768) -> (bsz, num_out_path)
#         """
#         生成随机路径概率分布
#         Args:
#             x: 输入张量 (bsz, L, embed_size)
#         Returns:
#             随机路径概率分布 (bsz, num_out_path)
#         """
#         bsz = x.size(0)
#         # 生成随机 logits 并通过 softmax 得到概率分布
#         random_logits = torch.rand(bsz, self.num_out_path, device=x.device)
#         return torch.softmax(random_logits, dim=-1)


# ============================================================================
# Hard Router (消融实验版本 - 已注释)
# ============================================================================
# class Router(nn.Module):
#     """
#     Hard Routing (用于消融实验)
#     基于输入特征选择单一最优路径,输出one-hot向量
#     return: 路径概率向量 p_m^(n-1) (one-hot形式)
#     """
#
#     def __init__(self, num_out_path, embed_size, hid, temperature=1.0):
#         """
#         Args:
#             num_out_path: 路由的输出维度
#             embed_size: 嵌入维度
#             hid: 隐藏层维度
#             temperature: softmax 温度系数 (保持接口一致性)
#         """
#         super(Router, self).__init__()
#         self.num_out_path = num_out_path
#         self.temperature = float(temperature)
#         self.pool = nn.Linear(embed_size, 1)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_size, hid),
#             nn.ReLU(True),
#             nn.Linear(hid, num_out_path),
#         )
#         self.init_weights()
#
#     def init_weights(self):
#         """
#         初始化路由层参数，保持 softmax 路由的可控性。
#         """
#         nn.init.zeros_(self.mlp[2].bias)
#
#     def forward(self, x):  # (bsz, L, 768) -> (bsz, num_out_path)
#         """
#         输入x通过注意力池化得到全局表示，然后经 MLP 转换为路径 logits，
#         选择最大logit对应的路径，输出one-hot向量。
#         """
#         # 注意力池化
#         attn_scores = self.pool(x).squeeze(-1)
#         attn_weights = torch.softmax(attn_scores, dim=-1)
#         pooled = torch.sum(attn_weights.unsqueeze(-1) * x, dim=-2)
#         
#         # 计算路径logits
#         logits = self.mlp(pooled)
#         logits = logits / max(self.temperature, 1e-6)
#         
#         # Hard routing: 选择最大logit对应的路径
#         # 训练时使用Gumbel-Softmax实现可微分的hard routing
#         if self.training:
#             # Gumbel-Softmax trick for differentiable hard routing
#             gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
#             logits_with_noise = (logits + gumbel_noise) / self.temperature
#             soft_routing = torch.softmax(logits_with_noise, dim=-1)
#             
#             # Straight-through estimator: forward用hard, backward用soft
#             hard_routing = torch.zeros_like(soft_routing)
#             hard_routing.scatter_(1, soft_routing.argmax(dim=-1, keepdim=True), 1.0)
#             
#             # Straight-through: forward=hard, backward=soft
#             return hard_routing + soft_routing - soft_routing.detach()
#         else:
#             # 推理时直接使用argmax
#             hard_routing = torch.zeros_like(logits)
#             hard_routing.scatter_(1, logits.argmax(dim=-1, keepdim=True), 1.0)
#             return hard_routing


# ============================================================================
# No Router / Uniform Router (消融实验版本 - 证明Router的必要性)
# ============================================================================
# class Router(nn.Module):
#     """
#     No Router / Uniform Routing (用于消融实验)
#     返回均匀分布的路径概率,所有路径权重相等
#     用于证明Sentiment-Aware Router动态调整路径权重的有效性
#     return: 路径概率向量 p_m^(n-1) (均匀分布)
#     """
#
#     def __init__(self, num_out_path, embed_size, hid, temperature=1.0):
#         """
#         Args:
#             num_out_path: 路由的输出维度
#             embed_size: 嵌入维度 (保持接口一致性,但不使用)
#             hid: 隐藏层维度 (保持接口一致性,但不使用)
#             temperature: softmax 温度系数 (保持接口一致性,但不使用)
#         """
#         super(Router, self).__init__()
#         self.num_out_path = num_out_path
#         # 不需要任何可学习参数
#
#     def forward(self, x):  # (bsz, L, 768) -> (bsz, num_out_path)
#         """
#         返回均匀分布的路径概率,每条路径的权重为 1/num_out_path
#         Args:
#             x: 输入张量 (bsz, L, embed_size)
#         Returns:
#             均匀分布的路径概率 (bsz, num_out_path)
#         """
#         bsz = x.size(0)
#         # 创建均匀分布: 每条路径的概率为 1/num_out_path
#         uniform_prob = torch.ones(bsz, self.num_out_path, device=x.device) / self.num_out_path
#         return uniform_prob
