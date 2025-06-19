import torch
import torch.nn as nn
import torch.nn.functional as F


def activateFunc(x):
    """
    Tanh  ->  ReLU
    """
    x = torch.tanh(x)
    return F.relu(x)


class Router(nn.Module):
    """
    return: 路径概率向量 p_m^(n-1)
    """
    def __init__(self, num_out_path, embed_size, hid):
        super(Router, self).__init__()
        self.num_out_path = num_out_path
        self.mlp = nn.Sequential(nn.Linear(embed_size, hid),
                                 nn.ReLU(True),
                                 nn.Linear(hid, num_out_path))
        self.init_weights()

    def init_weights(self):
        """
        将MLP中第三层（self.mlp[2]）的偏置值全部设置为1.5。
        """
        self.mlp[2].bias.data.fill_(1.5)

    def forward(self, x):  # (bsz, L, 768) -> (bsz, 4)
        """
        输入x首先通过mean(-2)操作在第二维度（通常是序列的长度）上求均值，然后通过MLP进行转换，最后通过activateFunc函数应用激活。
        """
        x = x.mean(-2)
        x = self.mlp(x)
        soft_g = activateFunc(x)
        return soft_g
