import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn import init
from models.DynamicInteraction import DynamicInteraction_Layer0, DynamicInteraction_Layer, Reversed_DynamicInteraction_Layer0, Reversed_DynamicInteraction_Layer


class InteractionModule(nn.Module):
    """
    Dynamic Sentiment Interaction Module overall architecture(以文本为中心)
    Args:
        num_layer_routing 表示有几层
    """
    def __init__(self, args, num_layer_routing=3, num_cells=4, path_hid=128):
        super(InteractionModule, self).__init__()
        self.args = args
        self.num_cells = num_cells
        self.dynamic_itr_l0 = DynamicInteraction_Layer0(args, num_cells, num_cells)
        # self.dynamic_itr_l1 = DynamicInteraction_Layer(args, num_cells, num_cells)
        self.dynamic_itr_l1 = nn.ModuleList([DynamicInteraction_Layer(args, num_cells, num_cells) for i in range(num_layer_routing-2)])
        self.dynamic_itr_l2 = DynamicInteraction_Layer(args, num_cells, 1)
        # 所有层的总路径数 total_paths
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)

    def forward(self, text, image):
        # 处理输入
        mid_paths = []
        # 初始层
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(text, image)
        # 中间层
        for module in self.dynamic_itr_l1:
            pairs_emb_lst, paths_l1 = module(pairs_emb_lst, text, image)
            mid_paths.append(paths_l1)
        # 最终层
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, text, image)    # list:1  (32, L, 128)

        n_img, n_stc = paths_l2.size()[:2]

        # 将各层的路径概率张量重塑为统一格式：
        # paths_l0 = paths_l0.contiguous().view(n_stc, -1).unsqueeze(0).expand(n_img, -1, -1)    # (32, 1, 512)
        paths_l0 = paths_l0.view(n_img, n_stc, -1)  # (32, 1, 16)

        for i in range(0, len(mid_paths)):
            if i == 0:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)  # (32, 1, 16)
                paths_l1 = mid_paths[i]
            else:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)  # (32, 1, 16)
                paths_l1 = torch.cat([paths_l1, mid_paths[i]], dim=-1)
        # paths_l1 = paths_l1.view(n_img, n_stc, -1)   # (32, 1, 16)
        paths_l2 = paths_l2.view(n_img, n_stc, -1)   # (32, 1, 4)

        # 将所有路径拼接成完整的路径张量，并计算批次间的路径相似度。相似性矩阵 （sim_paths） 用于衡量批次中不同实例的相互关联程度。此信息对于对比学习和一致性正则化很有用。
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1) # (n_img, n_stc, total_paths)    (32, 1, 36)
        # paths = paths.mean(dim=0) # (n_stc, total_paths)   (1, 36)
        #
        # paths = self.path_mapping(paths)   # (32, 1, 128)
        # paths = F.normalize(paths, dim=-1)
        # sim_paths = paths.matmul(paths.t())   # (1, 1)
        sim_paths = torch.matmul(paths.squeeze(-2), paths.squeeze(-2).transpose(-1, -2))   # (bsz, bsz)

        return pairs_emb_lst, sim_paths


class Reversed_InteractionModule(nn.Module):
    """
    Dynamic Sentiment Interaction Module overall architecture(以图片为中心)
    """
    def __init__(self, args, num_layer_routing=3, num_cells=4, path_hid=128):
        super(Reversed_InteractionModule, self).__init__()
        self.args = args
        self.num_cells = num_cells
        self.dynamic_itr_l0 = Reversed_DynamicInteraction_Layer0(args, num_cells, num_cells)
        # self.dynamic_itr_l1 = DynamicInteraction_Layer(args, num_cells, num_cells)
        self.dynamic_itr_l1 = nn.ModuleList(
            [Reversed_DynamicInteraction_Layer(args, num_cells, num_cells) for i in range(num_layer_routing - 2)])
        self.dynamic_itr_l2 = Reversed_DynamicInteraction_Layer(args, num_cells, 1)
        total_paths = num_cells ** 2 * (num_layer_routing - 1) + num_cells
        self.path_mapping = nn.Linear(total_paths, path_hid)
        self.bn = nn.BatchNorm1d(args.embed_size)

    def forward(self, text, image):
        mid_paths = []
        # 初始层
        pairs_emb_lst, paths_l0 = self.dynamic_itr_l0(text, image)
        # 中间层
        for module in self.dynamic_itr_l1:
            pairs_emb_lst, paths_l1 = module(pairs_emb_lst, text, image)
            mid_paths.append(paths_l1)
        # 最终层
        pairs_emb_lst, paths_l2 = self.dynamic_itr_l2(pairs_emb_lst, text, image)  # list:1  (32, L, 128)

        n_img, n_stc = paths_l2.size()[:2]

        # paths_l0 = paths_l0.contiguous().view(n_stc, -1).unsqueeze(0).expand(n_img, -1, -1)    # (32, 1, 512)
        paths_l0 = paths_l0.view(n_img, n_stc, -1)  # (32, 1, 16)

        for i in range(0, len(mid_paths)):
            if i == 0:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)  # (32, 1, 16)
                paths_l1 = mid_paths[i]
            else:
                mid_paths[i] = mid_paths[i].view(n_img, n_stc, -1)  # (32, 1, 16)
                paths_l1 = torch.cat([paths_l1, mid_paths[i]], dim=-1)
        # paths_l1 = paths_l1.view(n_img, n_stc, -1)   # (32, 1, 16)
        paths_l2 = paths_l2.view(n_img, n_stc, -1)  # (32, 1, 4)
        paths = torch.cat([paths_l0, paths_l1, paths_l2], dim=-1)  # (n_img, n_stc, total_paths)    (32, 1, 36)
        # paths = paths.mean(dim=0)  # (n_stc, total_paths)   (1, 36)
        #
        # paths = self.path_mapping(paths)  # (32, 1, 128)
        # paths = F.normalize(paths, dim=-1)
        # sim_paths = paths.matmul(paths.t())  # (1, 1)
        sim_paths = torch.matmul(paths.squeeze(-2), paths.squeeze(-2).transpose(-1, -2))  # (bsz, bsz)

        return pairs_emb_lst, sim_paths