import torch
import torch.nn as nn
from models.Cells import RectifiedIdentityCell, IntraModelReasoningCell, CrossModalRefinementCell, ContextRichCrossModalCell, GlobalEnhancedSemanticCell


def unsqueeze2d(x):
    return x.unsqueeze(-1).unsqueeze(-1)


class DynamicInteraction_Layer0(nn.Module):
    """
    Dynamic Sentiment Interaction Module(以文本为中心) 的第一层
    消融实验：移除GlobalLocalAlignmentCell (GLSSF单元)
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell - 1  # 从6个单元减少到5个
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        # 移除 self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, text, image):

        # 只通过五个处理单元处理输入（移除GlobalLocalAlignmentCell）
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(text)
        # 移除: emb_lst[1], path_prob[1] = self.glac(text, image)
        emb_lst[1], path_prob[1] = self.imrc(text)
        emb_lst[2], path_prob[2] = self.cmrc(text, image)
        emb_lst[3], path_prob[3] = self.crcmc(text, image)
        emb_lst[4], path_prob[4] = self.gesc(text, image)

        # 计算门控掩码并将所有路径概率堆叠并进行归一化，确保概率和为1
        gate_mask = (sum(path_prob) < self.threshold).float() 
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        # 通过加权聚合生成最终输出（使用第一个单元作为skip connection）
        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]  # 使用ric作为skip
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j]
                else:  
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class DynamicInteraction_Layer(nn.Module):
    """
    Dynamic Sentiment Interaction Module(以文本为中心) 的中间层和最后一层。
    消融实验：移除GlobalLocalAlignmentCell (GLSSF单元)
    """
    def __init__(self, args, num_cell, num_out_path):
        super(DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell - 1  # 从6个单元减少到5个
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(args, num_out_path)
        # 移除 self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):

        # 通过5个不同的处理单元并行处理输入数据（移除GlobalLocalAlignmentCell）
        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        # 移除: emb_lst[1], path_prob[1] = self.glac(ref_wrd[1], image)
        emb_lst[1], path_prob[1] = self.imrc(ref_wrd[1])
        emb_lst[2], path_prob[2] = self.cmrc(ref_wrd[2], image)
        emb_lst[3], path_prob[3] = self.crcmc(ref_wrd[3], image)
        emb_lst[4], path_prob[4] = self.gesc(ref_wrd[4], image)

        # num_out_path == 1 表示最后一层
        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float() 
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class Reversed_DynamicInteraction_Layer0(nn.Module):
    """
    Dynamic Sentiment Interaction Module(以图片为中心) 的第一层
    消融实验：移除GlobalLocalAlignmentCell (GLSSF单元)
    """
    def __init__(self, args, num_cell, num_out_path):
        super(Reversed_DynamicInteraction_Layer0, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell - 1  # 从6个单元减少到5个
        self.num_out_path = num_out_path
        self.ric = RectifiedIdentityCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        # 移除 self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, text, image):

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(image)
        # 移除: emb_lst[1], path_prob[1] = self.glac(image, text)
        emb_lst[1], path_prob[1] = self.imrc(image)
        emb_lst[2], path_prob[2] = self.cmrc(image, text)
        emb_lst[3], path_prob[3] = self.crcmc(image, text)
        emb_lst[4], path_prob[4] = self.gesc(image, text)

        gate_mask = (sum(path_prob) < self.threshold).float()
        all_path_prob = torch.stack(path_prob, dim=2)
        all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
        path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

        aggr_res_lst = []
        for i in range(self.num_out_path):
            skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
            res = 0
            for j in range(self.num_cell):
                cur_path = unsqueeze2d(path_prob[j][:, i])
                if emb_lst[j].dim() == 3:
                    cur_emb = emb_lst[j]
                else:
                    cur_emb = emb_lst[j]
                res = res + cur_path * cur_emb
            res = res + skip_emb
            aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob


class Reversed_DynamicInteraction_Layer(nn.Module):
    """
    Dynamic Sentiment Interaction Module(以图片为中心) 的中间层和最后一层。
    消融实验：移除GlobalLocalAlignmentCell (GLSSF单元)
    """
    def __init__(self, args, num_cell, num_out_path):
        super(Reversed_DynamicInteraction_Layer, self).__init__()
        self.args = args
        self.threshold = 0.0001
        self.eps = 1e-8
        self.num_cell = num_cell - 1  # 从6个单元减少到5个
        self.num_out_path = num_out_path

        self.ric = RectifiedIdentityCell(args, num_out_path)
        # 移除 self.glac = GlobalLocalAlignmentCell(args, num_out_path)
        self.imrc = IntraModelReasoningCell(args, num_out_path)
        self.cmrc = CrossModalRefinementCell(args, num_out_path)
        self.crcmc = ContextRichCrossModalCell(args, num_out_path)
        self.gesc = GlobalEnhancedSemanticCell(args, num_out_path)

    def forward(self, ref_wrd, text, image):

        path_prob = [None] * self.num_cell
        emb_lst = [None] * self.num_cell
        emb_lst[0], path_prob[0] = self.ric(ref_wrd[0])
        # 移除: emb_lst[1], path_prob[1] = self.glac(ref_wrd[1], text)
        emb_lst[1], path_prob[1] = self.imrc(ref_wrd[1])
        emb_lst[2], path_prob[2] = self.cmrc(ref_wrd[2], text)
        emb_lst[3], path_prob[3] = self.crcmc(ref_wrd[3], text)
        emb_lst[4], path_prob[4] = self.gesc(ref_wrd[4], text)

        if self.num_out_path == 1:
            aggr_res_lst = []
            gate_mask_lst = []
            res = 0
            for j in range(self.num_cell):
                gate_mask = (path_prob[j] < self.threshold / self.num_cell).float()
                gate_mask_lst.append(gate_mask)
                skip_emb = gate_mask.unsqueeze(-1) * ref_wrd[j]
                res += path_prob[j].unsqueeze(-1) * emb_lst[j]
                res += skip_emb

            res = res / (sum(gate_mask_lst) + sum(path_prob)).unsqueeze(-1)
            all_path_prob = torch.stack(path_prob, dim=2)
            aggr_res_lst.append(res)
        else:
            gate_mask = (sum(path_prob) < self.threshold).float()
            all_path_prob = torch.stack(path_prob, dim=2)
            all_path_prob = all_path_prob / (all_path_prob.sum(dim=-1, keepdim=True) + self.eps)
            path_prob = [all_path_prob[:, :, i] for i in range(all_path_prob.size(2))]

            aggr_res_lst = []
            for i in range(self.num_out_path):
                skip_emb = unsqueeze2d(gate_mask[:, i]) * emb_lst[0]
                res = 0
                for j in range(self.num_cell):
                    cur_path = unsqueeze2d(path_prob[j][:, i])
                    res = res + cur_path * emb_lst[j]
                res = res + skip_emb
                aggr_res_lst.append(res)

        return aggr_res_lst, all_path_prob

