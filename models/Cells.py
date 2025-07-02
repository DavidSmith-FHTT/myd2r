import torch
import torch.nn as nn
from .SelfAttention import SelfAttention
from .Router import Router
from models.Refinement import Refinement
from transformers import BertConfig, CLIPConfig
from models.XModules import CrossModalAlignment, AttentionFiltration


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class BertPooler(nn.Module):
    """
    Pooling 操作
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class RectifiedIdentityCell(nn.Module):
    """
    SSSR 单元 -> 第一个
    """

    def __init__(self, args, num_out_path):
        super(RectifiedIdentityCell, self).__init__()
        self.keep_mapping = nn.ReLU()
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

    def forward(self, x):
        path_prob = self.router(x)  # (bsz, L, 768) -> (bsz, 4)
        emb = self.keep_mapping(x)

        return emb, path_prob


class IntraModelReasoningCell(nn.Module):
    """
    USSR 单元  ->  第二个
    """

    def __init__(self, args, num_out_path):
        super(IntraModelReasoningCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.sa = SelfAttention(args.embed_size, args.hid_IMRC, args.num_head_IMRC)

    def forward(self, inp):
        path_prob = self.router(inp)
        if inp.dim() == 4:
            n_img, n_stc, n_local, dim = inp.size()
            x = inp.view(-1, n_local, dim)
        else:
            x = inp

        sa_emb = self.sa(x)
        if inp.dim() == 4:
            sa_emb = sa_emb.view(n_img, n_stc, n_local, -1)
        return sa_emb, path_prob


class CrossModalRefinementCell(nn.Module):
    """
    CLSSM 单元 -> 第三个
    """

    def __init__(self, args, num_out_path):
        super(CrossModalRefinementCell, self).__init__()
        self.refine = Refinement(args, args.embed_size, args.raw_feature_norm_CMRC, args.lambda_softmax_CMRC)
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

    def forward(self, text, image):
        path_prob = self.router(text)
        rf_pairs_emb = self.refine(text, image)

        return rf_pairs_emb, path_prob


class GlobalEnhancedSemanticCell(nn.Module):
    """
    CGSSA 单元 -> 第四个
    """
    def __init__(self, args, num_out_path):
        super(GlobalEnhancedSemanticCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))

        self.fc_mlp = nn.Sequential(nn.Linear(768, 768),
                                    nn.Tanh(),
                                    nn.Linear(768, 768))

    def global_gate_fusion(self, text, image):
        text_cls = self.text_cls_pool(text)  # (bsz, 768)
        image_cls = self.image_cls_pool(image)  # (bsz, 768)

        # 门控机制   全局信息对齐、融合
        gate_all = self.fc_mlp(text_cls + image_cls)  # (bsz, 768)
        gate = torch.softmax(gate_all, dim=-1)  # (bsz, 768)
        gate_out = gate * text_cls + (1 - gate) * image_cls  # (bsz, 768)
        gate_out = gate_out.unsqueeze(-2).expand(-1, text.size(1), -1)  # 将gate_out张量的维度扩展，以便与text的尺寸匹配。

        return gate_out

    def forward(self, text, image):
        path_prob = self.router(text)
        gate_out = self.global_gate_fusion(text, image)

        return gate_out, path_prob


class GlobalLocalAlignmentCell(nn.Module):
    """
    GLSSF 单元 -> 第五个
    """
    def __init__(self, args, num_out_path):
        super(GlobalLocalAlignmentCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)
        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name), args)
        self.SAF_module = AttentionFiltration(BertConfig.from_pretrained(args.bert_name).hidden_size)
        self.text_cls_pool = BertPooler(BertConfig.from_pretrained(args.bert_name))
        self.image_cls_pool = BertPooler(CLIPConfig.from_pretrained(args.vit_name).vision_config)
        self.fc_sim_tranloc = nn.Linear(768, 768)
        self.fc_sim_tranglo = nn.Linear(768, 768)
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, text, image):
        # 局部相似性表征. text_aware_image表示与文本相关的图像特征.
        text_aware_image, _ = self.CrossModalAlignment(text, image)  # (32, 128, 768)

        sim_local = torch.pow(torch.sub(text, text_aware_image), 2)
        sim_local = l2norm(self.fc_sim_tranloc(sim_local), dim=-1)
        sim_local = self.fc_1(sim_local)

        # 全局相似性表征   （32, 768）
        text_cls_output = self.text_cls_pool(text)
        image_cls_output = self.image_cls_pool(image)
        sim_global = torch.pow(torch.sub(text_cls_output, image_cls_output), 2)
        sim_global = l2norm(self.fc_sim_tranglo(sim_global), dim=-1)
        sim_global = self.fc_2(sim_global)

        # concat the global and local alignments
        sim_emb = torch.cat([sim_global.unsqueeze(1), sim_local], 1)  # (bsz, L+1, 768)

        # 相似图推理
        sim_emb = self.SAF_module(sim_emb)  # (bsz, 768)

        return sim_emb

    def forward(self, text, image):
        path_prob = self.router(text)

        sim_emb = self.alignment(text, image)
        sim_emb = sim_emb.unsqueeze(-2).expand(-1, text.size(1), -1)

        return sim_emb, path_prob


class ContextRichCrossModalCell(nn.Module):
    """
    MVSSU 单元 -> 第六个
    """
    def __init__(self, args, num_out_path):
        super(ContextRichCrossModalCell, self).__init__()
        self.args = args
        self.router = Router(num_out_path, args.embed_size, args.hid_router)

        self.CrossModalAlignment = CrossModalAlignment(BertConfig.from_pretrained(args.bert_name), args)
        self.fc_mlp_1 = nn.Sequential(nn.Linear(768, 768),
                                      nn.Tanh())
        self.fc_mlp_2 = nn.Sequential(nn.Linear(768, 768),
                                      nn.Tanh())
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)

    def alignment(self, text, image):
        text_aware_image, _ = self.CrossModalAlignment(text, image)  # (32, 128, 768)
        Q_state = self.fc_mlp_1(text_aware_image)  # (32, 128, 768)
        K_state = self.fc_mlp_2(text)  # (32, 128, 768)
        Q = self.fc_1(Q_state)
        K = self.fc_2(K_state)

        scores = torch.matmul(Q, K.transpose(-1, -2))  # (bsz, 128, 128)
        scores = nn.Softmax(dim=-1)(scores)  # (bsz, 128, 128)
        output = Q_state + torch.bmm(scores, K_state)  # (32, 128, 768)

        return output

    def forward(self, text, image):
        path_prob = self.router(text)
        output = self.alignment(text, image)

        return output, path_prob
