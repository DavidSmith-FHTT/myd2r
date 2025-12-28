import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import numpy as np

device = torch.device("cuda:1")


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def js_div(p_output, q_output, get_softmax=True):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output, dim=-1)
        q_output = F.softmax(q_output, dim=-1)
    log_mean_output = ((p_output + q_output) / 2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2


class ContrastiveLoss(nn.Module):
    """
    下面的 CrossModalAlignment 用到了
    """
    def __init__(self, alpha, beta, margin=0.2, measure='cosine', max_violation=True):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.measure = measure
        self.max_violation = max_violation

    def forward(self, img_rep, txt_rep):
        """
            image_rep: (bs, 50, 768) -> attention weighted && reverse attention-> (bs, 4, 2, 768)
            label_rep: (bs, 4, 768) -> (bs, 4, 1, 768)
            where dim = -2 can be regarded as batch size
        """
        if self.measure == 'cosine':
            # shape: (bs, 4, 2)
            # CCR Part
            scores = self.cosine_sim_v1(img_rep, txt_rep).squeeze()
            # scores[0] representation positive result
            cost_ccr = (self.margin + scores - scores[:, :, 0].unsqueeze(-1)).clamp(0)

            # CCR mask
            mask = torch.tensor([1., 0.]).unsqueeze(0).unsqueeze(1).expand_as(scores) == 1.
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.to(device)
            cost_ccr = cost_ccr.masked_fill_(I, 0)

            # shape: (bs, 4, 4)
            # CCS Part
            scores = self.cosine_sim_v2(img_rep, txt_rep)
            diagonal = torch.diagonal(scores, dim1=-2, dim2=-1).view(scores.size(0), -1, 1)
            d = diagonal.expand_as(scores)
            cost_ccs = (self.margin + scores - d).clamp(min=0)

            # CCS mask
            mask = torch.eye(scores.size(-1)).expand_as(scores) > .5
            I = Variable(mask)
            if torch.cuda.is_available():
                I = I.to(device)
            cost_ccs = cost_ccs.masked_fill_(I, 0)

            if self.max_violation:
                cost_ccs = cost_ccs.max(-1)[0]
            return self.alpha * cost_ccr.sum() + self.beta * cost_ccs.sum()

    @staticmethod
    def cosine_sim_v1(img_rep, txt_rep):
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))

    @staticmethod
    def cosine_sim_v2(img_rep, txt_rep):
        img_rep = img_rep[:, :, 0, :]
        txt_rep = txt_rep.squeeze()
        return torch.matmul(img_rep, txt_rep.transpose(-1, -2).contiguous()) / math.sqrt(img_rep.size(-1))


class CrossModalAlignment(nn.Module):
    """
    跨模态对齐，Cells 里面第五个第六个用到了。
    """
    def __init__(self, config, args):
        super(CrossModalAlignment, self).__init__()
        self.config = config
        self.args = args
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.fc_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc_2 = nn.Linear(config.hidden_size, config.hidden_size)

        self.closs = ContrastiveLoss(alpha=args.alpha, beta=0, margin=args.margin)

    def forward(self, text_emb, image_emb):
        """
        inputs :
            text_emb : input feature maps( B X 128 X 768 )
            image_emb : input feature maps( B X 50 X 768 )
        returns :
            out : ( B X 128 X 768 )
        """
        query_layer = self.query(text_emb)  # (bsz, 128, 768)
        key_layer = self.key(image_emb)  # (bsz, 50, 768)
        value_layer = self.value(image_emb)  # (bsz, 50, 768)

        # (bsz, 128, 50)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))
        attn_score = attention_scores / math.sqrt(self.config.hidden_size)

        # Softmax attention   (bsz, 128, 768)
        attn_score = torch.softmax(100 * attn_score, dim=-1)
        text_img_rep_init = torch.bmm(attn_score, value_layer)

        # reverse Softmax attention    (bsz, 128, 768)
        reverse_score = torch.softmax(100 * (1 - attn_score), dim=-1)
        reverse_text_img_rep_init = torch.bmm(reverse_score, value_layer)

        # text_img_rep_init, reverse_text_img_rep_init = SCAN_attention(text_emb, image_emb, smooth=9.0)
        # # (32, 128, 768)
        # text_img_rep_init, reverse_text_img_rep_init = self.text_img_CrossAttention(text_emb, image_emb, image_attention_mask)
        # (32, 128, 1, 768)
        text_img_rep = self.fc_1(text_img_rep_init).unsqueeze(-2)
        reverse_text_img_rep = self.fc_2(reverse_text_img_rep_init).unsqueeze(-2)
        # (bsz, 128, 2, 768)
        total_text_img_rep = torch.cat((text_img_rep, reverse_text_img_rep), dim=-2)

        text_img_loss = self.closs(torch.nn.functional.normalize(total_text_img_rep),
                                   torch.nn.functional.normalize(text_emb.unsqueeze(-2)))

        return text_img_rep_init, text_img_loss


class AttentionFiltration(nn.Module):
    """
    Cells 第五个单元用到 AttentionFiltration
    实现了一个基于门控注意力机制的相似性过滤模块。它的作用是对全局和局部对齐信息进行加权处理，输出经过注意力加权后的聚合结果。
    Perform the similarity Attention Filtration with a gate-based attention
    Args: - sim_emb: global and local alignments, shape: (batch_size, L+1, 256)
    Returns; - sim_saf: aggregated alignment after attention filtration, shape: (batch_size, 256)
    """
    def __init__(self, sim_dim):
        super(AttentionFiltration, self).__init__()

        self.attn_sim_w = nn.Linear(sim_dim, 1)
        self.bn = nn.BatchNorm1d(1)

        self.init_weights()

    def forward(self, sim_emb):
        sim_attn = l1norm(torch.sigmoid(self.bn(self.attn_sim_w(sim_emb).permute(0, 2, 1))), dim=-1)
        sim_saf = torch.matmul(sim_attn, sim_emb)
        sim_saf = l2norm(sim_saf.squeeze(1), dim=-1)
        return sim_saf

    def init_weights(self):
        for m in self.children():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def get_sizes_list(dim, chunks):
    """
    下面的 Block 用到了
    """
    split_size = (dim + chunks - 1) // chunks  # split_size:80
    sizes_list = [split_size] * chunks  # list:20 [80, 80, ..., 80]
    sizes_list[-1] = sizes_list[-1] - (sum(sizes_list) - dim)  # Adjust last
    assert sum(sizes_list) == dim
    if sizes_list[-1] < 0:
        n_miss = sizes_list[-2] - sizes_list[-1]
        sizes_list[-1] = sizes_list[-2]
        for j in range(n_miss):
            sizes_list[-j - 1] -= 1
        assert sum(sizes_list) == dim
        assert min(sizes_list) > 0
    return sizes_list


def get_chunks(x, sizes):
    """
    下面的 Block 用到了
    """
    out = []
    begin = 0
    for s in sizes:
        y = x.narrow(1, begin, s)  # (32, 80)
        out.append(y)  # list:20
        begin += s
    return out


class Block(nn.Module):
    """
    Block Fusion Mechanism
    """
    def __init__(self, input_dims, output_dim, mm_dim=1600, chunks=20, rank=15, shared=False, dropout_input=0.,
                 dropout_pre_lin=0., dropout_output=0., pos_norm='before_cat'):
        super(Block, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim
        self.mm_dim = mm_dim
        self.chunks = chunks
        self.rank = rank
        self.shared = shared
        self.dropout_input = dropout_input
        self.dropout_pre_lin = dropout_pre_lin
        self.dropout_output = dropout_output
        assert (pos_norm in ['before_cat', 'after_cat'])
        self.pos_norm = pos_norm
        #  Modules
        self.linear0 = nn.Linear(input_dims[0], mm_dim)
        if shared:
            self.linear1 = self.linear0
        else:
            self.linear1 = nn.Linear(input_dims[1], mm_dim)
        merge_linears0, merge_linears1 = [], []
        self.sizes_list = get_sizes_list(mm_dim, chunks)
        for size in self.sizes_list:
            ml0 = nn.Linear(size, size * rank)  # (80, 1200)
            merge_linears0.append(ml0)
            if self.shared:
                ml1 = ml0
            else:
                ml1 = nn.Linear(size, size * rank)
            merge_linears1.append(ml1)
        self.merge_linears0 = nn.ModuleList(merge_linears0)
        self.merge_linears1 = nn.ModuleList(merge_linears1)
        self.linear_out = nn.Linear(mm_dim, output_dim)
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        x0 = self.linear0(x[0])  # (32, 1600)
        x1 = self.linear1(x[1])  # (32, 1600)
        bsize = x1.size(0)
        if self.dropout_input > 0:
            x0 = F.dropout(x0, p=self.dropout_input, training=self.training)
            x1 = F.dropout(x1, p=self.dropout_input, training=self.training)
        x0_chunks = get_chunks(x0, self.sizes_list)  # list:20  (32, 80)
        x1_chunks = get_chunks(x1, self.sizes_list)  # list:20  (32, 80)
        zs = []
        for chunk_id, m0, m1 in zip(range(len(self.sizes_list)),
                                    self.merge_linears0,
                                    self.merge_linears1):
            x0_c = x0_chunks[chunk_id]  # (32, 80)
            x1_c = x1_chunks[chunk_id]  # (32, 80)
            m = m0(x0_c) * m1(x1_c)  # bsize x split_size*rank   (32, 80 * 15)
            m = m.view(bsize, self.rank, -1)  # (32, 15, 80)
            z = torch.sum(m, 1)  # (32, 80)
            if self.pos_norm == 'before_cat':
                z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))  # (32, 80)
                z = F.normalize(z, p=2)  # (32, 80)
            zs.append(z)  # list:20
        z = torch.cat(zs, 1)  # (32, 1600)
        if self.pos_norm == 'after_cat':
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2)

        if self.dropout_pre_lin > 0:
            z = F.dropout(z, p=self.dropout_pre_lin, training=self.training)
        z = self.linear_out(z)  # (32, 768)
        if self.dropout_output > 0:
            z = F.dropout(z, p=self.dropout_output, training=self.training)
        return z
