import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import csv

# ====== 按 runH.py 的项目结构导入 ======
from models.unimo_model import UnimoModelF
from processor.dataset import MSDProcessor, MSDDataset
from transformers import CLIPProcessor, CLIPConfig, BertConfig


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def cluster_and_export_images(
    emb2d,          # (N, 2) t-SNE 坐标
    img_names,      # List[str]
    out_csv,
    n_clusters=4,
    topk=20
):
    """
    对 t-SNE 结果做 4 类聚类，并导出每个簇最代表性的图片名
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(emb2d)
    centers = kmeans.cluster_centers_

    # 每个点到自己簇中心的距离
    dist = np.linalg.norm(emb2d - centers[cluster_ids], axis=1)

    rows = []
    for cid in range(n_clusters):
        idx = np.where(cluster_ids == cid)[0]
        idx = idx[np.argsort(dist[idx])]   # 距离中心由近到远
        for rank, i in enumerate(idx[:topk], start=1):
            rows.append([
                cid,
                rank,
                float(dist[i]),
                img_names[i]
            ])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cluster_id", "rank", "dist_to_center", "img_name"])
        writer.writerows(rows)

    print(f"[OK] Exported cluster image list to {out_csv}")
    return cluster_ids


@torch.no_grad()
def collect_pool_out(model, dataloader, device):
    """
    从 dataloader 抽取 pool_out 与 labels
    依赖：你在 UnimoModelF.forward() 内部缓存 self.last_pool_out = pool_out
    """
    model.eval()
    feats, labs, names = [], [], []

    for batch in tqdm(dataloader, desc="Collecting pool_out"):
        # batch 结构来自你的 trainer._step：input_ids, input_mask, segment_ids, img_mask, labels, images :contentReference[oaicite:3]{index=3}
        batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
        input_ids, input_mask, segment_ids, img_mask, labels, images, img_names = batch

        # forward：保持你原本接口 return (loss, final_output)
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=labels,
            images=images
        )

        pool_out = model.last_pool_out  # (bsz, 768)
        feats.append(pool_out.detach().cpu())
        labs.append(labels.detach().cpu())
        names.extend(list(img_names))

    feats = torch.cat(feats, dim=0)
    labs = torch.cat(labs, dim=0)
    return feats, labs, names


def plot_tsne(feats, labels, out_png, perplexity=30, point_size=14):
    feats = feats.numpy()
    labels = labels.numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=42
    )
    emb = tsne.fit_transform(feats)

    # 你这个任务是 3 类（fc 输出 3）——颜色给 3 个即可
    colors = ["#8d6e63", "#1e88e5", "#e53935"]  # 棕/蓝/红（和你之前图区分开）

    plt.figure(figsize=(7.6, 4.4), dpi=160)
    for c in np.unique(labels):
        idx = labels == c
        plt.scatter(
            emb[idx, 0], emb[idx, 1],
            s=point_size, alpha=0.92,
            color=colors[int(c) % len(colors)],
            edgecolors="none"
        )

    ax = plt.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.show()

    return emb


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default="HFM", choices=["Single", "Multiple", "HFM"])
    parser.add_argument('--bert_name', type=str, default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/bert-base-uncased")
    parser.add_argument('--vit_name', type=str, default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/clip-vit-base-patch32")
    parser.add_argument('--ckpt', type=str, default="/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/Single/3_best_model.pth")
    # parser.add_argument('--ckpt', type=str, default="/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/Multiple/12_best_model.pth")
    # parser.add_argument('--ckpt', type=str, default="/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/HFM/11_best_model.pth")

    parser.add_argument('--split', type=str, default="test", choices=["dev", "test"])
    parser.add_argument('--device', type=str, default="cuda:1")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407)

    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--point_size', type=int, default=14)
    parser.add_argument('--out_dir', type=str, default="./viz_out")
    parser.add_argument('--clusters', type=int, default="4")

    # ++++++++++++++++++++++

    parser.add_argument('--num_epochs', default=20, type=int, help="num training epochs")
    parser.add_argument('--lr', default=2e-5, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.2, type=float)
    parser.add_argument('--load_path', default=None, type=str, help="Load model from load_path")
    parser.add_argument('--save_path', default='/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/HFM/', type=str,
                        help="save best model at save_path")
    parser.add_argument('--write_path', default=None, type=str, help="do_test=True, predictions will be write in write_path")
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")

    parser.add_argument('--do_train', action='store_true', default=True, help="Whether to run training.")
    parser.add_argument('--only_test', action='store_true', help="Only test.")
    parser.add_argument('--ignore_idx', default=0, type=int, help="Specify the index to be ignored")
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")

    parser.add_argument('--alpha', default=0, type=float, help="CCR")
    parser.add_argument('--margin', default=0.1, type=float, help="CCR")

    parser.add_argument('--beta', default=0.1, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--mild_margin', default=0.7, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--hetero', default=0.9, type=float, help="SoftContrastiveLoss")
    parser.add_argument('--homo', default=0.9, type=float, help="SoftContrastiveLoss")

    parser.add_argument('--DR_step', default=3, type=int, help="Dynamic Route steps")
    parser.add_argument('--weight_js_1', default=0.6, type=float, help="JS divergence")
    parser.add_argument('--weight_js_2', default=1.0, type=float, help="JS divergence")
    parser.add_argument('--weight_diff', default=0, type=float, help="diff_loss")

    parser.add_argument('--embed_size', default=768, type=int, help='Dimensionality of the joint embedding.')
    parser.add_argument('--num_head_IMRC', type=int, default=16, help='Number of heads in Intra-Modal Reasoning Cell')
    parser.add_argument('--hid_IMRC', type=int, default=768,
                        help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=768, help='Hidden size of MLP in routers')

    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # ===== 数据路径：按 runH.py 固定写法 ===== :contentReference[oaicite:5]{index=5}
    if args.dataset == "HFM":
        data_path = {
            'train': 'data/HFM/train.json',
            'dev': 'data/HFM/valid.json',
            'test': 'data/HFM/test.json'
        }
        img_path = 'data/HFM/dataset_image'
    elif args.dataset == "Multiple":
        data_path = {
            'train': 'data/MVSA-multiple/10-flod-1/train.json',
            'dev': 'data/MVSA-multiple/10-flod-1/dev.json',
            'test': 'data/MVSA-multiple/10-flod-1/test.json'
        }
        img_path = 'data/MVSA-multiple/data'
    elif args.dataset == "Single":
        data_path = {
            'train': 'data/MVSA-single/10-flod-1/train.json',
            'dev': 'data/MVSA-single/10-flod-1/dev.json',
            'test': 'data/MVSA-single/10-flod-1/test.json'
        }
        img_path = 'data/MVSA-single/data'
    else:
        data_path = {
            'train': 'data/HFM/train.json',
            'dev': 'data/HFM/valid.json',
            'test': 'data/HFM/test.json'
        }
        img_path = 'data/HFM/dataset_image'

    # ===== 构建 processor & dataset（只需要 CLIPProcessor，不需要 CLIPModel）=====
    clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
    processor = MSDProcessor(data_path, args.bert_name, clip_processor=clip_processor)

    mode = "dev" if args.split == "dev" else "test"
    dataset = MSDDataset(processor, img_path=img_path, max_seq=args.max_seq, mode=mode)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # ===== 构建模型并加载 ckpt（ckpt 已含训练后的权重）=====
    vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
    text_config = BertConfig.from_pretrained(args.bert_name)
    model = UnimoModelF(args=args, vision_config=vision_config, text_config=text_config).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[OK] Loaded ckpt: {args.ckpt}")

    # ===== 抽 pool_out 并可视化 =====
    feats, labels, img_names = collect_pool_out(model, dataloader, device)
    print("[OK] feats:", tuple(feats.shape), "labels:", tuple(labels.shape))

    out_png = os.path.join(args.out_dir, f"tsne_poolout_{args.dataset}.png")
    emb = plot_tsne(feats, labels, out_png, perplexity=args.perplexity, point_size=args.point_size)

    cluster_and_export_images(
        emb2d=emb,
        img_names=img_names,
        out_csv="cluster_" + args.dataset + "_test_images.csv",
        n_clusters=args.clusters,
        topk=20
    )

    print(f"[OK] saved: {out_png}")


if __name__ == "__main__":
    main()
