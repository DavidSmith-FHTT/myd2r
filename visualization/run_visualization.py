"""
Router可视化运行脚本
直接加载已训练的checkpoint，收集路由数据并生成可视化

使用方法:
    python visualization/run_visualization.py \
        --load_path /path/to/best_model.pth \
        --bert_name /path/to/bert \
        --vit_name /path/to/vit \
        --data_path ./data \
        --dataset Single
"""

import torch
import numpy as np
import pickle
import os
import sys
import argparse
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader
from transformers import CLIPConfig, BertConfig, CLIPProcessor
from processor.dataset import MSDProcessor, MSDDataset
from models.unimo_model import UnimoModelF
from router_visualization import RouterVisualization


class RoutingHook:
    """通过hook捕获路径概率"""
    
    def __init__(self):
        self.layer_probs = {
            'text_l0': [],
            'text_l1': [],
            'text_l2': [],
            'image_l0': [],
            'image_l1': [],
            'image_l2': [],
        }
        self.hooks = []
    
    def _make_hook(self, name):
        def hook(module, input, output):
            # DynamicInteraction_Layer的output是 (aggr_res_lst, all_path_prob)
            if isinstance(output, tuple) and len(output) == 2:
                all_path_prob = output[1]
                if all_path_prob is not None and isinstance(all_path_prob, torch.Tensor):
                    self.layer_probs[name].append(all_path_prob.detach().cpu())
        return hook
    
    def register_hooks(self, model):
        """注册hooks到模型的DynamicInteraction层（只注册顶层模块）"""
        registered_names = set()
        
        for name, module in model.named_modules():
            # 只匹配精确的层名称，避免匹配子模块
            # 文本中心交互模块
            if name == 'model.itr_module.dynamic_itr_l0':
                h = module.register_forward_hook(self._make_hook('text_l0'))
                self.hooks.append(h)
                registered_names.add(name)
            elif name == 'model.itr_module.dynamic_itr_l2':
                h = module.register_forward_hook(self._make_hook('text_l2'))
                self.hooks.append(h)
                registered_names.add(name)
            # 中间层 (ModuleList中的每个元素)
            elif 'model.itr_module.dynamic_itr_l1.' in name:
                # 只匹配 model.itr_module.dynamic_itr_l1.0, .1 等
                parts = name.split('.')
                if len(parts) == 4 and parts[-1].isdigit():
                    h = module.register_forward_hook(self._make_hook('text_l1'))
                    self.hooks.append(h)
                    registered_names.add(name)
            
            # 图像中心交互模块
            elif name == 'model.Reversed_itr_module.dynamic_itr_l0':
                h = module.register_forward_hook(self._make_hook('image_l0'))
                self.hooks.append(h)
                registered_names.add(name)
            elif name == 'model.Reversed_itr_module.dynamic_itr_l2':
                h = module.register_forward_hook(self._make_hook('image_l2'))
                self.hooks.append(h)
                registered_names.add(name)
            elif 'model.Reversed_itr_module.dynamic_itr_l1.' in name:
                parts = name.split('.')
                if len(parts) == 4 and parts[-1].isdigit():
                    h = module.register_forward_hook(self._make_hook('image_l1'))
                    self.hooks.append(h)
                    registered_names.add(name)
        
        print(f"Registered {len(self.hooks)} hooks on: {registered_names}")
    
    def clear(self):
        """清空收集的数据"""
        for key in self.layer_probs:
            self.layer_probs[key] = []
    
    def remove_hooks(self):
        """移除所有hooks"""
        for h in self.hooks:
            h.remove()
        self.hooks = []
    
    def get_concatenated_probs(self):
        """合并所有batch的概率"""
        result = {}
        for key, probs_list in self.layer_probs.items():
            if len(probs_list) > 0:
                # 检查张量维度并统一格式
                valid_probs = []
                for prob in probs_list:
                    if prob.dim() == 0:
                        # 跳过0维标量
                        continue
                    elif prob.dim() == 2:
                        # 2维张量，保持原样
                        valid_probs.append(prob)
                    elif prob.dim() == 3:
                        # 3维张量 (batch, num_out_path, num_cells)，保持原样
                        valid_probs.append(prob)
                    else:
                        print(f"Warning: Unexpected shape for {key}: {prob.shape}")
                
                if len(valid_probs) > 0:
                    # 检查所有张量维度是否一致
                    dims = [p.dim() for p in valid_probs]
                    if len(set(dims)) == 1:
                        # 维度一致，直接拼接
                        result[key] = torch.cat(valid_probs, dim=0)
                    else:
                        # 维度不一致，只保留最常见的维度
                        most_common_dim = max(set(dims), key=dims.count)
                        filtered_probs = [p for p in valid_probs if p.dim() == most_common_dim]
                        if len(filtered_probs) > 0:
                            result[key] = torch.cat(filtered_probs, dim=0)
                            print(f"Info: {key} filtered to {most_common_dim}D tensors ({len(filtered_probs)}/{len(valid_probs)})")
        return result


def load_model(args):
    """加载模型"""
    print("Loading model configuration...")
    
    vision_config = CLIPConfig.from_pretrained(args.vit_name).vision_config
    text_config = BertConfig.from_pretrained(args.bert_name)
    
    print("Building model...")
    model = UnimoModelF(args, vision_config, text_config)
    
    print(f"Loading checkpoint from {args.load_path}...")
    state_dict = torch.load(args.load_path, map_location=args.device)
    model.load_state_dict(state_dict)
    
    model.to(args.device)
    model.eval()
    
    return model


def collect_data(model, dataloader, device, routing_hook):
    """收集路由数据和预测结果"""
    
    all_labels = []
    all_preds = []
    all_logits = []
    
    model.eval()
    routing_hook.clear()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting routing data"):
            # MSDDataset返回: (input_ids, input_mask, segment_ids, img_mask, label, image, img)
            input_ids, attention_mask, token_type_ids, img_mask, labels, pixel_values, img = batch
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                images=pixel_values
            )
            
            # UnimoModelF返回 (loss, final_output)
            loss, logits = outputs
            preds = torch.argmax(logits, dim=-1)
            
            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_logits.append(logits.cpu())
    
    # 获取路径概率
    path_probs = routing_hook.get_concatenated_probs()
    
    # 计算置信度
    all_logits = torch.cat(all_logits, dim=0)
    confidences = torch.softmax(all_logits, dim=-1).max(dim=-1)[0].numpy()
    
    return {
        'path_probs': path_probs,
        'labels': np.array(all_labels),
        'predictions': np.array(all_preds),
        'confidences': confidences,
        'logits': all_logits.numpy()
    }


def run_visualization(data, save_dir):
    """运行所有可视化"""
    
    viz = RouterVisualization(save_dir=save_dir)
    
    labels = data['labels']
    confidences = data['confidences']
    path_probs = data['path_probs']
    
    # 选择一个有效的层进行可视化
    # 优先使用text_l0（第一层通常有完整的6个Cell）
    main_layer = None
    for layer_name in ['text_l0', 'text_l2', 'image_l0']:
        if layer_name in path_probs and len(path_probs[layer_name]) > 0:
            main_layer = layer_name
            main_probs = path_probs[layer_name]
            print(f"Using {layer_name} for visualization, shape: {main_probs.shape}")
            break
    
    if main_layer is None:
        print("Warning: No valid path probabilities found!")
        return
    
    print("\n" + "="*50)
    print("Generating visualizations...")
    print("="*50)
    
    # 1. 路径概率热力图
    print("\n1. Path Probability Heatmap...")
    viz.plot_path_probability_heatmap(
        main_probs, 
        layer_name=main_layer.replace('_', ' ').title(),
        save_name='01_path_prob_heatmap.png'
    )
    
    # 2. 类别条件路径分布
    print("\n2. Class-Conditional Routing Distribution...")
    viz.plot_class_conditional_routing(
        main_probs, labels,
        layer_name=main_layer.replace('_', ' ').title(),
        save_name='02_class_conditional_routing.png'
    )
    
    # 3. 路由熵分析
    print("\n3. Routing Entropy Analysis...")
    viz.plot_routing_entropy(
        main_probs, labels,
        save_name='03_routing_entropy.png'
    )
    
    # 4. 路由-置信度关系
    print("\n4. Routing vs Confidence Analysis...")
    viz.plot_routing_vs_confidence(
        main_probs, confidences,
        save_name='04_routing_vs_confidence.png'
    )
    
    # 5. t-SNE可视化
    print("\n5. Routing t-SNE Visualization...")
    viz.plot_routing_tsne(
        main_probs, labels,
        save_name='05_routing_tsne.png'
    )
    
    # 6. 跨层路由流动（如果有多层数据）
    # 只使用样本数量匹配的层（跳过中间层，因为有多个子层会导致样本数翻倍）
    layer_probs_list = []
    num_samples = len(labels)
    for layer_name in ['text_l0', 'text_l2']:  # 跳过text_l1
        if layer_name in path_probs:
            probs = path_probs[layer_name]
            if len(probs) == num_samples:
                layer_probs_list.append(probs)
            else:
                print(f"Skipping {layer_name}: {len(probs)} samples vs {num_samples} labels")
    
    if len(layer_probs_list) >= 2:
        print("\n6. Cross-Layer Routing Flow...")
        viz.plot_layer_routing_flow(
            layer_probs_list, labels,
            save_name='06_cross_layer_routing.png'
        )
    else:
        print("\n6. Cross-Layer Routing Flow... Skipped (not enough matching layers)")
    
    # 7. 与随机/均匀路由对比
    print("\n7. Ablation Comparison (vs Random & Uniform)...")
    random_probs = torch.softmax(torch.randn_like(main_probs), dim=-1)
    uniform_probs = torch.ones_like(main_probs) / main_probs.shape[-1]
    
    viz.plot_ablation_comparison({
        'Sentiment-Aware': main_probs,
        'Random': random_probs,
        'Uniform': uniform_probs
    }, save_name='07_ablation_comparison.png')
    
    print("\n" + "="*50)
    print(f"All visualizations saved to: {save_dir}")
    print("="*50)
    
    # 打印统计信息
    print("\n--- Statistics ---")
    print(f"Total samples: {len(labels)}")
    print(f"Class distribution: Negative={sum(labels==0)}, Neutral={sum(labels==1)}, Positive={sum(labels==2)}")
    accuracy = (data['predictions'] == labels).mean() * 100
    print(f"Model Accuracy: {accuracy:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Router Visualization')
    
    # 模型路径
    parser.add_argument('--load_path', type=str, default="/home/ningwang/SSD1T/cts/model_reproduction/D2R/output/HFM/11_best_model.pth", required=False,
                        help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--bert_name', default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/bert-base-uncased", type=str, required=False,
                        help='Path to BERT model')
    parser.add_argument('--vit_name', default="/home/ningwang/SSD1T/cts/MMSA/pretrained_berts/clip-vit-base-patch32", type=str, required=False,
                        help='Path to ViT/CLIP model')
    
    # 数据路径
    parser.add_argument('--img_path', type=str, default='../data/HFM/dataset_image', help='Path to image directory')
    
    # 其他参数
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_seq', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--save_dir', type=str, default='./visualization', help='Directory to save visualizations')
    
    # 模型配置参数（需要与训练时一致）
    parser.add_argument('--seed', default=2023, type=int, help="random seed")
    parser.add_argument('--ignore_idx', default=0, type=int, help="Specify the index to be ignored")
    
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
    parser.add_argument('--hid_IMRC', type=int, default=768, help='Hidden size of FeedForward in Intra-Modal Reasoning Cell')
    parser.add_argument('--raw_feature_norm_CMRC', default="clipped_l2norm", help='clipped_l2norm|l2norm|clipped_l1norm|l1norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax_CMRC', default=4., type=float, help='Attention softmax temperature.')
    parser.add_argument('--hid_router', type=int, default=768, help='Hidden size of MLP in routers')
    parser.add_argument('--num_labels', type=int, default=3)
    
    args = parser.parse_args()
    
    # 检查CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载模型
    model = load_model(args)
    
    # 注册hooks
    routing_hook = RoutingHook()
    routing_hook.register_hooks(model)
    
    # 加载数据
    print("\nLoading test data...")
    
    # 数据路径字典
    # data_path = {
    #     'train': '../data/MVSA-single/10-flod-1/train.json',
    #     'dev': '../data/MVSA-single/10-flod-1/dev.json',
    #     'test': '../data/MVSA-single/10-flod-1/test.json'
    # }
    data_path = {
        'train': '../data/HFM/train.json',
        'dev': '../data/HFM/valid.json',
        'test': '../data/HFM/test.json'
    }
    
    # 加载CLIP processor
    clip_processor = CLIPProcessor.from_pretrained(args.vit_name)
    
    processor = MSDProcessor(data_path, args.bert_name, clip_processor)
    test_dataset = MSDDataset(processor, args.img_path, args.max_seq, 'test')
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # 收集数据
    print("\nCollecting routing data...")
    data = collect_data(model, test_loader, args.device, routing_hook)
    
    # 保存原始数据
    data_save_path = os.path.join(args.save_dir, 'routing_data.pkl')
    with open(data_save_path, 'wb') as f:
        save_data = {k: v.numpy() if isinstance(v, torch.Tensor) else v 
                     for k, v in data.items()}
        # path_probs需要特殊处理
        save_data['path_probs'] = {k: v.numpy() for k, v in data['path_probs'].items()}
        pickle.dump(save_data, f)
    print(f"Routing data saved to: {data_save_path}")
    
    # 运行可视化
    run_visualization(data, args.save_dir)
    
    # 清理
    routing_hook.remove_hooks()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
