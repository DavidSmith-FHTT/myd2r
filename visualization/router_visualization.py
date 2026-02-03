"""
Router机制可视化实验
证明动态路由机制的有效性

可视化内容：
1. 路径概率热力图 - 展示每层各Cell的路径分配
2. 类别条件路径分析 - 不同情感类别的路径偏好
3. 路径熵分析 - 路由决策的确定性
4. 跨层路径流动 - Sankey图展示层间路由
5. t-SNE可视化 - 路径向量在特征空间的分布
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RouterVisualization:
    """Router机制可视化工具类"""
    
    def __init__(self, save_dir='visualization_results'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Cell名称映射
        self.cell_names = [
            'SSRU',
            'USRU',
            'CLSMU',
            'CGSAU',
            'GLSFU',
            'MVSMU',
        ]
        
        # 情感类别名称
        self.sentiment_names = ['Negative', 'Neutral', 'Positive']
        
        # 颜色方案
        self.colors = {
            'negative': '#E74C3C',
            'neutral': '#3498DB', 
            'positive': '#2ECC71'
        }
        
    def collect_routing_data(self, model, dataloader, device):
        """
        收集模型推理过程中的路由数据
        
        Returns:
            routing_data: dict containing path probabilities per layer and sample info
        """
        model.eval()
        routing_data = {
            'layer0_probs': [],      # 第一层路径概率
            'layer1_probs': [],      # 中间层路径概率
            'layer2_probs': [],      # 最后层路径概率
            'labels': [],            # 真实标签
            'predictions': [],       # 预测标签
            'confidences': [],       # 预测置信度
        }
        
        # 需要hook来捕获中间层的路径概率
        layer_probs = {}
        
        def make_hook(name):
            def hook(module, input, output):
                # output = (aggr_res_lst, all_path_prob)
                if isinstance(output, tuple) and len(output) == 2:
                    layer_probs[name] = output[1].detach().cpu()
            return hook
        
        # 注册hooks (需要根据实际模型结构调整)
        hooks = []
        
        with torch.no_grad():
            for batch in dataloader:
                # 根据你的数据加载器结构调整
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播并收集路径概率
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    return_routing_probs=True  # 需要模型支持返回路径概率
                )
                
                # 收集数据
                if hasattr(outputs, 'routing_probs'):
                    for layer_name, probs in outputs.routing_probs.items():
                        if layer_name not in routing_data:
                            routing_data[layer_name] = []
                        routing_data[layer_name].append(probs.cpu())
                
                routing_data['labels'].extend(labels.cpu().numpy())
                
                if hasattr(outputs, 'logits'):
                    probs = torch.softmax(outputs.logits, dim=-1)
                    preds = torch.argmax(probs, dim=-1)
                    confs = torch.max(probs, dim=-1)[0]
                    routing_data['predictions'].extend(preds.cpu().numpy())
                    routing_data['confidences'].extend(confs.cpu().numpy())
        
        # 移除hooks
        for h in hooks:
            h.remove()
            
        return routing_data
    
    def plot_path_probability_heatmap(self, path_probs, layer_name='Layer0', 
                                       figsize=(12, 6), save_name=None):
        """
        绘制路径概率热力图
        
        Args:
            path_probs: (num_samples, num_cells) 或 (num_samples, num_out_path, num_cells) 路径概率张量
        """
        # 计算平均路径概率
        if isinstance(path_probs, list):
            path_probs = torch.cat(path_probs, dim=0)
        
        # 处理2D和3D张量
        if path_probs.dim() == 2:
            # 2D: (samples, num_cells) -> 直接计算每个cell的平均概率
            mean_probs = path_probs.mean(dim=0).numpy()  # (num_cells,)
            num_cells = len(mean_probs)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # 使用条形图而不是热力图
            x = np.arange(num_cells)
            bars = ax.bar(x, mean_probs, color='steelblue', alpha=0.7, edgecolor='black')
            
            # 在每个条上标注数值
            for i, (bar, val) in enumerate(zip(bars, mean_probs)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.cell_names[:num_cells], rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Average Path Probability', fontsize=12)
            ax.set_title(f'Path Probability Distribution - {layer_name}', fontsize=14)
            ax.set_ylim(0, max(mean_probs) * 1.2)
        else:
            # 3D: (samples, num_out_path, num_cells)
            mean_probs = path_probs.mean(dim=0).numpy()  # (num_out_path, num_cells)
            
            fig, ax = plt.subplots(figsize=figsize)
            
            sns.heatmap(mean_probs, 
                        annot=True, 
                        fmt='.5f',
                        cmap='YlOrRd',
                        xticklabels=self.cell_names[:mean_probs.shape[1]],
                        yticklabels=[f'Path {i+1}' for i in range(mean_probs.shape[0])],
                        ax=ax,
                        cbar_kws={'label': 'Probability'})
            
            # ax.set_xlabel('Processing Cells', fontsize=12)
            # ax.set_ylabel('Output Paths', fontsize=12)
            # ax.set_title(f'Path Probability Distribution - {layer_name}', fontsize=14)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_class_conditional_routing(self, path_probs, labels, layer_name='Layer0',
                                        figsize=(14, 5), save_name=None):
        """
        绘制类别条件下的路径分布对比
        
        证明：不同情感类别有不同的路径偏好
        """
        if isinstance(path_probs, list):
            path_probs = torch.cat(path_probs, dim=0)
        
        labels = np.array(labels)
        num_cells = path_probs.shape[-1]
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for cls_idx, (cls_name, color) in enumerate(zip(self.sentiment_names, 
                                                         self.colors.values())):
            ax = axes[cls_idx]
            
            # 获取该类别的样本
            mask = labels == cls_idx
            cls_probs = path_probs[mask]
            
            if len(cls_probs) == 0:
                continue
                
            # 计算每个Cell的平均路径概率
            if cls_probs.dim() == 2:
                # 2D: (samples, num_cells)
                mean_probs = cls_probs.mean(dim=0).numpy()
                std_probs = cls_probs.std(dim=0).numpy()
            else:
                # 3D: (samples, num_out_path, num_cells)
                mean_probs = cls_probs.mean(dim=(0, 1)).numpy()
                std_probs = cls_probs.std(dim=(0, 1)).numpy()
            
            x = np.arange(num_cells)
            bars = ax.bar(x, mean_probs, yerr=std_probs, color=color, 
                         alpha=0.7, capsize=3, edgecolor='black', linewidth=1)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.cell_names[:num_cells], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Path Probability', fontsize=11)
            ax.set_title(f'{cls_name} Samples\n(n={mask.sum()})', fontsize=12)
            ax.set_ylim(0, max(mean_probs) * 1.3)
            
            # 标注最高概率的Cell
            max_idx = np.argmax(mean_probs)
            ax.annotate(f'{mean_probs[max_idx]:.3f}', 
                       xy=(max_idx, mean_probs[max_idx]),
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Class-Conditional Path Distribution - {layer_name}', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_routing_entropy(self, path_probs, labels, figsize=(10, 6), save_name=None):
        """
        绘制路由熵分析
        
        低熵 = 路由决策更确定（更集中于某些Cell）
        高熵 = 路由决策更分散（均匀分配）
        
        有效的Router应该对不同样本有不同的熵值
        """
        if isinstance(path_probs, list):
            path_probs = torch.cat(path_probs, dim=0)
        
        labels = np.array(labels)
        
        # 计算每个样本的路由熵 (在Cell维度上)
        probs_np = path_probs.numpy()
        
        # 对每个样本计算熵
        sample_entropies = []
        for i in range(len(probs_np)):
            if probs_np.ndim == 2:
                # 2D: (samples, num_cells) - 直接计算熵
                ent = entropy(probs_np[i] + 1e-8)
            else:
                # 3D: (samples, num_out_path, num_cells) - 在output path上平均后计算熵
                ent = entropy(probs_np[i].mean(axis=0) + 1e-8)
            sample_entropies.append(ent)
        sample_entropies = np.array(sample_entropies)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 图1: 按类别的熵分布
        ax1 = axes[0]
        entropy_by_class = [sample_entropies[labels == i] for i in range(3)]
        
        bp = ax1.boxplot(entropy_by_class, labels=self.sentiment_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], self.colors.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Routing Entropy', fontsize=12)
        ax1.set_title('Routing Entropy by Sentiment Class', fontsize=12)
        ax1.axhline(y=np.log(path_probs.shape[-1]), color='gray', linestyle='--', 
                   label='Max Entropy (uniform)')
        ax1.legend()
        
        # 图2: 熵的整体分布
        ax2 = axes[1]
        ax2.hist(sample_entropies, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.log(path_probs.shape[-1]), color='red', linestyle='--', 
                   label=f'Max Entropy = {np.log(path_probs.shape[-1]):.2f}')
        ax2.axvline(x=sample_entropies.mean(), color='green', linestyle='-', 
                   label=f'Mean Entropy = {sample_entropies.mean():.2f}')
        ax2.set_xlabel('Routing Entropy', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Distribution of Routing Entropy', fontsize=12)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_routing_vs_confidence(self, path_probs, confidences, figsize=(10, 5), save_name=None):
        """
        绘制路由确定性与预测置信度的关系
        
        假设：高置信度预测 ↔ 更确定的路由决策（低熵）
        """
        if isinstance(path_probs, list):
            path_probs = torch.cat(path_probs, dim=0)
        
        probs_np = path_probs.numpy()
        confidences = np.array(confidences)
        
        # 计算路由熵
        sample_entropies = []
        for i in range(len(probs_np)):
            if probs_np.ndim == 2:
                # 2D: (samples, num_cells)
                ent = entropy(probs_np[i] + 1e-8)
            else:
                # 3D: (samples, num_out_path, num_cells)
                ent = entropy(probs_np[i].mean(axis=0) + 1e-8)
            sample_entropies.append(ent)
        sample_entropies = np.array(sample_entropies)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 散点图
        ax1 = axes[0]
        scatter = ax1.scatter(confidences, sample_entropies, alpha=0.5, c='steelblue', s=20)
        
        # 添加趋势线
        z = np.polyfit(confidences, sample_entropies, 1)
        p = np.poly1d(z)
        x_line = np.linspace(confidences.min(), confidences.max(), 100)
        ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label=f'Trend (slope={z[0]:.3f})')
        
        ax1.set_xlabel('Prediction Confidence', fontsize=12)
        ax1.set_ylabel('Routing Entropy', fontsize=12)
        ax1.set_title('Routing Entropy vs Prediction Confidence', fontsize=12)
        ax1.legend()
        
        # 分组对比
        ax2 = axes[1]
        high_conf = sample_entropies[confidences > np.median(confidences)]
        low_conf = sample_entropies[confidences <= np.median(confidences)]
        
        bp = ax2.boxplot([low_conf, high_conf], 
                         labels=['Low Confidence', 'High Confidence'],
                         patch_artist=True)
        bp['boxes'][0].set_facecolor('#E74C3C')
        bp['boxes'][1].set_facecolor('#2ECC71')
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax2.set_ylabel('Routing Entropy', fontsize=12)
        ax2.set_title('Routing Entropy by Confidence Level', fontsize=12)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_layer_routing_flow(self, layer_probs_list, labels, figsize=(14, 8), save_name=None):
        """
        绘制跨层路由流动图 (Sankey-style)
        
        展示从Layer0 → Layer1 → Layer2的路径概率变化
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        labels = np.array(labels)
        
        for cls_idx, cls_name in enumerate(self.sentiment_names):
            ax = axes[cls_idx]
            mask = labels == cls_idx
            
            layer_means = []
            for layer_probs in layer_probs_list:
                if isinstance(layer_probs, list):
                    layer_probs = torch.cat(layer_probs, dim=0)
                cls_probs = layer_probs[mask]
                if cls_probs.dim() == 2:
                    mean_probs = cls_probs.mean(dim=0).numpy()
                else:
                    mean_probs = cls_probs.mean(dim=(0, 1)).numpy()
                layer_means.append(mean_probs)
            
            num_cells = len(layer_means[0])
            x = np.arange(len(layer_means))
            
            for cell_idx in range(num_cells):
                values = [lm[cell_idx] for lm in layer_means]
                ax.plot(x, values, 'o-', linewidth=2, markersize=8,
                       label=self.cell_names[cell_idx].replace('\n', ' '))
            
            ax.set_xticks(x)
            ax.set_xticklabels([f'Layer {i}' for i in range(len(layer_means))])
            ax.set_ylabel('Path Probability', fontsize=11)
            ax.set_title(f'{cls_name}', fontsize=12)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Cross-Layer Routing Flow by Sentiment Class', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_routing_tsne(self, path_probs, labels, figsize=(10, 8), save_name=None):
        """
        使用t-SNE可视化路径概率向量
        
        如果Router有效，同类样本的路径向量应该聚集在一起
        """
        if isinstance(path_probs, list):
            path_probs = torch.cat(path_probs, dim=0)
        
        # 将路径概率展平为向量
        routing_vectors = path_probs.view(path_probs.shape[0], -1).numpy()
        labels = np.array(labels)
        
        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(routing_vectors)-1))
        routing_2d = tsne.fit_transform(routing_vectors)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for cls_idx, (cls_name, color) in enumerate(zip(self.sentiment_names, 
                                                         self.colors.values())):
            mask = labels == cls_idx
            ax.scatter(routing_2d[mask, 0], routing_2d[mask, 1], 
                      c=color, label=cls_name, alpha=0.6, s=30)
        
        # ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        # ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        # ax.set_title('t-SNE Visualization of Routing Vectors', fontsize=14)
        # ax.legend(fontsize=11)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def plot_ablation_comparison(self, routing_data_dict, figsize=(14, 5), save_name=None):
        """
        对比不同Router版本的路径分布
        
        Args:
            routing_data_dict: {
                'Sentiment-Aware': path_probs,
                'Random': path_probs,
                'Uniform': path_probs
            }
        """
        fig, axes = plt.subplots(1, len(routing_data_dict), figsize=figsize)
        
        for idx, (router_name, path_probs) in enumerate(routing_data_dict.items()):
            ax = axes[idx]
            
            if isinstance(path_probs, list):
                path_probs = torch.cat(path_probs, dim=0)
            
            if path_probs.dim() == 2:
                mean_probs = path_probs.mean(dim=0).numpy()
                std_probs = path_probs.std(dim=0).numpy()
            else:
                mean_probs = path_probs.mean(dim=(0, 1)).numpy()
                std_probs = path_probs.std(dim=(0, 1)).numpy()
            
            num_cells = len(mean_probs)
            x = np.arange(num_cells)
            
            bars = ax.bar(x, mean_probs, yerr=std_probs, 
                         color='steelblue', alpha=0.7, capsize=3,
                         edgecolor='black', linewidth=1)
            
            ax.set_xticks(x)
            ax.set_xticklabels(self.cell_names[:num_cells], rotation=45, ha='right', fontsize=9)
            ax.set_ylabel('Path Probability', fontsize=11)
            ax.set_title(f'{router_name} Router', fontsize=12)
            
            # 计算并显示熵
            ent = entropy(mean_probs + 1e-8)
            ax.text(0.95, 0.95, f'Entropy: {ent:.3f}', transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        fig.suptitle('Router Ablation Comparison', fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(os.path.join(self.save_dir, save_name), dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig


def generate_demo_data(num_samples=500, num_cells=6, num_out_path=4):
    """
    生成演示数据用于展示可视化效果
    模拟真实的Router行为：不同类别有不同的路径偏好
    """
    labels = np.random.choice([0, 1, 2], size=num_samples, p=[0.3, 0.4, 0.3])
    
    # 模拟不同类别的路径偏好
    class_preferences = {
        0: [0.3, 0.1, 0.1, 0.25, 0.15, 0.1],  # Negative: 偏好RIC和CMRC
        1: [0.15, 0.15, 0.2, 0.15, 0.15, 0.2], # Neutral: 较均匀
        2: [0.1, 0.25, 0.15, 0.1, 0.2, 0.2],   # Positive: 偏好GLAC和CRCMC
    }
    
    path_probs = []
    for label in labels:
        base_probs = np.array(class_preferences[label])
        noise = np.random.normal(0, 0.03, size=num_cells)
        probs = base_probs + noise
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum()
        
        # 扩展到 (num_out_path, num_cells)
        sample_probs = np.tile(probs, (num_out_path, 1))
        sample_probs += np.random.normal(0, 0.02, size=sample_probs.shape)
        sample_probs = np.clip(sample_probs, 0.01, None)
        sample_probs = sample_probs / sample_probs.sum(axis=1, keepdims=True)
        
        path_probs.append(sample_probs)
    
    path_probs = torch.tensor(np.array(path_probs), dtype=torch.float32)
    
    # 生成置信度（与熵负相关）
    confidences = []
    for i, label in enumerate(labels):
        ent = entropy(path_probs[i].mean(dim=0).numpy() + 1e-8)
        conf = 0.9 - 0.3 * (ent / np.log(num_cells)) + np.random.normal(0, 0.05)
        confidences.append(np.clip(conf, 0.4, 0.99))
    
    return path_probs, labels, np.array(confidences)


if __name__ == '__main__':
    # 生成演示数据
    print("生成演示数据...")
    path_probs, labels, confidences = generate_demo_data(num_samples=500)
    
    # 创建可视化工具
    viz = RouterVisualization(save_dir='visualization_results')
    
    print("\n1. 绘制路径概率热力图...")
    viz.plot_path_probability_heatmap(path_probs, layer_name='Layer0',
                                       save_name='01_path_prob_heatmap.png')
    
    print("\n2. 绘制类别条件路径分布...")
    viz.plot_class_conditional_routing(path_probs, labels, layer_name='Layer0',
                                        save_name='02_class_conditional_routing.png')
    
    print("\n3. 绘制路由熵分析...")
    viz.plot_routing_entropy(path_probs, labels,
                             save_name='03_routing_entropy.png')
    
    print("\n4. 绘制路由-置信度关系...")
    viz.plot_routing_vs_confidence(path_probs, confidences,
                                   save_name='04_routing_vs_confidence.png')
    
    print("\n5. 绘制t-SNE可视化...")
    viz.plot_routing_tsne(path_probs, labels,
                          save_name='05_routing_tsne.png')
    
    # 生成消融对比数据
    print("\n6. 绘制消融实验对比...")
    random_probs = torch.softmax(torch.randn_like(path_probs), dim=-1)
    uniform_probs = torch.ones_like(path_probs) / path_probs.shape[-1]
    
    viz.plot_ablation_comparison({
        'Sentiment-Aware': path_probs,
        'Random': random_probs,
        'Uniform': uniform_probs
    }, save_name='06_ablation_comparison.png')
    
    print("\n可视化完成！结果保存在 visualization_results/ 目录下")
