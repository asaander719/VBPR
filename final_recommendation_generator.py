import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import argparse
from Models.VBPR import VBPR

class OutfitRecommendationGenerator:
    """
    为新的服装数据集生成top5搭配推荐的类
    
    使用预训练的VBPR模型和新的ResNet152特征来生成推荐
    """
    
    def __init__(self, model_path, new_features_path, device=None):
        """
        初始化推荐生成器
        
        Args:
            model_path: 训练好的VBPR模型路径 (VBPR.pth.tar)
            new_features_path: 新的特征文件路径 (resnet152_features.pth)
            device: 计算设备 ('cpu' 或 'cuda')
        """
        # 设置设备
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"使用设备: {self.device}")
        
        # 加载新的特征数据
        print("正在加载新的特征数据...")
        self.new_features = torch.load(new_features_path, map_location='cpu')
        print(f"已加载 {len(self.new_features)} 个物品的特征 (ID范围: {min(self.new_features.keys())}-{max(self.new_features.keys())})")
        
        # 准备特征tensor
        self.feature_tensor = self._prepare_feature_tensor()
        
        # 加载预训练模型
        print("正在加载预训练的VBPR模型...")
        self._load_pretrained_model(model_path)
        
        # 为新数据创建embedding
        self._create_new_item_embeddings()
        
    def _prepare_feature_tensor(self):
        """将特征字典转换为tensor格式"""
        max_id = max(self.new_features.keys())
        min_id = min(self.new_features.keys())
        
        # 获取特征维度
        sample_feature = list(self.new_features.values())[0]
        if isinstance(sample_feature, torch.Tensor):
            feature_dim = sample_feature.shape[0]
        elif isinstance(sample_feature, (list, np.ndarray)):
            feature_dim = len(sample_feature)
        else:
            raise ValueError(f"不支持的特征类型: {type(sample_feature)}")
        
        print(f"特征维度: {feature_dim}")
        
        # 创建特征tensor
        feature_tensor = torch.zeros(max_id + 1, feature_dim)
        
        for item_id, feature in self.new_features.items():
            if isinstance(feature, torch.Tensor):
                feature_tensor[item_id] = feature
            elif isinstance(feature, (list, np.ndarray)):
                feature_tensor[item_id] = torch.tensor(feature, dtype=torch.float32)
            else:
                raise ValueError(f"不支持的特征类型: {type(feature)}")
                
        return feature_tensor
    
    def _load_pretrained_model(self, model_path):
        """加载预训练模型"""
        try:
            # 添加安全加载设置
            torch.serialization.add_safe_globals([VBPR])
            self.original_model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 获取原始模型参数
            self.hidden_dim = self.original_model.hidden_dim
            self.original_item_num = self.original_model.item_num
            
            print(f"原始模型参数: hidden_dim={self.hidden_dim}, item_num={self.original_item_num}")
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
    
    def _create_new_item_embeddings(self):
        """为新的items创建embedding"""
        num_new_items = len(self.new_features)
        visual_feature_dim = self.feature_tensor.shape[1]
        
        print(f"为 {num_new_items} 个新物品创建embedding...")
        
        # 创建新的VBPR模型实例
        self.model = VBPR(
            item_num=num_new_items,
            hidden_dim=self.hidden_dim,
            visual_feature_dim=visual_feature_dim,
            visual_features=self.feature_tensor,
            with_Nor=True
        )
        
        # 复制预训练模型的权重（除了item相关的embedding）
        self._transfer_model_weights()
        
        # 移动到设备
        self.model.to(self.device)
        self.model.eval()
        
        print("新embedding创建完成")
    
    def _transfer_model_weights(self):
        """从原始模型转移可重用的权重"""
        try:
            # 转移visual相关的权重
            if hasattr(self.original_model, 'visual_trans') and hasattr(self.model, 'visual_trans'):
                # 检查维度是否匹配
                if (self.original_model.visual_trans.weight.shape == self.model.visual_trans.weight.shape):
                    self.model.visual_trans.weight.data = self.original_model.visual_trans.weight.data.clone()
                    if self.original_model.visual_trans.bias is not None:
                        self.model.visual_trans.bias.data = self.original_model.visual_trans.bias.data.clone()
                    print("已转移visual transformation权重")
                else:
                    print("Visual transformation维度不匹配，使用随机初始化")
            
            # 转移其他可能的共享权重
            # 这里可以根据具体的VBPR实现添加更多权重转移逻辑
            
        except Exception as e:
            print(f"权重转移时出现警告: {e}")
            print("将使用随机初始化的权重")
    
    def compute_compatibility_scores(self, batch_size=64):
        """计算所有item pairs的兼容性分数"""
        item_ids = list(self.new_features.keys())
        n_items = len(item_ids)
        
        print(f"计算 {n_items}x{n_items} 的兼容性矩阵...")
        
        # 创建兼容性矩阵
        compatibility_matrix = torch.zeros(n_items, n_items)
        
        with torch.no_grad():
            for i in range(0, n_items, batch_size):
                end_i = min(i + batch_size, n_items)
                batch_tops = torch.tensor(item_ids[i:end_i], dtype=torch.long).to(self.device)
                
                for j in range(0, n_items, batch_size):
                    end_j = min(j + batch_size, n_items)
                    batch_bottoms = torch.tensor(item_ids[j:end_j], dtype=torch.long).to(self.device)
                    
                    # 创建所有pairs
                    tops_expanded = batch_tops.unsqueeze(1).expand(-1, len(batch_bottoms))
                    bottoms_expanded = batch_bottoms.unsqueeze(0).expand(len(batch_tops), -1)
                    
                    # 展平
                    tops_flat = tops_expanded.flatten()
                    bottoms_flat = bottoms_expanded.flatten()
                    
                    # 计算兼容性分数
                    try:
                        scores = self.model.forward(tops_flat, bottoms_flat)
                        scores = scores.view(len(batch_tops), len(batch_bottoms))
                        compatibility_matrix[i:end_i, j:end_j] = scores.cpu()
                    except Exception as e:
                        print(f"计算兼容性分数时出错: {e}")
                        # 使用简化的相似度计算
                        scores = self._compute_simple_similarity(tops_flat, bottoms_flat)
                        scores = scores.view(len(batch_tops), len(batch_bottoms))
                        compatibility_matrix[i:end_i, j:end_j] = scores.cpu()
                
                # 显示进度
                if (i // batch_size + 1) % 10 == 0:
                    progress = (i + batch_size) / n_items * 100
                    print(f"进度: {progress:.1f}%")
        
        return compatibility_matrix, item_ids
    
    def _compute_simple_similarity(self, item1_ids, item2_ids):
        """计算简化的相似度（基于visual特征）"""
        # 获取visual特征
        item1_features = self.model.visual_features[item1_ids]
        item2_features = self.model.visual_features[item2_ids]
        
        # 计算余弦相似度
        similarity = F.cosine_similarity(item1_features, item2_features, dim=-1)
        
        return similarity
    
    def generate_recommendations(self, top_k=5):
        """为每个item生成top-k推荐"""
        compatibility_matrix, item_ids = self.compute_compatibility_scores()
        
        print(f"生成top-{top_k}推荐...")
        
        recommendations = {}
        
        for i, item_id in enumerate(item_ids):
            # 获取该item与所有其他items的兼容性分数
            scores = compatibility_matrix[i]
            
            # 排除自己
            scores[i] = float('-inf')
            
            # 获取top-k
            actual_k = min(top_k, len(item_ids) - 1)
            if actual_k > 0:
                top_k_indices = torch.topk(scores, k=actual_k).indices
                top_k_items = [item_ids[idx.item()] for idx in top_k_indices]
                top_k_scores = [scores[idx.item()].item() for idx in top_k_indices]
            else:
                top_k_items = []
                top_k_scores = []
            
            recommendations[item_id] = {
                'recommended_items': top_k_items,
                'compatibility_scores': top_k_scores
            }
        
        return recommendations
    
    def save_to_excel(self, recommendations, output_path='outfit_recommendations.xlsx'):
        """将推荐结果保存为Excel文件"""
        print(f"保存推荐结果到 {output_path}...")
        
        # 准备数据
        data = []
        for item_id, rec in recommendations.items():
            row = {'Item_ID': item_id}
            
            # 添加top5推荐和分数
            for i in range(5):
                if i < len(rec['recommended_items']):
                    row[f'Top{i+1}_Match'] = rec['recommended_items'][i]
                    row[f'Top{i+1}_Score'] = round(rec['compatibility_scores'][i], 4)
                else:
                    row[f'Top{i+1}_Match'] = None
                    row[f'Top{i+1}_Score'] = None
            
            # 添加所有推荐的汇总
            row['All_Matches'] = ', '.join(map(str, rec['recommended_items']))
            
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df = df.sort_values('Item_ID')  # 按ID排序
        
        # 保存为Excel
        df.to_excel(output_path, index=False)
        
        print(f"成功保存 {len(data)} 个推荐到 {output_path}")
        
        # 显示示例结果
        print("\n推荐示例:")
        for i in range(min(10, len(data))):
            item_id = df.iloc[i]['Item_ID']
            matches = df.iloc[i]['All_Matches']
            print(f"物品 {item_id} 的最佳搭配: [{matches}]")
        
        return df

def main():
    parser = argparse.ArgumentParser(description='生成服装搭配推荐')
    parser.add_argument('--model_path', 
                       default='VBPR/saved/IQON3000/VBPR/VBPR.pth.tar',
                       help='VBPR模型文件路径')
    parser.add_argument('--features_path', 
                       default='resnet152_features.pth',
                       help='ResNet152特征文件路径')
    parser.add_argument('--output_path', 
                       default='outfit_recommendations.xlsx',
                       help='输出Excel文件路径')
    parser.add_argument('--device', 
                       default=None,
                       help='计算设备 (cpu/cuda)')
    parser.add_argument('--top_k', 
                       type=int, default=5,
                       help='每个物品推荐的数量')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 错误: 模型文件未找到 {args.model_path}")
        print("请确保训练好的VBPR模型文件存在")
        return
    
    if not os.path.exists(args.features_path):
        print(f"❌ 错误: 特征文件未找到 {args.features_path}")
        print("请确保resnet152_features.pth文件存在")
        return
    
    try:
        print("🚀 开始生成服装搭配推荐...")
        
        # 创建推荐生成器
        generator = OutfitRecommendationGenerator(
            args.model_path, 
            args.features_path,
            args.device
        )
        
        # 生成推荐
        recommendations = generator.generate_recommendations(top_k=args.top_k)
        
        # 保存结果
        df = generator.save_to_excel(recommendations, args.output_path)
        
        print(f"\n✅ 成功为 {len(recommendations)} 个物品生成推荐!")
        print(f"📊 结果已保存到: {args.output_path}")
        print(f"📈 每个物品推荐了 top-{args.top_k} 个最搭配的服装")
        
    except Exception as e:
        print(f"❌ 生成推荐时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()