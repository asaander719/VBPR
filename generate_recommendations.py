import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from Models.VBPR import VBPR
import os

class RecommendationGenerator:
    def __init__(self, model_path, new_features_path):
        """
        初始化推荐生成器
        
        Args:
            model_path: 训练好的VBPR模型路径
            new_features_path: 新的特征文件路径 (resnet152_features.pth)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载新的特征数据
        print("Loading new features...")
        self.new_features = torch.load(new_features_path, map_location='cpu')
        print(f"Loaded features for {len(self.new_features)} items (IDs: {min(self.new_features.keys())}-{max(self.new_features.keys())})")
        
        # 将特征转换为tensor格式
        self.feature_tensor = self.prepare_feature_tensor()
        
        # 加载预训练模型
        print("Loading pre-trained VBPR model...")
        # 添加安全加载设置
        torch.serialization.add_safe_globals([VBPR])
        self.model = torch.load(model_path, map_location='cpu', weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        # 获取模型参数
        self.hidden_dim = self.model.hidden_dim
        self.visual_feature_dim = self.feature_tensor.shape[1]
        
        # 创建新的item embedding
        self.create_new_item_embeddings()
        
    def prepare_feature_tensor(self):
        """将特征字典转换为tensor"""
        # 确保ID是连续的0-200
        max_id = max(self.new_features.keys())
        min_id = min(self.new_features.keys())
        
        print(f"Feature IDs range: {min_id} to {max_id}")
        
        # 创建特征tensor
        feature_dim = len(list(self.new_features.values())[0])
        feature_tensor = torch.zeros(max_id + 1, feature_dim)
        
        for item_id, feature in self.new_features.items():
            if isinstance(feature, list):
                feature_tensor[item_id] = torch.tensor(feature)
            else:
                feature_tensor[item_id] = feature
                
        return feature_tensor
    
    def create_new_item_embeddings(self):
        """为新的items创建embedding"""
        num_new_items = len(self.new_features)
        
        # 创建新的item embedding (随机初始化)
        new_item_emb = F.normalize(
            torch.normal(
                mean=torch.zeros(num_new_items + 1, self.hidden_dim), 
                std=1/(self.hidden_dim)**0.5
            ), 
            p=2, dim=-1
        )
        
        # 创建新的bias
        new_item_bias = torch.zeros([num_new_items + 1, 1])
        
        # 更新模型的embedding层
        self.model.item_embs = torch.nn.Embedding.from_pretrained(
            new_item_emb, freeze=False, padding_idx=num_new_items
        )
        self.model.item_bias = torch.nn.Embedding.from_pretrained(
            new_item_bias, freeze=False, padding_idx=num_new_items
        )
        self.model.item_bias_v = torch.nn.Embedding.from_pretrained(
            new_item_bias, freeze=False, padding_idx=num_new_items
        )
        
        # 更新visual features
        self.model.visual_features = self.feature_tensor
        self.model.item_num = num_new_items
        
        # 移动到设备
        self.model.to(self.device)
        
        print(f"Created new embeddings for {num_new_items} items")
    
    def compute_similarity_matrix(self):
        """计算所有item pairs的相似度矩阵"""
        item_ids = list(self.new_features.keys())
        n_items = len(item_ids)
        
        print("Computing similarity matrix...")
        
        # 创建相似度矩阵
        similarity_matrix = torch.zeros(n_items, n_items)
        
        with torch.no_grad():
            # 批量计算相似度
            batch_size = 32
            
            for i in range(0, n_items, batch_size):
                end_i = min(i + batch_size, n_items)
                batch_tops = torch.tensor(item_ids[i:end_i]).to(self.device)
                
                for j in range(0, n_items, batch_size):
                    end_j = min(j + batch_size, n_items)
                    batch_bottoms = torch.tensor(item_ids[j:end_j]).to(self.device)
                    
                    # 计算所有pairs的相似度
                    tops_expanded = batch_tops.unsqueeze(1).expand(-1, len(batch_bottoms))
                    bottoms_expanded = batch_bottoms.unsqueeze(0).expand(len(batch_tops), -1)
                    
                    # 展平进行计算
                    tops_flat = tops_expanded.flatten()
                    bottoms_flat = bottoms_expanded.flatten()
                    
                    # 使用模型的forward方法计算相似度
                    similarities = self.model.forward(tops_flat, bottoms_flat)
                    
                    # 重新reshape
                    similarities = similarities.view(len(batch_tops), len(batch_bottoms))
                    
                    # 存储结果
                    similarity_matrix[i:end_i, j:end_j] = similarities.cpu()
                
                if (i // batch_size + 1) % 10 == 0:
                    print(f"Processed {i + batch_size}/{n_items} items")
        
        return similarity_matrix, item_ids
    
    def generate_top5_recommendations(self):
        """为每个item生成top5推荐"""
        similarity_matrix, item_ids = self.compute_similarity_matrix()
        
        print("Generating top-5 recommendations...")
        
        recommendations = {}
        
        for i, item_id in enumerate(item_ids):
            # 获取该item与所有其他items的相似度
            similarities = similarity_matrix[i]
            
            # 排除自己
            similarities[i] = float('-inf')
            
            # 获取top5
            top5_indices = torch.topk(similarities, k=min(5, len(item_ids)-1)).indices
            top5_items = [item_ids[idx.item()] for idx in top5_indices]
            top5_scores = [similarities[idx.item()].item() for idx in top5_indices]
            
            recommendations[item_id] = {
                'recommended_items': top5_items,
                'similarity_scores': top5_scores
            }
        
        return recommendations
    
    def save_to_excel(self, recommendations, output_path='outfit_recommendations.xlsx'):
        """将推荐结果保存为Excel文件"""
        print(f"Saving recommendations to {output_path}...")
        
        # 准备数据
        data = []
        for item_id, rec in recommendations.items():
            row = {
                'Item_ID': item_id,
                'Top1_Match': rec['recommended_items'][0] if len(rec['recommended_items']) > 0 else None,
                'Top1_Score': rec['similarity_scores'][0] if len(rec['similarity_scores']) > 0 else None,
                'Top2_Match': rec['recommended_items'][1] if len(rec['recommended_items']) > 1 else None,
                'Top2_Score': rec['similarity_scores'][1] if len(rec['similarity_scores']) > 1 else None,
                'Top3_Match': rec['recommended_items'][2] if len(rec['recommended_items']) > 2 else None,
                'Top3_Score': rec['similarity_scores'][2] if len(rec['similarity_scores']) > 2 else None,
                'Top4_Match': rec['recommended_items'][3] if len(rec['recommended_items']) > 3 else None,
                'Top4_Score': rec['similarity_scores'][3] if len(rec['similarity_scores']) > 3 else None,
                'Top5_Match': rec['recommended_items'][4] if len(rec['recommended_items']) > 4 else None,
                'Top5_Score': rec['similarity_scores'][4] if len(rec['similarity_scores']) > 4 else None,
                'All_Matches': ', '.join(map(str, rec['recommended_items']))
            }
            data.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(data)
        df = df.sort_values('Item_ID')  # 按ID排序
        df.to_excel(output_path, index=False)
        
        print(f"Successfully saved {len(data)} recommendations to {output_path}")
        
        # 打印一些示例
        print("\nExample recommendations:")
        for i in range(min(5, len(data))):
            item_id = df.iloc[i]['Item_ID']
            matches = df.iloc[i]['All_Matches']
            print(f"Item {item_id} -> Top matches: [{matches}]")

def main():
    # 设置路径
    model_path = "VBPR/saved/IQON3000/VBPR/VBPR.pth.tar"
    features_path = "resnet152_features.pth"
    output_path = "outfit_recommendations.xlsx"
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure the trained VBPR model is available.")
        return
    
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Please make sure the resnet152_features.pth file is available.")
        return
    
    try:
        # 创建推荐生成器
        generator = RecommendationGenerator(model_path, features_path)
        
        # 生成推荐
        recommendations = generator.generate_top5_recommendations()
        
        # 保存结果
        generator.save_to_excel(recommendations, output_path)
        
        print(f"\n✅ Successfully generated recommendations for {len(recommendations)} items!")
        print(f"Results saved to: {output_path}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()