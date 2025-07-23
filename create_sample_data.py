import torch
import numpy as np
from Models.VBPR import VBPR
import os

def create_sample_model():
    """创建一个示例VBPR模型用于测试"""
    print("Creating sample VBPR model...")
    
    # 模型参数
    item_num = 1000  # 原始数据集的item数量
    hidden_dim = 512
    visual_feature_dim = 2048
    
    # 创建示例visual features
    visual_features = torch.randn(item_num + 1, visual_feature_dim)
    
    # 创建VBPR模型
    model = VBPR(item_num, hidden_dim, visual_feature_dim, visual_features, with_Nor=True)
    
    # 确保目录存在
    os.makedirs("VBPR/saved/IQON3000/VBPR", exist_ok=True)
    
    # 保存模型
    model_path = "VBPR/saved/IQON3000/VBPR/VBPR.pth.tar"
    torch.save(model, model_path)
    print(f"Sample model saved to: {model_path}")
    
    return model_path

def create_sample_features():
    """创建示例的resnet152特征文件"""
    print("Creating sample features...")
    
    # 创建201个items的特征 (ID: 0-200)
    features = {}
    feature_dim = 2048  # ResNet152特征维度
    
    for i in range(201):
        # 为每个item创建随机特征
        features[i] = torch.randn(feature_dim)
    
    # 保存特征
    features_path = "resnet152_features.pth"
    torch.save(features, features_path)
    print(f"Sample features saved to: {features_path}")
    print(f"Created features for items: {min(features.keys())}-{max(features.keys())}")
    
    return features_path

def main():
    """创建所有示例数据"""
    print("Creating sample data for testing...")
    
    # 创建示例模型
    model_path = create_sample_model()
    
    # 创建示例特征
    features_path = create_sample_features()
    
    print("\n✅ Sample data created successfully!")
    print(f"Model: {model_path}")
    print(f"Features: {features_path}")
    print("\nYou can now run: python generate_recommendations.py")

if __name__ == "__main__":
    main()