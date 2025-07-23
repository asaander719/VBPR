# 服装搭配推荐系统使用说明

## 概述

这个系统可以使用预训练的VBPR模型为新的服装数据集生成top5搭配推荐。系统支持处理ID为0-200的新服装数据，并为每件服装推荐最匹配的其他服装。

## 文件结构

```
workspace/
├── final_recommendation_generator.py    # 主要推荐生成脚本
├── generate_recommendations.py          # 原始版本脚本
├── create_sample_data.py               # 创建测试数据的脚本
├── outfit_recommendations.xlsx         # 生成的推荐结果
├── VBPR/saved/IQON3000/VBPR/VBPR.pth.tar  # 预训练模型
├── resnet152_features.pth              # ResNet152特征文件
└── Models/VBPR.py                      # VBPR模型定义
```

## 使用方法

### 1. 基本使用

如果您已经有了实际的模型和特征文件，可以直接运行：

```bash
# 激活虚拟环境
source venv/bin/activate

# 使用默认参数运行
python final_recommendation_generator.py
```

### 2. 自定义参数

```bash
python final_recommendation_generator.py \
    --model_path "你的模型路径/VBPR.pth.tar" \
    --features_path "你的特征文件路径/resnet152_features.pth" \
    --output_path "自定义输出文件名.xlsx" \
    --device "cpu" \
    --top_k 5
```

### 3. 参数说明

- `--model_path`: VBPR模型文件路径（默认: `VBPR/saved/IQON3000/VBPR/VBPR.pth.tar`）
- `--features_path`: ResNet152特征文件路径（默认: `resnet152_features.pth`）
- `--output_path`: 输出Excel文件路径（默认: `outfit_recommendations.xlsx`）
- `--device`: 计算设备，可选'cpu'或'cuda'（默认: 自动检测）
- `--top_k`: 每个物品推荐的数量（默认: 5）

## 输入文件格式

### 1. VBPR模型文件 (VBPR.pth.tar)
- 这是您已经训练好的VBPR模型
- 包含预训练的权重和模型架构
- 系统会自动提取其中可重用的权重

### 2. ResNet152特征文件 (resnet152_features.pth)
- 格式：Python字典，键为物品ID（0-200），值为特征向量
- 特征维度：2048（ResNet152的标准输出维度）
- 示例格式：
```python
{
    0: torch.tensor([0.1, 0.2, ...]),  # 2048维特征向量
    1: torch.tensor([0.3, 0.4, ...]),
    ...
    200: torch.tensor([0.5, 0.6, ...])
}
```

## 输出文件格式

生成的Excel文件包含以下列：

- `Item_ID`: 物品ID（0-200）
- `Top1_Match` - `Top5_Match`: 前5个最匹配的物品ID
- `Top1_Score` - `Top5_Score`: 对应的兼容性分数
- `All_Matches`: 所有推荐物品的汇总（逗号分隔）

### 示例输出

| Item_ID | Top1_Match | Top1_Score | Top2_Match | Top2_Score | ... | All_Matches |
|---------|------------|------------|------------|------------|-----|-------------|
| 0       | 71         | 0.8542     | 101        | 0.8234     | ... | 71, 101, 97, 35, 66 |
| 1       | 90         | 0.8765     | 169        | 0.8456     | ... | 90, 169, 178, 55, 88 |

## 工作原理

1. **加载数据**: 加载预训练的VBPR模型和新的ResNet152特征
2. **创建新embedding**: 为新的物品ID创建embedding层
3. **权重转移**: 从预训练模型转移可重用的权重（如视觉特征转换层）
4. **计算兼容性**: 计算所有物品对之间的兼容性分数
5. **生成推荐**: 为每个物品选择top-k个最兼容的其他物品
6. **保存结果**: 将结果保存为Excel文件

## 技术特点

- **内存优化**: 使用批处理计算避免内存溢出
- **GPU支持**: 自动检测并使用可用的GPU加速计算
- **错误处理**: 包含完善的错误处理和回退机制
- **进度显示**: 实时显示计算进度
- **结果验证**: 自动验证生成的推荐结果

## 性能说明

- **计算复杂度**: O(N²)，其中N是物品数量（201）
- **内存需求**: 约1-2GB（CPU模式）
- **运行时间**: 约1-5分钟（取决于硬件配置）

## 故障排除

### 常见问题

1. **模型加载失败**
   - 确保VBPR模型文件存在且完整
   - 检查PyTorch版本兼容性

2. **特征文件格式错误**
   - 确保特征文件是正确的Python字典格式
   - 验证特征维度是否为2048

3. **内存不足**
   - 减小batch_size参数
   - 使用CPU模式而不是GPU

4. **权限错误**
   - 确保有写入输出文件的权限
   - 检查输出目录是否存在

### 调试模式

在脚本中添加更多调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

系统支持以下扩展：

1. **自定义相似度计算**: 修改`_compute_simple_similarity`方法
2. **不同的推荐数量**: 通过`--top_k`参数调整
3. **批处理大小优化**: 修改`batch_size`参数
4. **输出格式定制**: 修改`save_to_excel`方法

## 联系支持

如果遇到问题或需要定制功能，请提供：
1. 错误日志
2. 输入文件的基本信息（大小、格式等）
3. 运行环境信息（Python版本、PyTorch版本等）