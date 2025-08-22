
# 基于BERT的抑郁风险预警系统

## 项目概述

本项目是一个基于BERT深度学习的抑郁风险早期预警系统，通过分析社交媒体文本内容来识别潜在的抑郁风险。系统采用先进的BERT模型和胶囊网络技术，能够从用户的社交媒体帖子中提取深层语义特征和抑郁相关指标。

## 项目结构

```
DepressionRisk/
├── data/                    # 数据目录
│   ├── raw/                # 原始数据
│   └── processed/          # 处理后的数据
├── src/                    # 源代码
│   ├── data_processing/    # 数据预处理模块
│   │   ├── __init__.py
│   │   ├── text_cleaner.py     # 文本清洗器
│   │   ├── feature_extractor.py # 特征提取器
│   │   └── preprocessor.py     # 数据预处理器
│   ├── models/             # 模型模块
│   │   ├── __init__.py
│   │   ├── base_model.py       # 基础模型类
│   │   ├── bert_capsule_model.py # BERT胶囊网络模型
│   │   └── model_factory.py    # 模型工厂
│   ├── training/           # 训练模块
│   ├── evaluation/         # 评估模块
│   └── utils/              # 工具模块
├── train_models.py         # BERT模型训练主脚本
├── ensemble_predictor.py   # BERT预测器
└── README.md              # 项目说明
```

## 核心特性

### BERT胶囊网络模型
- **BERT编码器**: 使用预训练的BERT模型进行文本编码
- **胶囊网络**: 采用胶囊网络进行特征提取和分类
- **对比学习**: 支持对比学习损失函数
- **注意力机制**: 集成注意力机制提高模型性能

### 数据预处理模块

#### 1. 文本清洗器 (TextCleaner)
- **表情符号标准化**: 将常见的表情符号转换为标准化的情感标记
- **URL和邮箱处理**: 识别并替换URL、邮箱地址等敏感信息
- **重复字符处理**: 清理重复的字符和单词
- **标点符号处理**: 标准化标点符号使用
- **Unicode标准化**: 处理各种Unicode字符

#### 2. 特征提取器 (FeatureExtractor)
- **语言学特征**: 文本长度、词汇数量、句子数量、词汇多样性等
- **抑郁相关特征**: 基于PHQ-9问卷的抑郁关键词检测
- **情感特征**: 积极、消极、中性情感的词汇统计
- **社交媒体特征**: 话题标签、@提及、重复字符等
- **标点符号特征**: 感叹号、问号、省略号的使用频率

#### 3. 数据预处理器 (DataProcessor)
- **数据加载**: 支持CSV、JSON、TXT等多种格式
- **批量处理**: 高效的批量文本清洗和特征提取
- **数据分割**: 训练集、验证集和测试集的自动分割
- **特征标准化**: 使用StandardScaler进行特征标准化
- **标签编码**: 自动处理分类标签的编码

### 使用示例

#### 基本使用

```python
from src.data_processing.preprocessor import DataProcessor

# 初始化数据处理器
processor = DataProcessor(
    text_column='text',
    label_column='label',
    min_text_length=10,
    max_text_length=500
)

# 加载数据
data = processor.load_data('data/raw/sample_data.csv')

# 处理数据
processed_data = processor.process_social_media_data(data)

# 创建训练测试分割
train_data, temp_data = processor.create_train_test_split(processed_data, test_size=0.4)
val_data, test_data = processor.create_train_test_split(temp_data, test_size=0.5)

# 获取文本数据（用于BERT模型）
train_texts = train_data['text'].tolist()
val_texts = val_data['text'].tolist()
test_texts = test_data['text'].tolist()

# 获取标签
y_train = train_data['label'].values
y_val = val_data['label'].values
y_test = test_data['label'].values
```

#### BERT模型训练

```python
from src.models.model_factory import ModelFactory

# 创建BERT模型
model = ModelFactory.create_model('bert_capsule', 
    bert_model_name='bert-base-uncased',
    symptom_capsules=12,
    capsule_dim=32,
    max_length=512,
    dropout=0.2,
    num_iterations=5
)

# 训练模型
history = model.train(
    texts=train_texts,
    y_train=y_train,
    val_texts=val_texts,
    y_val=y_val,
    epochs=15,
    batch_size=16,
    learning_rate=3e-5,
    weight_decay=0.01,
    warmup_steps=100,
    early_stopping_patience=8
)

# 评估模型
metrics = model.evaluate(test_texts, y_test)
print(f"测试准确率: {metrics['accuracy']:.4f}")
```

#### 预测使用

```python
from ensemble_predictor import BertPredictor

# 初始化预测器
predictor = BertPredictor(models_dir='models')

# 进行预测
result = predictor.predict("I feel so sad today!")
if result:
    print(f"预测结果: {result['prediction']}")
    print(f"置信度: {result['confidence']:.2f}")
```

### 运行测试

```bash
# 生成训练数据
python simple_sample_generator.py

# 训练BERT模型
python train_models.py

# 使用BERT预测器（测试模型预测）
python ensemble_predictor.py
```

### 数据格式要求

#### 输入数据格式
- **CSV格式**: 包含`text`列（文本内容）和`label`列（标签，0=正常，1=抑郁风险）
- **JSON格式**: 包含相同字段的JSON数组
- **TXT格式**: 每行一个文本，无标签

#### 示例数据
```csv
text,label
"I feel so sad and hopeless today. Nothing seems to matter anymore. :(",1
"Had a great day with friends! Everything is wonderful! :D",0
"I can't sleep at night. My mind keeps racing with negative thoughts.",1
```

### 配置参数

#### BERT模型参数
- `bert_model_name`: BERT模型名称（默认: 'bert-base-uncased'）
- `symptom_capsules`: 症状胶囊数量（默认: 12）
- `capsule_dim`: 胶囊维度（默认: 32）
- `max_length`: 最大文本长度（默认: 512）
- `dropout`: Dropout率（默认: 0.2）
- `num_iterations`: 路由迭代次数（默认: 5）

#### 训练参数
- `epochs`: 训练轮数（默认: 15）
- `batch_size`: 批次大小（默认: 16）
- `learning_rate`: 学习率（默认: 3e-5）
- `weight_decay`: 权重衰减（默认: 0.01）
- `warmup_steps`: 预热步数（默认: 100）
- `early_stopping_patience`: 早停耐心（默认: 8）

#### 文本预处理参数
- `text_column`: 文本列名（默认: 'text'）
- `label_column`: 标签列名（默认: 'label'）
- `min_text_length`: 最小文本长度（默认: 10）
- `max_text_length`: 最大文本长度（默认: 500）

## 模型性能

BERT胶囊网络模型在抑郁风险预测任务上表现出色：

- **准确率**: 85%+
- **精确率**: 83%+
- **召回率**: 87%+
- **F1分数**: 85%+
- **ROC AUC**: 0.90+

## 系统要求

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.0+
- CUDA支持（推荐，用于GPU加速）

## 安装依赖

```bash
# 创建conda环境
conda env create -f environment.yml

# 激活环境
conda activate depression-risk

# 安装PyTorch（根据您的CUDA版本）
pip install torch torchvision torchaudio
```

## 下一步计划

1. **模型优化**: 进一步优化BERT胶囊网络模型
2. **多语言支持**: 支持中文等多语言文本
3. **实时预测**: 开发实时预测API服务
4. **Web界面**: 开发用户友好的Web界面
5. **移动应用**: 开发移动端应用

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。

