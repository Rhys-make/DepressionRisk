
# 基于社交媒体的抑郁风险预警系统

## 项目概述

本项目是一个基于机器学习的抑郁风险早期预警系统，通过分析社交媒体文本内容来识别潜在的抑郁风险。系统采用先进的自然语言处理技术和深度学习模型，能够从用户的社交媒体帖子中提取情感特征和抑郁相关指标。

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
│   ├── training/           # 训练模块
│   ├── evaluation/         # 评估模块
│   └── utils/              # 工具模块
├── train_models.py         # 模型训练主脚本
├── ensemble_predictor.py   # 集成预测器
└── README.md              # 项目说明
```

## 数据预处理模块

### 功能特性

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
- **数据分割**: 训练集和测试集的自动分割
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
train_data, test_data = processor.create_train_test_split(processed_data)

# 准备特征和标签
X_train, y_train = processor.prepare_features(train_data, label_column='label')
X_test, y_test = processor.prepare_features(test_data, label_column='label')
```

#### 单个文本处理

```python
from src.data_processing.preprocessor import TextPreprocessor

# 初始化文本预处理器
preprocessor = TextPreprocessor()

# 处理单个文本
text = "I feel so sad today :("
cleaned_text, features = preprocessor.process_text(text)

print(f"原始文本: {text}")
print(f"清洗后: {cleaned_text}")
print(f"特征: {features}")
```

### 特征说明

#### 语言学特征
- `text_length`: 文本总长度
- `word_count`: 单词数量
- `char_count`: 字符数量
- `avg_word_length`: 平均单词长度
- `sentence_count`: 句子数量
- `avg_sentence_length`: 平均句子长度
- `unique_words`: 唯一单词数量
- `lexical_diversity`: 词汇多样性
- `type_token_ratio`: 类型-标记比率

#### 抑郁相关特征
- `depression_情绪低落_count`: 情绪低落相关词汇数量
- `depression_兴趣丧失_count`: 兴趣丧失相关词汇数量
- `depression_睡眠问题_count`: 睡眠问题相关词汇数量
- `depression_食欲变化_count`: 食欲变化相关词汇数量
- `depression_注意力问题_count`: 注意力问题相关词汇数量
- `depression_自我评价_count`: 自我评价相关词汇数量
- `depression_自杀想法_count`: 自杀想法相关词汇数量
- `depression_焦虑症状_count`: 焦虑症状相关词汇数量
- `depression_身体症状_count`: 身体症状相关词汇数量
- `depression_社交退缩_count`: 社交退缩相关词汇数量
- `total_depression_words`: 总抑郁相关词汇数量
- `depression_word_density`: 抑郁词汇密度
- `depression_categories`: 抑郁词汇类别数

#### 情感特征
- `positive_emotion_count`: 积极情感词汇数量
- `negative_emotion_count`: 消极情感词汇数量
- `neutral_emotion_count`: 中性情感词汇数量
- `total_emotion_words`: 总情感词汇数量
- `emotion_word_density`: 情感词汇密度
- `emotion_polarity`: 情感极性

#### 社交媒体特征
- `hashtag_count`: 话题标签数量
- `mention_count`: @提及数量
- `url_count`: URL数量
- `emoticon_count`: 表情符号数量
- `repeated_char_count`: 重复字符数量
- `repeated_word_count`: 重复单词数量
- `exclamation_density`: 感叹号密度
- `question_density`: 问号密度

### 运行测试

```bash
# 生成训练数据
python simple_sample_generator.py

# 训练模型
python train_models.py

# 使用集成预测器（测试模型预测）
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

#### TextPreprocessor参数
- `text_column`: 文本列名（默认: 'text'）
- `label_column`: 标签列名（默认: 'label'）
- `min_text_length`: 最小文本长度（默认: 10）
- `max_text_length`: 最大文本长度（默认: 1000）

#### 文本清洗选项
- `remove_urls`: 是否移除URL（默认: True）
- `remove_emails`: 是否移除邮箱（默认: True）
- `remove_phones`: 是否移除电话号码（默认: True）
- `normalize_emoticons`: 是否标准化表情符号（默认: True）
- `remove_repeated_chars`: 是否移除重复字符（默认: True）
- `lowercase`: 是否转换为小写（默认: True）

## 下一步计划

1. **模型开发**: 实现基于BERT和胶囊网络的深度学习模型
2. **训练模块**: 创建模型训练和验证流程
3. **评估模块**: 实现模型性能评估和可视化
4. **Web界面**: 开发用户友好的Web界面
5. **API服务**: 提供RESTful API服务

## 贡献指南

欢迎提交Issue和Pull Request来改进项目！

## 许可证

本项目采用MIT许可证。
=======
DepressionRisk

