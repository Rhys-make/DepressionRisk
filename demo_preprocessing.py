#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理模块演示脚本
演示如何使用数据预处理模块处理社交媒体文本
"""

import pandas as pd
import numpy as np
from src.data_processing import TextCleaner, FeatureExtractor, DataProcessor

def demo_text_cleaner():
    """演示文本清洗器"""
    print("🧹 文本清洗器演示")
    print("=" * 50)
    
    # 创建文本清洗器
    cleaner = TextCleaner()
    
    # 示例文本（模拟社交媒体文本）
    sample_texts = [
        "今天心情真的很差:( 感觉什么都不想做... #depression #sad",
        "I'm feeling so down today :( nothing seems to matter anymore...",
        "今天很开心！:) 和朋友一起出去玩，感觉超棒的！",
        "Feeling great today! :D Had an amazing time with friends!",
        "为什么我总是这么没用... 感觉自己是个失败者 T_T",
        "Why am I always so useless... I feel like such a failure T_T"
    ]
    
    print("原始文本:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n清洗后的文本:")
    for i, text in enumerate(sample_texts, 1):
        cleaned = cleaner.clean_text(text)
        print(f"{i}. {cleaned}")
    
    print("\n文本特征:")
    for i, text in enumerate(sample_texts, 1):
        features = cleaner.extract_features(text)
        print(f"{i}. 长度: {features['text_length']}, 单词数: {features['word_count']}, 表情符号: {features['emoticon_count']}")

def demo_feature_extractor():
    """演示特征提取器"""
    print("\n🔍 特征提取器演示")
    print("=" * 50)
    
    # 创建特征提取器
    extractor = FeatureExtractor()
    
    # 示例文本
    sample_texts = [
        "I feel so sad and depressed today. Nothing makes me happy anymore.",
        "I'm feeling great! Life is wonderful and I'm so excited about everything!",
        "Sometimes I think about ending it all. I'm so tired of this pain.",
        "I love spending time with friends and family. Everything is perfect!"
    ]
    
    print("示例文本:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    
    print("\n提取的特征:")
    for i, text in enumerate(sample_texts, 1):
        print(f"\n文本 {i}:")
        features = extractor.extract_all_features(text)
        
        # 显示关键特征
        key_features = {
            '语言学特征': ['text_length', 'word_count', 'lexical_diversity'],
            '抑郁特征': ['total_depression_count', 'depression_density'],
            '情感特征': ['emotion_polarity', 'emotion_intensity'],
            '社交媒体特征': ['hashtag_count', 'mention_count']
        }
        
        for category, feature_names in key_features.items():
            print(f"  {category}:")
            for name in feature_names:
                if name in features:
                    print(f"    {name}: {features[name]:.4f}")

def demo_data_processor():
    """演示数据处理器"""
    print("\n🎛️ 数据处理器演示")
    print("=" * 50)
    
    # 创建示例数据
    sample_data = pd.DataFrame({
        'text': [
            "I feel so sad and depressed today. Nothing makes me happy anymore.",
            "I'm feeling great! Life is wonderful and I'm so excited about everything!",
            "Sometimes I think about ending it all. I'm so tired of this pain.",
            "I love spending time with friends and family. Everything is perfect!",
            "I feel worthless and like a complete failure. What's the point?",
            "Today was amazing! I accomplished so much and feel proud of myself!"
        ],
        'label': [1, 0, 1, 0, 1, 0]  # 1=抑郁风险, 0=正常
    })
    
    print("示例数据:")
    print(sample_data)
    
    # 创建数据处理器
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=10,
        max_text_length=500
    )
    
    # 处理数据
    print("\n处理步骤:")
    print("1. 处理社交媒体数据...")
    processed_data = processor.process_social_media_data(sample_data)
    print(f"   处理后数据量: {len(processed_data)} 行")
    
    print("2. 准备特征...")
    features, labels = processor.prepare_features(processed_data, label_column='label')
    print(f"   特征维度: {features.shape}")
    print(f"   标签数量: {len(labels)}")
    
    print("3. 分割数据...")
    train_data, test_data = processor.create_train_test_split(processed_data, test_size=0.3, stratify='label')
    print(f"   训练集: {len(train_data)} 样本")
    print(f"   测试集: {len(test_data)} 样本")
    
    print("4. 特征统计:")
    if features is not None:
        print(f"   特征数量: {features.shape[1]}")
        print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   特征均值: {features.mean():.4f}")
        print(f"   特征标准差: {features.std():.4f}")
    
    print("5. 标签分布:")
    if labels is not None:
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"   标签 {label}: {count} 样本 ({count/len(labels)*100:.1f}%)")
    else:
        print("   标签列未找到")

def main():
    """主函数"""
    print("📊 数据预处理模块完整演示")
    print("=" * 60)
    
    try:
        # 演示各个组件
        demo_text_cleaner()
        demo_feature_extractor()
        demo_data_processor()
        
        print("\n✅ 演示完成！")
        print("\n📝 总结:")
        print("- 文本清洗器: 清理和标准化社交媒体文本")
        print("- 特征提取器: 提取48个多维特征")
        print("- 数据处理器: 整合处理流程，准备训练数据")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        print("请确保已安装所有必要的依赖包")

if __name__ == "__main__":
    main()
