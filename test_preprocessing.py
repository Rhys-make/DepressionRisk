#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理功能测试脚本
测试文本清洗、特征提取、数据标准化等功能
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.preprocessor import DataProcessor
from data_processing.text_cleaner import TextCleaner
from data_processing.feature_extractor import FeatureExtractor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_text_cleaner():
    """测试文本清洗功能"""
    print("🧹 测试文本清洗功能")
    print("=" * 50)
    
    cleaner = TextCleaner()
    
    # 测试用例
    test_texts = [
        "I am so happy today!!! 😊 #blessed @friend",
        "I feel so sad and hopeless... :( #depression",
        "RT @user: This is a retweet with http://example.com",
        "I can't sleep at night... my mind keeps racing...",
        "Had a great day with friends! Everything is wonderful! :D",
        "Sometimes I think about hurting myself...",
        "I'm feeling blessed and thankful for everything! 🙏",
        "I feel worthless and like a complete failure in life...",
        "Just finished a challenging project! Feeling accomplished!",
        "I want to disappear and never be seen again..."
    ]
    
    print("原始文本 -> 清洗后文本:")
    print("-" * 80)
    
    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean_text(text)
        print(f"{i:2d}. {text[:60]:<60} -> {cleaned[:60]:<60}")
    
    print()

def test_feature_extractor():
    """测试特征提取功能"""
    print("🔍 测试特征提取功能")
    print("=" * 50)
    
    extractor = FeatureExtractor()
    
    # 测试用例
    test_texts = [
        "I am so happy today! Everything feels wonderful!",
        "I am so sad today! I feel completely hopeless and worthless.",
        "I can't sleep at night. My mind keeps racing with suicidal thoughts.",
        "I love my life and everything in it!",
        "I think about death a lot and can't stop these suicidal thoughts."
    ]
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 测试文本 {i}: {text}")
        print("-" * 60)
        
        # 提取所有特征
        features = extractor.extract_all_features(text)
        
        # 显示关键特征
        key_categories = [
            ('语言学特征', ['text_length', 'word_count', 'avg_word_length']),
            ('抑郁特征', [f for f in features.keys() if 'depression' in f]),
            ('情感特征', [f for f in features.keys() if 'emotion' in f]),
            ('标点特征', ['exclamation_count', 'question_count', 'ellipsis_count'])
        ]
        
        for category, feature_list in key_categories:
            print(f"\n{category}:")
            for feature in feature_list[:5]:  # 只显示前5个
                if feature in features:
                    print(f"  {feature}: {features[feature]}")

def test_data_processor():
    """测试数据处理器功能"""
    print("\n⚙️  测试数据处理器功能")
    print("=" * 50)
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'text': [
            "I am so happy today! Everything feels wonderful!",
            "I am so sad today! I feel completely hopeless.",
            "I can't sleep at night. My mind keeps racing.",
            "I love my life and everything in it!",
            "I think about death a lot and can't stop these thoughts.",
            "I'm feeling blessed and thankful for everything!",
            "I feel worthless and like a complete failure.",
            "Just finished a challenging project! Feeling accomplished!",
            "I want to disappear and never be seen again.",
            "I'm excited about the future and all possibilities!"
        ],
        'label': [0, 1, 1, 0, 1, 0, 1, 0, 1, 0]  # 0=低风险, 1=高风险
    })
    
    print(f"📊 原始数据: {len(test_data)} 行")
    print(f"标签分布: {test_data['label'].value_counts().to_dict()}")
    
    # 初始化数据处理器
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=1,
        max_text_length=500
    )
    
    # 处理数据
    print("\n🔄 开始数据处理...")
    processed_data = processor.process_social_media_data(test_data)
    
    print(f"✅ 处理完成: {len(processed_data)} 行")
    print(f"📈 特征数量: {len(processor.feature_names)}")
    
    # 显示特征名称
    print(f"\n📋 特征列表 (前10个):")
    for i, feature in enumerate(processor.feature_names[:10]):
        print(f"  {i+1:2d}. {feature}")
    
    if len(processor.feature_names) > 10:
        print(f"  ... 还有 {len(processor.feature_names) - 10} 个特征")

def test_feature_preparation():
    """测试特征准备功能"""
    print("\n🎯 测试特征准备功能")
    print("=" * 50)
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'text': [
            "I am so happy today! Everything feels wonderful!",
            "I am so sad today! I feel completely hopeless.",
            "I can't sleep at night. My mind keeps racing.",
            "I love my life and everything in it!",
            "I think about death a lot and can't stop these thoughts."
        ],
        'label': [0, 1, 1, 0, 1]
    })
    
    # 初始化数据处理器
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=1,
        max_text_length=500
    )
    
    # 处理数据
    processed_data = processor.process_social_media_data(test_data)
    
    # 准备特征
    print("🔄 准备训练特征...")
    X_train, y_train = processor.prepare_features(processed_data, label_column='label', fit_scaler=True)
    
    print(f"✅ 特征准备完成:")
    print(f"  - 特征矩阵形状: {X_train.shape}")
    print(f"  - 标签向量形状: {y_train.shape}")
    print(f"  - 标签分布: {np.bincount(y_train)}")
    
    # 测试预测时的特征准备
    print("\n🔄 测试预测特征准备...")
    test_text = "I feel sad and hopeless today"
    test_df = pd.DataFrame({'text': [test_text]})
    test_processed = processor.process_social_media_data(test_df)
    X_test, _ = processor.prepare_features(test_processed, fit_scaler=False)
    
    print(f"✅ 预测特征准备完成:")
    print(f"  - 测试特征形状: {X_test.shape}")
    print(f"  - 特征维度匹配: {'✅' if X_test.shape[1] == X_train.shape[1] else '❌'}")

def test_data_persistence():
    """测试数据持久化功能"""
    print("\n💾 测试数据持久化功能")
    print("=" * 50)
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'text': [
            "I am so happy today! Everything feels wonderful!",
            "I am so sad today! I feel completely hopeless.",
            "I can't sleep at night. My mind keeps racing.",
            "I love my life and everything in it!",
            "I think about death a lot and can't stop these thoughts."
        ],
        'label': [0, 1, 1, 0, 1]
    })
    
    # 初始化数据处理器
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=1,
        max_text_length=500
    )
    
    # 处理数据
    processed_data = processor.process_social_media_data(test_data)
    X_train, y_train = processor.prepare_features(processed_data, label_column='label', fit_scaler=True)
    
    # 保存处理器
    save_path = Path("test_processor.pkl")
    print(f"💾 保存处理器到: {save_path}")
    processor.save_processor(save_path)
    
    # 加载处理器
    print(f"📂 加载处理器从: {save_path}")
    new_processor = DataProcessor()
    new_processor.load_processor(save_path)
    
    # 验证加载的处理器
    print("✅ 验证加载的处理器:")
    print(f"  - 特征名称数量: {len(new_processor.feature_names)}")
    print(f"  - 特征名称匹配: {'✅' if new_processor.feature_names == processor.feature_names else '❌'}")
    
    # 测试使用加载的处理器
    test_text = "I feel sad and hopeless today"
    test_df = pd.DataFrame({'text': [test_text]})
    test_processed = new_processor.process_social_media_data(test_df)
    X_test, _ = new_processor.prepare_features(test_processed, fit_scaler=False)
    
    print(f"  - 预测特征形状: {X_test.shape}")
    print(f"  - 特征维度匹配: {'✅' if X_test.shape[1] == X_train.shape[1] else '❌'}")
    
    # 清理测试文件
    if save_path.exists():
        save_path.unlink()
        print(f"🗑️  清理测试文件: {save_path}")

def test_error_handling():
    """测试错误处理功能"""
    print("\n⚠️  测试错误处理功能")
    print("=" * 50)
    
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=1,
        max_text_length=500
    )
    
    # 测试空文本
    print("📝 测试空文本处理:")
    empty_data = pd.DataFrame({
        'text': ['', '   ', 'I am happy', ''],
        'label': [0, 1, 0, 1]
    })
    
    try:
        processed = processor.process_social_media_data(empty_data)
        print(f"  ✅ 空文本处理成功: {len(processed)} 行")
    except Exception as e:
        print(f"  ❌ 空文本处理失败: {e}")
    
    # 测试缺失标签
    print("\n📝 测试缺失标签处理:")
    missing_label_data = pd.DataFrame({
        'text': ['I am happy', 'I am sad', 'I am okay'],
        'label': [0, np.nan, 1]
    })
    
    try:
        processed = processor.process_social_media_data(missing_label_data)
        print(f"  ✅ 缺失标签处理成功: {len(processed)} 行")
    except Exception as e:
        print(f"  ❌ 缺失标签处理失败: {e}")
    
    # 测试超长文本
    print("\n📝 测试超长文本处理:")
    long_text = "I am " + "very " * 1000 + "happy!"
    long_data = pd.DataFrame({
        'text': [long_text],
        'label': [0]
    })
    
    try:
        processed = processor.process_social_media_data(long_data)
        print(f"  ✅ 超长文本处理成功: {len(processed)} 行")
    except Exception as e:
        print(f"  ❌ 超长文本处理失败: {e}")

def main():
    """主函数"""
    print("🚀 数据预处理功能测试")
    print("=" * 60)
    
    try:
        # 1. 测试文本清洗
        test_text_cleaner()
        
        # 2. 测试特征提取
        test_feature_extractor()
        
        # 3. 测试数据处理器
        test_data_processor()
        
        # 4. 测试特征准备
        test_feature_preparation()
        
        # 5. 测试数据持久化
        test_data_persistence()
        
        # 6. 测试错误处理
        test_error_handling()
        
        print("\n🎉 所有测试完成！")
        print("✅ 数据预处理功能正常工作")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        logger.error(f"测试失败: {e}", exc_info=True)

if __name__ == "__main__":
    main()
