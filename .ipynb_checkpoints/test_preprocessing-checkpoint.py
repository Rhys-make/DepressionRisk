#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理测试脚本
测试文本清洗和特征提取功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_text_cleaner():
    """测试文本清洗功能"""
    print("🧪 测试文本清洗功能...")
    
    try:
        from data_processing.text_cleaner import TextCleaner
        
        # 创建文本清洗器
        cleaner = TextCleaner()
        
        # 测试文本
        test_texts = [
            "I feel so sad and hopeless today. Nothing seems to matter anymore. :(",
            "Had a great day with friends! Everything is wonderful! :D",
            "I can't sleep at night. My mind keeps racing with negative thoughts.",
            "Just finished a fantastic workout! Feeling energized and happy!",
            "I feel worthless and like a failure. Maybe everyone would be better off without me."
        ]
        
        print("原始文本:")
        for i, text in enumerate(test_texts, 1):
            print(f"{i}. {text}")
        
        print("\n清洗后的文本:")
        for i, text in enumerate(test_texts, 1):
            cleaned = cleaner.clean_text(text)
            print(f"{i}. {cleaned}")
        
        print("✅ 文本清洗功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 文本清洗功能测试失败: {e}")
        return False

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n🧪 测试特征提取功能...")
    
    try:
        from data_processing.feature_extractor import FeatureExtractor
        
        # 创建特征提取器
        extractor = FeatureExtractor()
        
        # 测试文本
        test_text = "I feel so sad and hopeless today. Nothing seems to matter anymore. :("
        
        # 提取特征
        features = extractor.extract_all_features(test_text)
        
        print("提取的特征:")
        for feature_name, value in features.items():
            print(f"  {feature_name}: {value}")
        
        print("✅ 特征提取功能测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 特征提取功能测试失败: {e}")
        return False

def test_data_processing():
    """测试数据处理功能"""
    print("\n🧪 测试数据处理功能...")
    
    try:
        from data_processing.preprocessor import DataProcessor
        
        # 创建数据处理器
        processor = DataProcessor()
        
        # 检查示例数据文件是否存在
        sample_file = "data/raw/sample_data.csv"
        if os.path.exists(sample_file):
            print(f"找到示例数据文件: {sample_file}")
            
            # 加载数据
            data = processor.load_data(sample_file)
            print(f"加载了 {len(data)} 条数据")
            
            # 处理数据
            processed_data = processor.process_social_media_data(data)
            print(f"处理了 {len(processed_data)} 条数据")
            
            print("✅ 数据处理功能测试通过")
            return True
        else:
            print(f"⚠️ 示例数据文件不存在: {sample_file}")
            return False
            
    except Exception as e:
        print(f"❌ 数据处理功能测试失败: {e}")
        return False

def test_imports():
    """测试核心包导入"""
    print("🧪 测试核心包导入...")
    
    try:
        import pandas as pd
        import numpy as np
        import torch
        import transformers
        import nltk
        import sklearn
        
        print("✅ 核心包导入成功")
        print(f"  - pandas: {pd.__version__}")
        print(f"  - numpy: {np.__version__}")
        print(f"  - torch: {torch.__version__}")
        print(f"  - transformers: {transformers.__version__}")
        print(f"  - scikit-learn: {sklearn.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 核心包导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 基于社交媒体的抑郁风险预警系统 - 数据预处理测试")
    print("=" * 60)
    
    # 测试核心包导入
    if not test_imports():
        return
    
    # 测试文本清洗
    if not test_text_cleaner():
        return
    
    # 测试特征提取
    if not test_feature_extraction():
        return
    
    # 测试数据处理
    if not test_data_processing():
        return
    
    print("\n" + "=" * 60)
    print("🎉 所有测试通过！数据预处理模块工作正常")
    print("\n📋 下一步:")
    print("1. 运行 'python preprocess_data.py' 处理完整示例数据")
    print("2. 开始开发你的模型！")

if __name__ == "__main__":
    main()
