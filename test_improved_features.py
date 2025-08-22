#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试改进后的特征提取功能
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.feature_extractor import FeatureExtractor

def test_depression_features():
    """测试抑郁特征提取"""
    extractor = FeatureExtractor()
    
    # 测试文本
    test_texts = [
        "everyone thinks i'm fine lol but inside i'm drowning 😵‍💫 #fakesmile",
        "I feel so sad and hopeless today",
        "Had a great day with friends!",
        "I can't sleep at night. My mind keeps racing",
        "I feel worthless and like a failure"
    ]
    
    print("🧪 测试改进后的抑郁特征提取")
    print("=" * 60)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{i}. 文本: {text}")
        
        # 提取抑郁特征
        features = extractor.extract_depression_features(text)
        
        # 显示关键特征
        print("   抑郁特征:")
        for key, value in features.items():
            if value > 0:  # 只显示非零特征
                print(f"     {key}: {value}")
        
        # 计算抑郁风险分数
        total_depression = features.get('total_depression_words', 0)
        depression_density = features.get('depression_word_density', 0)
        depression_categories = features.get('depression_categories', 0)
        
        print(f"   总抑郁词汇: {total_depression}")
        print(f"   抑郁词汇密度: {depression_density:.3f}")
        print(f"   抑郁类别数: {depression_categories}")
        
        # 简单的风险评估
        if total_depression >= 3 or depression_density >= 0.1:
            risk = "高风险"
        elif total_depression >= 1 or depression_density >= 0.05:
            risk = "中风险"
        else:
            risk = "低风险"
        
        print(f"   风险评估: {risk}")

if __name__ == "__main__":
    test_depression_features()
