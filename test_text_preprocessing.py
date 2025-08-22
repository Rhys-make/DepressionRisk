#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试文本预处理功能
"""

import sys
import os
import pandas as pd
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_text_preprocessing():
    """测试文本预处理"""
    try:
        from data_processing.preprocessor import DataProcessor
        
        print("🧪 测试文本预处理功能")
        print("=" * 50)
        
        # 创建数据处理器
        processor = DataProcessor()
        
        # 测试数据
        test_texts = [
            "Tired af all the time… maybe just work stress idk 😵‍💫",
            "I feel so sad and hopeless today :(",
            "Had a great day with friends! :)",
            "I can't sleep at night. My mind keeps racing...",
            "Just finished a challenging project! Feeling accomplished! 🎉"
        ]
        
        # 创建测试数据框
        test_data = pd.DataFrame({'text': test_texts})
        
        print("📝 原始文本:")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        print("\n🔄 开始预处理...")
        
        # 处理数据
        processed_data = processor.process_social_media_data(test_data)
        
        print("\n✨ 处理后的文本:")
        for i, (original, processed) in enumerate(zip(test_texts, processed_data['text']), 1):
            print(f"  {i}. 原始: {original}")
            print(f"     处理后: {processed}")
            print(f"     是否相同: {'❌ 相同' if original == processed else '✅ 不同'}")
            print()
        
        # 检查特征
        if 'feature_names' in dir(processor) and processor.feature_names:
            print(f"📊 提取的特征数量: {len(processor.feature_names)}")
            print("特征列名:")
            for i, feature in enumerate(processor.feature_names[:10], 1):  # 只显示前10个
                print(f"  {i}. {feature}")
            if len(processor.feature_names) > 10:
                print(f"  ... 还有 {len(processor.feature_names) - 10} 个特征")
        else:
            print("❌ 没有提取到特征")
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_text_preprocessing()
    
    if success:
        print("\n🎉 文本预处理测试完成！")
    else:
        print("\n❌ 文本预处理测试失败！")

if __name__ == "__main__":
    main()
