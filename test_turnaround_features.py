#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试情感转折特征提取效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.feature_extractor import FeatureExtractor

def test_turnaround_features():
    """测试情感转折特征提取"""
    
    extractor = FeatureExtractor()
    
    # 测试文本
    test_texts = [
        # 你提到的例子
        "Work was stressful af 😩 but ice cream fixed it lol 🍦😋",
        
        # 其他情感转折例子
        "今天心情很糟糕，但是和朋友聊天后好多了 😊",
        "I was so sad today, but my dog cheered me up! 🐕❤️",
        "最近压力很大，不过运动后感觉轻松多了 💪",
        
        # 对比：纯负面文本
        "I feel so sad and hopeless today. Nothing seems to matter anymore. :(",
        "今天心情很差，什么都不想做，感觉很绝望",
        
        # 对比：纯正面文本
        "Had a great day with friends! Everything is wonderful! :D",
        "今天很开心，和朋友一起玩得很尽兴！",
        
        # 复杂情感
        "今天心情很复杂，既有开心也有烦恼 😅",
        "I'm feeling mixed emotions today - happy and sad at the same time"
    ]
    
    print("🔍 情感转折特征提取测试")
    print("=" * 80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n📝 测试文本 {i}: {text}")
        
        # 提取所有特征
        features = extractor.extract_all_features(text)
        
        # 显示关键特征
        print("📊 关键特征:")
        print(f"  整体情感倾向: {features.get('overall_sentiment', 0):.3f}")
        print(f"  转折词数量: {features.get('turnaround_word_count', 0)}")
        print(f"  转折模式匹配: {features.get('turnaround_pattern_count', 0)}")
        print(f"  情感转折方向: {features.get('sentiment_turnaround', 0)}")
        print(f"  负面词汇密度: {features.get('negative_density', 0):.3f}")
        print(f"  正面词汇密度: {features.get('positive_density', 0):.3f}")
        print(f"  转折强度: {features.get('turnaround_intensity', 0):.3f}")
        
        # 预测判断
        if features.get('overall_sentiment', 0) > 0.1:
            prediction = "正常情绪"
        elif features.get('overall_sentiment', 0) < -0.1:
            prediction = "抑郁风险"
        else:
            prediction = "中性"
        
        print(f"  🎯 预测结果: {prediction}")
        
        # 转折分析
        if features.get('turnaround_word_count', 0) > 0:
            print(f"  🔄 检测到情感转折: 是")
            if features.get('sentiment_turnaround', 0) > 0:
                print(f"  📈 转折方向: 负面 → 正面")
            elif features.get('sentiment_turnaround', 0) < 0:
                print(f"  📉 转折方向: 正面 → 负面")
            else:
                print(f"  ➡️ 转折方向: 无明显变化")
        else:
            print(f"  🔄 检测到情感转折: 否")

def test_specific_example():
    """专门测试你提到的例子"""
    
    extractor = FeatureExtractor()
    target_text = "Work was stressful af 😩 but ice cream fixed it lol 🍦😋"
    
    print("\n🎯 专门测试你的例子")
    print("=" * 80)
    print(f"📝 文本: {target_text}")
    
    # 提取特征
    features = extractor.extract_all_features(target_text)
    
    print("\n📊 详细特征分析:")
    print(f"  转折词数量: {features.get('turnaround_word_count', 0)}")
    print(f"  转折模式匹配: {features.get('turnaround_pattern_count', 0)}")
    print(f"  转折强度: {features.get('turnaround_intensity', 0):.3f}")
    print(f"  情感转折方向: {features.get('sentiment_turnaround', 0)}")
    print(f"  整体情感倾向: {features.get('overall_sentiment', 0):.3f}")
    
    # 分析转折词
    text_lower = target_text.lower()
    turnaround_words_found = []
    for word in extractor.turnaround_words:
        if word in text_lower:
            turnaround_words_found.append(word)
    
    print(f"\n🔄 检测到的转折词: {turnaround_words_found}")
    
    # 分析情感词汇
    negative_words_found = []
    positive_words_found = []
    
    for word in extractor.emotion_words['negative']:
        if word in text_lower:
            negative_words_found.append(word)
    
    for word in extractor.emotion_words['positive']:
        if word in text_lower:
            positive_words_found.append(word)
    
    print(f"😔 负面词汇: {negative_words_found}")
    print(f"😊 正面词汇: {positive_words_found}")
    
    # 最终判断
    sentiment_score = features.get('overall_sentiment', 0)
    if sentiment_score > 0.1:
        final_prediction = "✅ 正常情绪（能自我调节）"
    elif sentiment_score < -0.1:
        final_prediction = "❌ 抑郁风险"
    else:
        final_prediction = "⚠️ 中性（需要更多信息）"
    
    print(f"\n🎯 最终判断: {final_prediction}")
    print(f"💡 判断依据: 整体情感倾向 {sentiment_score:.3f}")

def main():
    """主函数"""
    print("🚀 开始测试情感转折特征提取...")
    
    # 测试所有例子
    test_turnaround_features()
    
    # 专门测试你的例子
    test_specific_example()
    
    print("\n🎉 测试完成！")
    print("💡 建议:")
    print("  1. 如果效果满意，可以重新训练模型")
    print("  2. 运行: python train_models.py")
    print("  3. 测试: python ensemble_predictor.py --interactive-only")

if __name__ == "__main__":
    main()
