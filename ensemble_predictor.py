#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成预测器
使用多个模型的投票机制进行预测
"""

import sys
import os
import pandas as pd
import numpy as np
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self):
        """初始化集成预测器"""
        from data_processing.preprocessor import DataProcessor
        from models.model_factory import ModelFactory
        
        # 加载数据处理器
        self.processor = DataProcessor()
        self.processor.load_processor("models/data_processor.pkl")
        
        # 加载所有可用模型
        self.models = {}
        model_names = ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting']
        
        for model_name in model_names:
            model_path = f"models/{model_name}_model.pkl"
            if os.path.exists(model_path):
                model = ModelFactory.create_model(model_name, 'traditional')
                model.load_model(model_path)
                self.models[model_name] = model
                print(f"✅ 加载模型: {model_name}")
        
        if not self.models:
            raise Exception("没有找到可用的训练模型")
        
        print(f"📊 共加载 {len(self.models)} 个模型")
    
    def predict(self, text):
        """集成预测"""
        try:
            # 预处理文本
            data = pd.DataFrame({'text': [text]})
            processed_data = self.processor.process_social_media_data(data)
            
            # 检查是否有特征
            if len(self.processor.feature_names) == 0:
                raise ValueError("没有提取到特征，请检查文本内容")
                
            X, _ = self.processor.prepare_features(processed_data, fit_scaler=False)
        except Exception as e:
            print(f"❌ 文本处理失败: {e}")
            print(f"💡 请尝试输入更详细的文本（至少包含一些情感词汇）")
            return None
        
        # 收集所有模型的预测结果
        predictions = {}
        probabilities = {}
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            predictions[name] = pred
            probabilities[name] = prob
        
        # 投票机制
        vote_result = self._voting_mechanism(predictions, probabilities)
        
        return {
            'text': text,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'ensemble_prediction': vote_result['prediction'],
            'ensemble_confidence': vote_result['confidence'],
            'voting_method': vote_result['method'],
            'agreement_level': vote_result['agreement']
        }
    
    def _voting_mechanism(self, predictions, probabilities):
        """投票机制"""
        # 1. 简单多数投票
        votes = list(predictions.values())
        vote_counts = Counter(votes)
        majority_prediction = vote_counts.most_common(1)[0][0]
        majority_count = vote_counts.most_common(1)[0][1]
        total_models = len(predictions)
        
        # 2. 计算一致性
        agreement = majority_count / total_models
        
        # 3. 计算平均置信度
        avg_confidence = np.mean([max(prob) for prob in probabilities.values()])
        
        # 4. 加权投票（基于置信度）
        weighted_votes = {0: 0, 1: 0}
        for name, pred in predictions.items():
            confidence = max(probabilities[name])
            weighted_votes[pred] += confidence
        
        weighted_prediction = max(weighted_votes, key=weighted_votes.get)
        weighted_confidence = weighted_votes[weighted_prediction] / sum(weighted_votes.values())
        
        # 5. 选择最终结果
        if agreement >= 0.75:  # 75%以上模型一致
            final_prediction = majority_prediction
            final_confidence = avg_confidence
            method = "多数投票"
        else:
            final_prediction = weighted_prediction
            final_confidence = weighted_confidence
            method = "加权投票"
        
        return {
            'prediction': final_prediction,
            'confidence': final_confidence,
            'method': method,
            'agreement': agreement
        }
    
    def test_ensemble(self):
        """测试集成预测器"""
        print("🧪 集成预测器测试")
        print("=" * 60)
        
        test_cases = [
            ("I feel so sad today!", "应该高风险"),
            ("I feel so sad and hopeless today", "应该高风险"),
            ("Had a great day with friends!", "应该低风险"),
            ("I can't sleep at night. My mind keeps racing", "应该高风险"),
            ("Sometimes I wonder if anyone would notice if I wasn't here", "应该高风险"),
            ("Just finished a challenging project! Feeling accomplished!", "应该低风险"),
            ("I feel worthless and like a failure", "应该高风险"),
            ("I'm excited about the new opportunities!", "应该低风险"),
            ("I think about death a lot", "应该高风险"),
            ("I feel blessed to have wonderful people in my life!", "应该低风险")
        ]
        
        correct_predictions = 0
        total_predictions = len(test_cases)
        
        for i, (text, expected) in enumerate(test_cases, 1):
            print(f"\n{i}. 文本: {text[:50]}...")
            print(f"   期望: {expected}")
            
            result = self.predict(text)
            
            if result is None:
                print("   ❌ 预测失败")
                continue
                
            # 显示各模型预测
            print("   各模型预测:")
            for name, pred in result['individual_predictions'].items():
                prob = result['individual_probabilities'][name]
                risk = "高风险" if pred == 1 else "低风险"
                conf = max(prob)
                print(f"     {name}: {risk} ({conf:.1%})")
            
            # 显示集成结果
            ensemble_risk = "高风险" if result['ensemble_prediction'] == 1 else "低风险"
            print(f"   集成预测: {ensemble_risk} ({result['ensemble_confidence']:.1%})")
            print(f"   投票方法: {result['voting_method']} (一致性: {result['agreement_level']:.1%})")
            
            # 判断是否正确
            is_correct = (expected == "应该高风险" and result['ensemble_prediction'] == 1) or \
                        (expected == "应该低风险" and result['ensemble_prediction'] == 0)
            
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"   结果: {status}")
        
        # 计算准确率
        accuracy = correct_predictions / total_predictions
        print(f"\n📈 集成模型准确率: {accuracy:.1%} ({correct_predictions}/{total_predictions})")
        
        if accuracy >= 0.8:
            print("🎉 集成模型表现优秀！")
        elif accuracy >= 0.7:
            print("👍 集成模型表现良好！")
        else:
            print("⚠️ 集成模型需要改进")
    
    def interactive_predict(self):
        """交互式预测"""
        print(f"\n🔍 集成预测器交互模式")
        print("输入文本进行预测，输入 'quit' 退出")
        print("-" * 40)
        
        while True:
            user_text = input("\n请输入要预测的文本: ").strip()
            if user_text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_text:
                continue
            
            result = self.predict(user_text)
            
            if result is None:
                continue
                
            print(f"\n📊 预测结果:")
            print("各模型预测:")
            for name, pred in result['individual_predictions'].items():
                prob = result['individual_probabilities'][name]
                risk = "高风险" if pred == 1 else "低风险"
                conf = max(prob)
                print(f"  {name}: {risk} ({conf:.1%})")
            
            ensemble_risk = "高风险" if result['ensemble_prediction'] == 1 else "低风险"
            print(f"\n🎯 最终预测: {ensemble_risk} ({result['ensemble_confidence']:.1%})")
            print(f"📋 投票方法: {result['voting_method']}")
            print(f"🤝 模型一致性: {result['agreement_level']:.1%}")
        
        print("👋 集成预测完成！")

def main():
    """主函数"""
    try:
        predictor = EnsemblePredictor()
        
        # 测试集成预测器
        predictor.test_ensemble()
        
        # 交互式预测
        predictor.interactive_predict()
        
    except Exception as e:
        print(f"❌ 集成预测器失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
