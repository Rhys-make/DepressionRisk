#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成预测器
使用多个模型的投票机制进行预测
支持任意类别数的多分类问题
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class EnsemblePredictor:
    """集成预测器"""
    
    def __init__(self, models_dir: str = "models", use_soft_voting: bool = True):
        """
        初始化集成预测器
        
        Args:
            models_dir: 模型目录
            use_soft_voting: 是否使用软投票（推荐）
        """
        from data_processing.preprocessor import DataProcessor
        from models.model_factory import ModelFactory
        
        self.models_dir = Path(models_dir)
        self.use_soft_voting = use_soft_voting
        self.n_classes = None  # 动态检测类别数
        self.class_names = None  # 类别名称映射
        
        # 健壮性检查
        if not self.models_dir.exists():
            raise FileNotFoundError(f"模型目录不存在: {self.models_dir}")
        
        # 加载数据处理器
        try:
            self.processor = DataProcessor()
            processor_path = self.models_dir / "data_processor.pkl"
            if not processor_path.exists():
                raise FileNotFoundError(f"数据处理器文件不存在: {processor_path}")
            
            self.processor.load_processor(processor_path)
            logger.info("✅ 数据处理器加载成功")
            
            # 健壮性检查：确保特征名存在
            if not self.processor.feature_names:
                logger.warning("⚠️ 数据处理器特征名为空，可能影响预测")
                
        except Exception as e:
            logger.error(f"❌ 加载数据处理器失败: {e}")
            raise
        
        # 加载所有可用模型
        self.models = {}
        self.model_weights = {}  # 模型权重（基于历史性能）
        model_names = ['random_forest', 'svm', 'logistic_regression', 'gradient_boosting']
        
        for model_name in model_names:
            model_path = self.models_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                try:
                    model = ModelFactory.create_model(model_name, 'traditional')
                    model.load_model(model_path)
                    self.models[model_name] = model
                    
                    # 检测类别数（使用第一个模型）
                    if self.n_classes is None:
                        self.n_classes = len(model.label_encoder.classes_) if hasattr(model, 'label_encoder') else 2
                        self.class_names = list(model.label_encoder.classes_) if hasattr(model, 'label_encoder') else ['低风险', '高风险']
                        logger.info(f"📊 检测到 {self.n_classes} 个类别: {self.class_names}")
                    
                    # 设置模型权重（基于训练历史）
                    if hasattr(model, 'training_history') and model.training_history:
                        history = model.training_history
                        if 'val_accuracy' in history and history['val_accuracy'] is not None:
                            self.model_weights[model_name] = history['val_accuracy']
                        elif 'train_accuracy' in history:
                            self.model_weights[model_name] = history['train_accuracy']
                        else:
                            self.model_weights[model_name] = 1.0
                    else:
                        self.model_weights[model_name] = 1.0
                    
                    logger.info(f"✅ 加载模型: {model_name} (权重: {self.model_weights[model_name]:.3f})")
                    
                except Exception as e:
                    logger.error(f"❌ 加载模型 {model_name} 失败: {e}")
                    continue
        
        if not self.models:
            raise Exception("没有找到可用的训练模型")
        
        logger.info(f"📊 共加载 {len(self.models)} 个模型")
        logger.info(f"🎯 使用{'软投票' if use_soft_voting else '硬投票'}策略")
    
    def predict(self, text: str) -> Optional[Dict[str, Any]]:
        """
        集成预测
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典，失败时返回None
        """
        start_time = time.time()
        
        try:
            # 输入验证
            if not text or not isinstance(text, str):
                logger.error("输入文本无效")
                return None
            
            # 预处理文本
            data = pd.DataFrame({'text': [text]})
            processed_data = self.processor.process_social_media_data(data)
            
            # 健壮性检查：确保特征名存在
            if not self.processor.feature_names:
                logger.error("数据处理器特征名为空")
                return None
            
            # 检查是否有特征
            if len(self.processor.feature_names) == 0:
                logger.warning("没有提取到特征，请检查文本内容")
                return None
                
            X, _ = self.processor.prepare_features(processed_data, fit_scaler=False)
            
            # 健壮性检查：验证特征维度
            expected_features = len(self.processor.feature_names)
            if X.shape[1] != expected_features:
                logger.error(f"特征维度不匹配: 期望 {expected_features}, 实际 {X.shape[1]}")
                return None
                
            logger.debug(f"特征维度: {X.shape}")
            
        except Exception as e:
            logger.error(f"❌ 文本处理失败: {e}")
            logger.info(f"💡 请尝试输入更详细的文本（至少包含一些情感词汇）")
            return None
        
        # 收集所有模型的预测结果
        predictions = {}
        probabilities = {}
        failed_models = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0]
                
                # 验证概率向量
                if len(prob) != self.n_classes:
                    logger.warning(f"模型 {name} 概率向量维度不匹配: {len(prob)} != {self.n_classes}")
                    continue
                
                predictions[name] = pred
                probabilities[name] = prob
                
            except Exception as e:
                logger.error(f"模型 {name} 预测失败: {e}")
                failed_models.append(name)
                continue
        
        if not predictions:
            logger.error("所有模型预测都失败了")
            return None
        
        if failed_models:
            logger.warning(f"以下模型预测失败: {failed_models}")
        
        # 投票机制
        if self.use_soft_voting:
            vote_result = self._voting_mechanism_soft(predictions, probabilities)
        else:
            vote_result = self._voting_mechanism_hard(predictions, probabilities)
        
        # 记录预测时间
        prediction_time = time.time() - start_time
        
        # 记录预测分布（用于drift检测）
        self._log_prediction_distribution(predictions, probabilities)
        
        return {
            'text': text,
            'individual_predictions': predictions,
            'individual_probabilities': probabilities,
            'ensemble_prediction': vote_result['prediction'],
            'ensemble_confidence': vote_result['confidence'],
            'voting_method': vote_result['method'],
            'agreement_level': vote_result['agreement'],
            'prediction_time': prediction_time,
            'n_classes': self.n_classes,
            'class_names': self.class_names
        }
    
    def _voting_mechanism_soft(self, predictions: Dict[str, int], probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        软投票机制（推荐）
        平均所有模型的概率向量，然后取argmax
        
        Args:
            predictions: 各模型预测结果
            probabilities: 各模型概率向量
            
        Returns:
            投票结果
        """
        try:
            # 确保所有概率向量维度一致
            prob_list = list(probabilities.values())
            if not prob_list:
                raise ValueError("没有有效的概率向量")
            
            # 验证所有概率向量维度一致
            first_dim = len(prob_list[0])
            for i, prob in enumerate(prob_list):
                if len(prob) != first_dim:
                    logger.warning(f"概率向量维度不一致: {len(prob)} != {first_dim}")
                    continue
            
            # 堆叠概率向量
            probs = np.vstack([p for p in prob_list if len(p) == first_dim])
            
            # 加权平均（基于模型权重）
            weights = np.array([self.model_weights[name] for name in predictions.keys() 
                              if name in probabilities and len(probabilities[name]) == first_dim])
            
            if len(weights) != len(probs):
                logger.warning("权重数量与概率向量数量不匹配，使用等权重")
                weights = np.ones(len(probs))
            
            # 归一化权重
            weights = weights / weights.sum()
            
            # 加权平均
            avg_prob = np.average(probs, axis=0, weights=weights)
            
            # 最终预测
            final_prediction = int(np.argmax(avg_prob))
            confidence = float(avg_prob[final_prediction])
            
            # 计算一致性（预测相同类别的模型比例）
            votes = [int(p == final_prediction) for p in predictions.values()]
            agreement = sum(votes) / len(votes) if votes else 0.0
            
            return {
                'prediction': final_prediction,
                'confidence': confidence,
                'method': 'soft_voting',
                'agreement': agreement,
                'avg_probabilities': avg_prob.tolist()
            }
            
        except Exception as e:
            logger.error(f"软投票失败: {e}")
            # 回退到硬投票
            return self._voting_mechanism_hard(predictions, probabilities)
    
    def _voting_mechanism_hard(self, predictions: Dict[str, int], probabilities: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        硬投票机制（备选）
        基于预测结果的多数投票
        
        Args:
            predictions: 各模型预测结果
            probabilities: 各模型概率向量
            
        Returns:
            投票结果
        """
        try:
            # 1. 简单多数投票
            votes = list(predictions.values())
            vote_counts = Counter(votes)
            majority_prediction = vote_counts.most_common(1)[0][0]
            majority_count = vote_counts.most_common(1)[0][1]
            total_models = len(predictions)
            
            # 2. 计算一致性
            agreement = majority_count / total_models
            
            # 3. 处理平局
            if agreement == 0.5 and len(vote_counts) == 2:
                # 平局时，优先选择高风险（类别1）
                majority_prediction = 1
                logger.info("检测到平局，优先选择高风险类别")
            
            # 4. 计算加权置信度
            weighted_confidence = 0.0
            total_weight = 0.0
            
            for name, pred in predictions.items():
                if pred == majority_prediction:
                    weight = self.model_weights.get(name, 1.0)
                    prob = probabilities[name]
                    confidence = prob[pred] if pred < len(prob) else 0.5
                    weighted_confidence += weight * confidence
                    total_weight += weight
            
            final_confidence = weighted_confidence / total_weight if total_weight > 0 else agreement
            
            return {
                'prediction': majority_prediction,
                'confidence': final_confidence,
                'method': 'hard_voting',
                'agreement': agreement
            }
            
        except Exception as e:
            logger.error(f"硬投票失败: {e}")
            # 最后的回退方案
            return {
                'prediction': 0,
                'confidence': 0.5,
                'method': 'fallback',
                'agreement': 0.0
            }
    
    def _log_prediction_distribution(self, predictions: Dict[str, int], probabilities: Dict[str, np.ndarray]) -> None:
        """
        记录预测分布（用于drift检测）
        
        Args:
            predictions: 各模型预测结果
            probabilities: 各模型概率向量
        """
        try:
            # 统计预测分布
            pred_counts = Counter(predictions.values())
            total_models = len(predictions)
            
            # 计算平均概率分布
            avg_probs = np.zeros(self.n_classes)
            for prob in probabilities.values():
                if len(prob) == self.n_classes:
                    avg_probs += prob
            avg_probs /= len(probabilities)
            
            # 记录分布信息
            distribution_info = {
                'timestamp': time.time(),
                'prediction_counts': dict(pred_counts),
                'total_models': total_models,
                'average_probabilities': avg_probs.tolist(),
                'class_names': self.class_names
            }
            
            logger.debug(f"预测分布: {distribution_info}")
            
        except Exception as e:
            logger.warning(f"记录预测分布失败: {e}")
    
    def test_ensemble(self):
        """测试集成预测器"""
        logger.info("🧪 集成预测器测试")
        logger.info("=" * 60)
        
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
        failed_predictions = 0
        
        for i, (text, expected) in enumerate(test_cases, 1):
            logger.info(f"\n{i}. 文本: {text[:50]}...")
            logger.info(f"   期望: {expected}")
            
            result = self.predict(text)
            
            if result is None:
                logger.error("   ❌ 预测失败")
                failed_predictions += 1
                continue
                
            # 显示各模型预测
            logger.info("   各模型预测:")
            for name, pred in result['individual_predictions'].items():
                prob = result['individual_probabilities'][name]
                class_name = self.class_names[pred] if pred < len(self.class_names) else f"类别{pred}"
                conf = prob[pred] if pred < len(prob) else 0.5
                logger.info(f"     {name}: {class_name} ({conf:.1%})")
            
            # 显示集成结果
            ensemble_class = self.class_names[result['ensemble_prediction']] if result['ensemble_prediction'] < len(self.class_names) else f"类别{result['ensemble_prediction']}"
            logger.info(f"   集成预测: {ensemble_class} ({result['ensemble_confidence']:.1%})")
            logger.info(f"   投票方法: {result['voting_method']} (一致性: {result['agreement_level']:.1%})")
            logger.info(f"   预测时间: {result['prediction_time']:.3f}秒")
            
            # 判断是否正确（支持多类别）
            expected_class = "高风险" if "高风险" in expected else "低风险"
            predicted_class = self.class_names[result['ensemble_prediction']] if result['ensemble_prediction'] < len(self.class_names) else f"类别{result['ensemble_prediction']}"
            
            is_correct = expected_class in predicted_class or predicted_class in expected_class
            
            if is_correct:
                correct_predictions += 1
                status = "✅"
            else:
                status = "❌"
            
            logger.info(f"   结果: {status}")
        
        # 计算准确率
        successful_predictions = total_predictions - failed_predictions
        if successful_predictions > 0:
            accuracy = correct_predictions / successful_predictions
            logger.info(f"\n📈 集成模型准确率: {accuracy:.1%} ({correct_predictions}/{successful_predictions})")
            logger.info(f"❌ 失败预测: {failed_predictions}")
            
            if accuracy >= 0.8:
                logger.info("🎉 集成模型表现优秀！")
            elif accuracy >= 0.7:
                logger.info("👍 集成模型表现良好！")
            else:
                logger.info("⚠️ 集成模型需要改进")
        else:
            logger.error("❌ 所有预测都失败了")
    
    def interactive_predict(self):
        """交互式预测"""
        logger.info(f"\n🔍 集成预测器交互模式")
        logger.info("输入文本进行预测，输入 'quit' 退出")
        logger.info("-" * 40)
        
        while True:
            try:
                user_text = input("\n请输入要预测的文本: ").strip()
                if user_text.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not user_text:
                    continue
                
                result = self.predict(user_text)
                
                if result is None:
                    logger.warning("预测失败，请重试")
                    continue
                    
                print(f"\n📊 预测结果:")
                print("各模型预测:")
                for name, pred in result['individual_predictions'].items():
                    prob = result['individual_probabilities'][name]
                    class_name = self.class_names[pred] if pred < len(self.class_names) else f"类别{pred}"
                    conf = prob[pred] if pred < len(prob) else 0.5
                    print(f"  {name}: {class_name} ({conf:.1%})")
                
                ensemble_class = self.class_names[result['ensemble_prediction']] if result['ensemble_prediction'] < len(self.class_names) else f"类别{result['ensemble_prediction']}"
                print(f"\n🎯 最终预测: {ensemble_class} ({result['ensemble_confidence']:.1%})")
                print(f"📋 投票方法: {result['voting_method']}")
                print(f"🤝 模型一致性: {result['agreement_level']:.1%}")
                print(f"⏱️ 预测时间: {result['prediction_time']:.3f}秒")
                
                # 显示详细概率分布（如果使用软投票）
                if result['voting_method'] == 'soft_voting' and 'avg_probabilities' in result:
                    print(f"\n📈 类别概率分布:")
                    for i, (class_name, prob) in enumerate(zip(self.class_names, result['avg_probabilities'])):
                        print(f"  {class_name}: {prob:.1%}")
                
            except KeyboardInterrupt:
                logger.info("\n👋 用户中断，退出交互模式")
                break
            except Exception as e:
                logger.error(f"交互预测异常: {e}")
                continue
        
        logger.info("👋 集成预测完成！")

def main():
    """主函数"""
    try:
        # 支持命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='集成预测器')
        parser.add_argument('--models-dir', default='models', help='模型目录')
        parser.add_argument('--use-soft-voting', action='store_true', default=True, help='使用软投票（推荐）')
        parser.add_argument('--use-hard-voting', action='store_true', help='使用硬投票')
        parser.add_argument('--test-only', action='store_true', help='仅运行测试')
        parser.add_argument('--interactive-only', action='store_true', help='仅运行交互模式')
        
        args = parser.parse_args()
        
        # 确定投票策略
        use_soft_voting = args.use_soft_voting and not args.use_hard_voting
        
        logger.info(f"🚀 启动集成预测器")
        logger.info(f"📁 模型目录: {args.models_dir}")
        logger.info(f"🎯 投票策略: {'软投票' if use_soft_voting else '硬投票'}")
        
        predictor = EnsemblePredictor(
            models_dir=args.models_dir,
            use_soft_voting=use_soft_voting
        )
        
        if args.test_only:
            # 仅测试
            predictor.test_ensemble()
        elif args.interactive_only:
            # 仅交互
            predictor.interactive_predict()
        else:
            # 默认：先测试再交互
            predictor.test_ensemble()
            predictor.interactive_predict()
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {e}")
        logger.info("💡 请先运行 train_models.py 训练并保存模型")
    except Exception as e:
        logger.error(f"❌ 集成预测器失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
