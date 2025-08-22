#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT预测器
使用BERT模型进行抑郁风险预测
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
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

class BertPredictor:
    """BERT预测器"""
    
    def __init__(self, models_dir: str = "models"):
        """
        初始化BERT预测器
        
        Args:
            models_dir: 模型目录
        """
        from data_processing.preprocessor import DataProcessor
        from models.model_factory import ModelFactory
        
        self.models_dir = Path(models_dir)
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
            
        except Exception as e:
            logger.error(f"❌ 加载数据处理器失败: {e}")
            raise
        
        # 尝试加载BERT模型
        bert_model_names = ['simple_bert', 'bert_capsule', 'bert_capsule_advanced']
        
        for model_name in bert_model_names:
            try:
                # 尝试加载模型
                pkl_path = self.models_dir / f"{model_name}_model.pkl"
                if pkl_path.exists():
                    # 首先尝试加载元数据来获取训练时的参数
                    meta_path = self.models_dir / f"{model_name}_model.meta.json"
                    if meta_path.exists():
                        import json
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta_data = json.load(f)
                        
                        # 使用训练时的参数重新创建模型
                        model_params = meta_data.get('model_params', {})
                        bert_model_name = meta_data.get('bert_model_name', 'distilbert-base-uncased')
                        
                        model = ModelFactory.create_model(
                            model_type='bert_capsule',
                            bert_model_name=bert_model_name,
                            symptom_capsules=model_params.get('symptom_capsules', 4),
                            capsule_dim=model_params.get('capsule_dim', 8),
                            max_length=model_params.get('max_length', 64),
                            dropout=model_params.get('dropout', 0.1),
                            num_iterations=model_params.get('num_iterations', 3)
                        )
                    else:
                        # 如果没有元数据，使用默认的简化参数
                        model = ModelFactory.create_model(
                            model_type='bert_capsule',
                            bert_model_name='distilbert-base-uncased',
                            symptom_capsules=4,
                            capsule_dim=8,
                            max_length=64,
                            dropout=0.1,
                            num_iterations=3
                        )
                    
                    model.load_model(pkl_path)
                    self.model = model
                    
                    # 检测类别数
                    if self.n_classes is None:
                        self.n_classes = 2  # 默认二分类
                        self.class_names = ['低风险', '高风险']
                        logger.info(f"📊 检测到 {self.n_classes} 个类别: {self.class_names}")
                    
                    logger.info(f"✅ 加载BERT模型: {model_name}")
                    break
                    
            except Exception as e:
                logger.error(f"❌ 加载模型 {model_name} 失败: {e}")
                continue
        
        if self.model is None:
            logger.error("没有找到可用的训练BERT模型")
            logger.info("💡 请先运行以下命令训练模型:")
            logger.info("   python train_simple_bert.py  # 快速CPU训练")
            logger.info("   python train_models.py       # 完整训练（需要GPU）")
            raise Exception("没有找到可用的训练BERT模型")
        
        logger.info(f"🎯 BERT预测器初始化完成")
    
    def predict(self, text: str) -> Optional[Dict[str, Any]]:
        """
        BERT预测
        
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
            
            # 检查是否有特征
            if len(processed_data) == 0:
                logger.warning("没有提取到特征，请检查文本内容")
                return None
                
            # 获取处理后的文本
            processed_text = processed_data['text'].iloc[0]
            
            logger.debug(f"处理后的文本: {processed_text[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ 文本处理失败: {e}")
            logger.info(f"💡 请尝试输入更详细的文本（至少包含一些情感词汇）")
            return None
        
        # 进行预测
        try:
            pred = self.model.predict([processed_text])[0]
            prob = self.model.predict_proba([processed_text])[0]
            
            # 验证概率向量
            if len(prob) != self.n_classes:
                logger.warning(f"概率向量维度不匹配: {len(prob)} != {self.n_classes}")
                return None
            
            # 记录预测时间
            prediction_time = time.time() - start_time
            
            return {
                'text': text,
                'processed_text': processed_text,
                'prediction': pred,
                'confidence': float(prob[pred]),
                'probabilities': prob.tolist(),
                'prediction_time': prediction_time,
                'n_classes': self.n_classes,
                'class_names': self.class_names
            }
            
        except Exception as e:
            logger.error(f"BERT模型预测失败: {e}")
            return None
    
    def test_predictor(self):
        """测试BERT预测器"""
        logger.info("🧪 BERT预测器测试")
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
            
            # 显示预测结果
            predicted_class = self.class_names[result['prediction']] if result['prediction'] < len(self.class_names) else f"类别{result['prediction']}"
            logger.info(f"   预测: {predicted_class} ({result['confidence']:.1%})")
            logger.info(f"   预测时间: {result['prediction_time']:.3f}秒")
            
            # 显示概率分布
            logger.info("   概率分布:")
            for j, (class_name, prob) in enumerate(zip(self.class_names, result['probabilities'])):
                logger.info(f"     {class_name}: {prob:.1%}")
            
            # 判断是否正确
            expected_class = "高风险" if "高风险" in expected else "低风险"
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
            logger.info(f"\n📈 BERT模型准确率: {accuracy:.1%} ({correct_predictions}/{successful_predictions})")
            logger.info(f"❌ 失败预测: {failed_predictions}")
            
            if accuracy >= 0.8:
                logger.info("🎉 BERT模型表现优秀！")
            elif accuracy >= 0.7:
                logger.info("👍 BERT模型表现良好！")
            else:
                logger.info("⚠️ BERT模型需要改进")
        else:
            logger.error("❌ 所有预测都失败了")
    
    def interactive_predict(self):
        """交互式预测"""
        logger.info(f"\n🔍 BERT预测器交互模式")
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
                predicted_class = self.class_names[result['prediction']] if result['prediction'] < len(self.class_names) else f"类别{result['prediction']}"
                print(f"🎯 预测: {predicted_class} ({result['confidence']:.1%})")
                print(f"⏱️ 预测时间: {result['prediction_time']:.3f}秒")
                
                # 显示详细概率分布
                print(f"\n📈 类别概率分布:")
                for i, (class_name, prob) in enumerate(zip(self.class_names, result['probabilities'])):
                    print(f"  {class_name}: {prob:.1%}")
                
                # 显示处理后的文本
                print(f"\n📝 处理后的文本: {result['processed_text'][:100]}...")
                
            except KeyboardInterrupt:
                logger.info("\n👋 用户中断，退出交互模式")
                break
            except Exception as e:
                logger.error(f"交互预测异常: {e}")
                continue
        
        logger.info("👋 BERT预测完成！")

def main():
    """主函数"""
    try:
        # 支持命令行参数
        import argparse
        parser = argparse.ArgumentParser(description='BERT预测器')
        parser.add_argument('--models-dir', default='models', help='模型目录')
        parser.add_argument('--test-only', action='store_true', help='仅运行测试')
        parser.add_argument('--interactive-only', action='store_true', help='仅运行交互模式')
        
        args = parser.parse_args()
        
        logger.info(f"🚀 启动BERT预测器")
        logger.info(f"📁 模型目录: {args.models_dir}")
        
        predictor = BertPredictor(models_dir=args.models_dir)
        
        if args.test_only:
            # 仅测试
            predictor.test_predictor()
        elif args.interactive_only:
            # 仅交互
            predictor.interactive_predict()
        else:
            # 默认：先测试再交互
            predictor.test_predictor()
            predictor.interactive_predict()
        
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {e}")
        logger.info("💡 请先运行 train_models.py 训练并保存BERT模型")
    except Exception as e:
        logger.error(f"❌ BERT预测器失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
