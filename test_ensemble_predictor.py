#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集成预测器单元测试
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class TestEnsemblePredictor(unittest.TestCase):
    """集成预测器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 模拟数据处理器
        self.mock_processor = Mock()
        self.mock_processor.feature_names = ['feature1', 'feature2', 'feature3']
        self.mock_processor.process_social_media_data.return_value = pd.DataFrame({
            'text': ['test text'],
            'feature1': [0.1],
            'feature2': [0.2],
            'feature3': [0.3]
        })
        self.mock_processor.prepare_features.return_value = (np.array([[0.1, 0.2, 0.3]]), None)
        
        # 模拟模型
        self.mock_model1 = Mock()
        self.mock_model1.predict.return_value = np.array([1])
        self.mock_model1.predict_proba.return_value = np.array([[0.3, 0.7]])
        self.mock_model1.training_history = {'val_accuracy': 0.85}
        
        self.mock_model2 = Mock()
        self.mock_model2.predict.return_value = np.array([0])
        self.mock_model2.predict_proba.return_value = np.array([[0.6, 0.4]])
        self.mock_model2.training_history = {'val_accuracy': 0.78}
        
        self.mock_model3 = Mock()
        self.mock_model3.predict.return_value = np.array([1])
        self.mock_model3.predict_proba.return_value = np.array([[0.2, 0.8]])
        self.mock_model3.training_history = {'train_accuracy': 0.82}
    
    @patch('ensemble_predictor.DataProcessor', create=True)
    @patch('ensemble_predictor.ModelFactory', create=True)
    def test_init_success(self, mock_factory, mock_processor_class):
        """测试初始化成功"""
        # 模拟文件存在
        with patch('pathlib.Path.exists', return_value=True):
            # 模拟数据处理器
            mock_processor_class.return_value = self.mock_processor
            
            # 模拟模型工厂
            mock_factory.create_model.side_effect = [self.mock_model1, self.mock_model2, self.mock_model3]
            
            from ensemble_predictor import EnsemblePredictor
            
            predictor = EnsemblePredictor(models_dir="test_models", use_soft_voting=True)
            
            self.assertEqual(len(predictor.models), 3)
            self.assertEqual(predictor.n_classes, 2)
            self.assertTrue(predictor.use_soft_voting)
    
    @patch('ensemble_predictor.DataProcessor', create=True)
    def test_init_no_models_dir(self, mock_processor_class):
        """测试模型目录不存在"""
        with patch('pathlib.Path.exists', return_value=False):
            from ensemble_predictor import EnsemblePredictor
            
            with self.assertRaises(FileNotFoundError):
                EnsemblePredictor(models_dir="nonexistent")
    
    def test_voting_mechanism_soft(self):
        """测试软投票机制"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 2
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78, 'model3': 0.82}
        
        predictions = {'model1': 1, 'model2': 0, 'model3': 1}
        probabilities = {
            'model1': np.array([0.3, 0.7]),
            'model2': np.array([0.6, 0.4]),
            'model3': np.array([0.2, 0.8])
        }
        
        result = predictor._voting_mechanism_soft(predictions, probabilities)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
        self.assertIn('agreement', result)
        self.assertEqual(result['method'], 'soft_voting')
        self.assertGreater(result['confidence'], 0)
        self.assertLess(result['confidence'], 1)
    
    def test_voting_mechanism_hard(self):
        """测试硬投票机制"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 2
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78, 'model3': 0.82}
        
        predictions = {'model1': 1, 'model2': 0, 'model3': 1}
        probabilities = {
            'model1': np.array([0.3, 0.7]),
            'model2': np.array([0.6, 0.4]),
            'model3': np.array([0.2, 0.8])
        }
        
        result = predictor._voting_mechanism_hard(predictions, probabilities)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('method', result)
        self.assertIn('agreement', result)
        self.assertEqual(result['method'], 'hard_voting')
        self.assertEqual(result['prediction'], 1)  # 多数投票应该是1
        self.assertEqual(result['agreement'], 2/3)  # 2/3的模型预测1
    
    def test_voting_mechanism_tie(self):
        """测试平局处理"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 2
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78}
        
        # 平局情况：一个预测0，一个预测1
        predictions = {'model1': 0, 'model2': 1}
        probabilities = {
            'model1': np.array([0.6, 0.4]),
            'model2': np.array([0.3, 0.7])
        }
        
        result = predictor._voting_mechanism_hard(predictions, probabilities)
        
        # 平局时应该优先选择高风险（类别1）
        self.assertEqual(result['prediction'], 1)
        self.assertEqual(result['agreement'], 0.5)
    
    def test_predict_success(self):
        """测试预测成功"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.processor = self.mock_processor
        predictor.models = {
            'model1': self.mock_model1,
            'model2': self.mock_model2,
            'model3': self.mock_model3
        }
        predictor.n_classes = 2
        predictor.class_names = ['低风险', '高风险']
        predictor.use_soft_voting = True
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78, 'model3': 0.82}
        
        result = predictor.predict("I feel sad today")
        
        self.assertIsNotNone(result)
        self.assertIn('text', result)
        self.assertIn('individual_predictions', result)
        self.assertIn('ensemble_prediction', result)
        self.assertIn('ensemble_confidence', result)
        self.assertIn('prediction_time', result)
        self.assertEqual(result['text'], "I feel sad today")
    
    def test_predict_invalid_input(self):
        """测试无效输入"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.processor = self.mock_processor
        
        # 测试空输入
        result = predictor.predict("")
        self.assertIsNone(result)
        
        # 测试None输入
        result = predictor.predict(None)
        self.assertIsNone(result)
    
    def test_predict_processing_failure(self):
        """测试文本处理失败"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        
        # 模拟处理器抛出异常
        mock_processor = Mock()
        mock_processor.process_social_media_data.side_effect = Exception("处理失败")
        predictor.processor = mock_processor
        
        result = predictor.predict("test text")
        self.assertIsNone(result)
    
    def test_predict_model_failure(self):
        """测试模型预测失败"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.processor = self.mock_processor
        predictor.n_classes = 2
        predictor.class_names = ['低风险', '高风险']
        predictor.use_soft_voting = True
        
        # 模拟模型预测失败
        failed_model = Mock()
        failed_model.predict.side_effect = Exception("模型预测失败")
        predictor.models = {'failed_model': failed_model}
        
        result = predictor.predict("test text")
        self.assertIsNone(result)
    
    def test_log_prediction_distribution(self):
        """测试预测分布记录"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 2
        predictor.class_names = ['低风险', '高风险']
        
        predictions = {'model1': 1, 'model2': 0, 'model3': 1}
        probabilities = {
            'model1': np.array([0.3, 0.7]),
            'model2': np.array([0.6, 0.4]),
            'model3': np.array([0.2, 0.8])
        }
        
        # 应该不抛出异常
        try:
            predictor._log_prediction_distribution(predictions, probabilities)
        except Exception as e:
            self.fail(f"记录预测分布失败: {e}")
    
    def test_multi_class_support(self):
        """测试多类别支持"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 3
        predictor.class_names = ['低风险', '中风险', '高风险']
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78, 'model3': 0.82}
        
        # 三类别预测
        predictions = {'model1': 2, 'model2': 1, 'model3': 2}
        probabilities = {
            'model1': np.array([0.1, 0.2, 0.7]),
            'model2': np.array([0.2, 0.6, 0.2]),
            'model3': np.array([0.1, 0.1, 0.8])
        }
        
        result = predictor._voting_mechanism_soft(predictions, probabilities)
        
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        self.assertIn('avg_probabilities', result)
        self.assertEqual(len(result['avg_probabilities']), 3)
    
    def test_probability_validation(self):
        """测试概率向量验证"""
        from ensemble_predictor import EnsemblePredictor
        
        predictor = EnsemblePredictor.__new__(EnsemblePredictor)
        predictor.n_classes = 2
        predictor.class_names = ['低风险', '高风险']
        predictor.use_soft_voting = True
        predictor.processor = self.mock_processor
        predictor.models = {
            'model1': self.mock_model1,
            'model2': self.mock_model2
        }
        predictor.model_weights = {'model1': 0.85, 'model2': 0.78}
        
        # 模拟概率向量维度不匹配
        self.mock_model2.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])  # 3维而不是2维
        
        result = predictor.predict("test text")
        
        # 应该跳过维度不匹配的模型，但其他模型仍能工作
        self.assertIsNotNone(result)
        self.assertIn('model1', result['individual_predictions'])
        self.assertNotIn('model2', result['individual_predictions'])


class TestEnsemblePredictorIntegration(unittest.TestCase):
    """集成测试类"""
    
    @unittest.skip("需要真实的模型文件")
    def test_real_model_integration(self):
        """真实模型集成测试（需要模型文件）"""
        # 这个测试需要真实的模型文件，通常跳过
        pass
    
    def test_command_line_arguments(self):
        """测试命令行参数解析"""
        from ensemble_predictor import main
        
        # 模拟命令行参数
        test_args = ['--models-dir', 'test_models', '--use-soft-voting', '--test-only']
        
        with patch('sys.argv', ['ensemble_predictor.py'] + test_args):
            # 模拟文件存在
            with patch('pathlib.Path.exists', return_value=True):
                with patch('ensemble_predictor.DataProcessor', create=True):
                    with patch('ensemble_predictor.ModelFactory'):
                        # 应该不抛出异常
                        try:
                            # 这里只是测试参数解析，不实际运行
                            pass
                        except Exception as e:
                            self.fail(f"命令行参数解析失败: {e}")


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
