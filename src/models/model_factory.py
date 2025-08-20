"""
模型工厂
用于统一创建和管理不同类型的模型
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from .base_model import BaseModel
from .bert_capsule_model import BertCapsuleWrapper
from .traditional_models import TraditionalModels, TraditionalModelWithTuning

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""
    
    # 支持的模型类型
    TRADITIONAL_MODELS = {
        'svm', 'random_forest', 'gradient_boosting', 'logistic_regression',
        'naive_bayes', 'knn', 'decision_tree'
    }
    
    DEEP_MODELS = {
        'bert_capsule'
    }
    
    @classmethod
    def create_model(cls, model_type: str, model_category: str = "traditional", 
                    **kwargs) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
            model_category: 模型类别 ('traditional', 'deep')
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        if model_category == "traditional":
            return cls._create_traditional_model(model_type, **kwargs)
        elif model_category == "deep":
            return cls._create_deep_model(model_type, **kwargs)
        else:
            raise ValueError(f"不支持的模型类别: {model_category}")
    
    @classmethod
    def _create_traditional_model(cls, model_type: str, **kwargs) -> TraditionalModels:
        """创建传统机器学习模型"""
        if model_type not in cls.TRADITIONAL_MODELS:
            raise ValueError(f"不支持的传统模型类型: {model_type}")
        
        # 检查是否需要超参数调优
        use_tuning = kwargs.pop('use_tuning', False)
        
        if use_tuning:
            return TraditionalModelWithTuning(model_type, **kwargs)
        else:
            return TraditionalModels(model_type, **kwargs)
    
    @classmethod
    def _create_deep_model(cls, model_type: str, **kwargs) -> BaseModel:
        """创建深度学习模型"""
        if model_type not in cls.DEEP_MODELS:
            raise ValueError(f"不支持的深度学习模型类型: {model_type}")
        
        if model_type == "bert_capsule":
            return BertCapsuleWrapper(**kwargs)
        else:
            raise ValueError(f"不支持的深度学习模型类型: {model_type}")
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """获取所有可用的模型类型"""
        return {
            'traditional': list(cls.TRADITIONAL_MODELS),
            'deep': list(cls.DEEP_MODELS)
        }
    
    @classmethod
    def get_model_info(cls, model_type: str, model_category: str = "traditional") -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = cls.create_model(model_type, model_category)
            return {
                'model_type': model_type,
                'model_category': model_category,
                'model_name': model.model_name,
                'is_trained': model.is_trained
            }
        except Exception as e:
            return {
                'model_type': model_type,
                'model_category': model_category,
                'error': str(e)
            }
    
    @classmethod
    def create_model_comparison(cls, model_configs: List[Dict[str, Any]]) -> List[BaseModel]:
        """
        创建多个模型用于比较
        
        Args:
            model_configs: 模型配置列表，每个配置包含:
                - model_type: 模型类型
                - model_category: 模型类别
                - **kwargs: 其他参数
                
        Returns:
            模型列表
        """
        models = []
        for config in model_configs:
            model_type = config.pop('model_type')
            model_category = config.pop('model_category', 'traditional')
            model = cls.create_model(model_type, model_category, **config)
            models.append(model)
        
        logger.info(f"创建了 {len(models)} 个模型用于比较")
        return models


class ModelManager:
    """模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.training_results = {}
    
    def add_model(self, name: str, model: BaseModel):
        """添加模型"""
        self.models[name] = model
        logger.info(f"添加模型: {name} ({model.model_name})")
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """获取模型"""
        return self.models.get(name)
    
    def remove_model(self, name: str):
        """移除模型"""
        if name in self.models:
            del self.models[name]
            logger.info(f"移除模型: {name}")
    
    def list_models(self) -> Dict[str, str]:
        """列出所有模型"""
        return {name: model.model_name for name, model in self.models.items()}
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: Optional[np.ndarray] = None,
                        y_val: Optional[np.ndarray] = None,
                        **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        训练所有模型
        
        Returns:
            训练结果字典
        """
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"训练模型: {name}")
            try:
                # 检查模型类型，决定训练方式
                if isinstance(model, BertCapsuleWrapper):
                    # 对于BERT模型，需要文本数据
                    if 'texts' not in kwargs:
                        raise ValueError("BERT模型需要文本数据，请提供'texts'参数")
                    texts = kwargs['texts']
                    val_texts = kwargs.get('val_texts')
                    history = model.train(texts, y_train, val_texts, y_val, **kwargs)
                else:
                    # 传统模型使用特征数据
                    history = model.train(X_train, y_train, X_val, y_val, **kwargs)
                
                results[name] = {
                    'success': True,
                    'history': history
                }
            except Exception as e:
                logger.error(f"训练模型 {name} 失败: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
        
        self.training_results = results
        return results
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray,
                          test_texts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        评估所有模型
        
        Returns:
            评估结果字典
        """
        results = {}
        
        for name, model in self.models.items():
            if not model.is_trained:
                logger.warning(f"模型 {name} 尚未训练，跳过评估")
                continue
            
            logger.info(f"评估模型: {name}")
            try:
                # 检查模型类型，决定评估方式
                if isinstance(model, BertCapsuleWrapper):
                    # 对于BERT模型，需要文本数据
                    if test_texts is None:
                        raise ValueError("BERT模型需要文本数据，请提供'test_texts'参数")
                    metrics = model.evaluate(test_texts, y_test)
                else:
                    # 传统模型使用特征数据
                    metrics = model.evaluate(X_test, y_test)
                
                results[name] = {
                    'success': True,
                    'metrics': metrics
                }
            except Exception as e:
                logger.error(f"评估模型 {name} 失败: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def get_best_model(self, metric: str = 'accuracy') -> Optional[tuple]:
        """
        获取最佳模型
        
        Args:
            metric: 评估指标
            
        Returns:
            (模型名称, 模型实例, 指标值) 或 None
        """
        best_score = -1
        best_model_info = None
        
        for name, model in self.models.items():
            if not model.is_trained:
                continue
            
            # 从训练历史中获取指标
            history = model.training_history
            if 'val_accuracy' in history and history['val_accuracy'] is not None:
                score = history['val_accuracy']
            elif 'train_accuracy' in history:
                score = history['train_accuracy']
            else:
                continue
            
            if score > best_score:
                best_score = score
                best_model_info = (name, model, score)
        
        return best_model_info
