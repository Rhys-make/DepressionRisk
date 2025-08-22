"""
模型工厂
用于统一创建和管理BERT模型
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np

from .base_model import BaseModel
from .bert_capsule_model import BertCapsuleWrapper

logger = logging.getLogger(__name__)


class ModelFactory:
    """模型工厂类"""
    
    # 支持的模型类型
    DEEP_MODELS = {
        'bert_capsule'
    }
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> BaseModel:
        """
        创建模型实例
        
        Args:
            model_type: 模型类型
            **kwargs: 模型参数
            
        Returns:
            模型实例
        """
        return cls._create_deep_model(model_type, **kwargs)
    
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
            'deep': list(cls.DEEP_MODELS)
        }
    
    @classmethod
    def get_model_info(cls, model_type: str) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            model = cls.create_model(model_type)
            return {
                'model_type': model_type,
                'model_category': 'deep',
                'model_name': model.model_name,
                'is_trained': model.is_trained
            }
        except Exception as e:
            return {
                'model_type': model_type,
                'model_category': 'deep',
                'error': str(e)
            }
    
    @classmethod
    def create_model_comparison(cls, model_configs: List[Dict[str, Any]]) -> List[BaseModel]:
        """
        创建多个模型用于比较
        
        Args:
            model_configs: 模型配置列表，每个配置包含:
                - model_type: 模型类型
                - **kwargs: 其他参数
                
        Returns:
            模型列表
        """
        models = []
        for config in model_configs:
            model_type = config.pop('model_type')
            model = cls.create_model(model_type, **config)
            models.append(model)
        
        logger.info(f"创建了 {len(models)} 个BERT模型用于比较")
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
    
    def train_all_models(self, texts: List[str], y_train: np.ndarray,
                        val_texts: Optional[List[str]] = None,
                        y_val: Optional[np.ndarray] = None,
                        **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        训练所有BERT模型
        
        Returns:
            训练结果字典
        """
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"训练模型: {name}")
            try:
                history = model.train(texts, y_train, val_texts, y_val, **kwargs)
                
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
    
    def evaluate_all_models(self, test_texts: List[str], y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        评估所有BERT模型
        
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
                metrics = model.evaluate(test_texts, y_test)
                
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
