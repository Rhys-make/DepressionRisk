"""
基础模型类
定义所有模型的通用接口和基本功能
"""

import os
import pickle
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """基础模型抽象类"""
    
    def __init__(self, model_name: str = "base_model"):
        """
        初始化基础模型
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = {}
        self.feature_names = None
        
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            **kwargs: 其他参数
            
        Returns:
            训练历史字典
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入特征
            
        Returns:
            预测概率
        """
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            X_test: 测试特征
            y_test: 测试标签
            
        Returns:
            评估指标字典
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        # 预测
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_pred_proba[:, 1]) if len(np.unique(y_test)) > 1 else 0.0
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # 分类报告
        report = classification_report(y_test, y_pred, output_dict=True)
        metrics['classification_report'] = report
        
        logger.info(f"模型评估完成 - {self.model_name}")
        logger.info(f"准确率: {metrics['accuracy']:.4f}")
        logger.info(f"精确率: {metrics['precision']:.4f}")
        logger.info(f"召回率: {metrics['recall']:.4f}")
        logger.info(f"F1分数: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def save_model(self, file_path: Union[str, Path]) -> None:
        """
        保存模型
        
        Args:
            file_path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型对象
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_history': self.training_history,
            'feature_names': self.feature_names
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # 保存模型信息为JSON
        info_path = file_path.with_suffix('.json')
        info_data = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {file_path}")
    
    def load_model(self, file_path: Union[str, Path]) -> None:
        """
        加载模型
        
        Args:
            file_path: 模型文件路径
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {file_path}")
        
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.is_trained = model_data['is_trained']
        self.training_history = model_data.get('training_history', {})
        self.feature_names = model_data.get('feature_names')
        
        logger.info(f"模型已从 {file_path} 加载")
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        获取特征重要性
        
        Returns:
            特征重要性字典
        """
        if not self.is_trained or self.feature_names is None:
            return None
        
        # 子类需要实现具体的特征重要性计算
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        return {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history
        }
