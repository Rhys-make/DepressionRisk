"""
传统机器学习模型
作为BERT-Capsule模型的对比基线
"""

import logging
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class TraditionalModels(BaseModel):
    """传统机器学习模型类"""
    
    def __init__(self, model_type: str = "random_forest", **kwargs):
        """
        初始化传统机器学习模型
        
        Args:
            model_type: 模型类型
            **kwargs: 模型参数
        """
        super().__init__(model_name=f"traditional_{model_type}")
        self.model_type = model_type
        self.model_params = kwargs
        self._initialize_model()
    
    def _initialize_model(self):
        """初始化模型"""
        if self.model_type == "svm":
            self.model = SVC(probability=True, random_state=42, **self.model_params)
        elif self.model_type == "random_forest":
            self.model = RandomForestClassifier(random_state=42, **self.model_params)
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(random_state=42, **self.model_params)
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, **self.model_params)
        elif self.model_type == "naive_bayes":
            self.model = MultinomialNB(**self.model_params)
        elif self.model_type == "knn":
            self.model = KNeighborsClassifier(**self.model_params)
        elif self.model_type == "decision_tree":
            self.model = DecisionTreeClassifier(random_state=42, **self.model_params)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        logger.info(f"初始化 {self.model_type} 模型")
    
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
        logger.info(f"开始训练 {self.model_type} 模型")
        logger.info(f"训练数据形状: {X_train.shape}")
        
        # 训练模型
        self.model.fit(X_train, y_train)
        
        # 计算训练集性能
        train_pred = self.model.predict(X_train)
        train_accuracy = np.mean(train_pred == y_train)
        
        # 计算验证集性能（如果有）
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_accuracy = np.mean(val_pred == y_val)
            logger.info(f"验证集准确率: {val_accuracy:.4f}")
        
        self.is_trained = True
        self.training_history = {
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_info': {
                'feature_count': X_train.shape[1],
                'sample_count': X_train.shape[0],
                'class_distribution': {
                    'class_0': np.sum(y_train == 0),
                    'class_1': np.sum(y_train == 1)
                }
            }
        }
        
        
        logger.info(f"模型训练完成 - 训练集准确率: {train_accuracy:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """获取特征重要性"""
        if not self.is_trained or self.feature_names is None:
            return None
        
        # 检查模型是否支持特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        elif hasattr(self.model, 'coef_'):
            # 对于线性模型，使用系数的绝对值
            importance_dict = dict(zip(self.feature_names, np.abs(self.model.coef_[0])))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            logger.warning(f"{self.model_type} 不支持特征重要性")
            return None


class TraditionalModelWithTuning(TraditionalModels):
    """带超参数调优的传统机器学习模型"""
    
    def __init__(self, model_type: str = "random_forest", 
                 param_grid: Optional[Dict[str, List]] = None,
                 cv: int = 5, **kwargs):
        """
        初始化带调优的模型
        
        Args:
            model_type: 模型类型
            param_grid: 参数网格
            cv: 交叉验证折数
            **kwargs: 其他参数
        """
        super().__init__(model_type, **kwargs)
        self.param_grid = param_grid or self._get_default_param_grid()
        self.cv = cv
    
    def _get_default_param_grid(self) -> Dict[str, List]:
        """获取默认参数网格"""
        if self.model_type == "svm":
            return {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        elif self.model_type == "random_forest":
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        elif self.model_type == "gradient_boosting":
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif self.model_type == "logistic_regression":
            return {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        else:
            return {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型（带超参数调优）
        """
        logger.info(f"开始训练 {self.model_type} 模型（带超参数调优）")
        
        # 使用网格搜索进行超参数调优
        grid_search = GridSearchCV(
            self.model, self.param_grid, cv=self.cv, 
            scoring='accuracy', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 更新最佳模型
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        # 记录训练历史
        self.training_history = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'model_type': self.model_type,
            'param_grid': self.param_grid
        }
        
        logger.info(f"最佳参数: {grid_search.best_params_}")
        logger.info(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
        
        return self.training_history
