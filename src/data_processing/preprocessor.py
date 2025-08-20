#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理器
整合文本清洗和特征提取功能
"""

import pandas as pd
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder

from .text_cleaner import TextCleaner
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

class TextPreprocessor:
    """文本预处理器"""
    
    def __init__(self, 
                 text_column: str = 'text',
                 label_column: str = 'label',
                 min_text_length: int = 10,
                 max_text_length: int = 1000):
        """
        初始化文本预处理器
        
        Args:
            text_column: 文本列名
            label_column: 标签列名
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
        """
        self.text_column = text_column
        self.label_column = label_column
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        # 初始化组件
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        
        # 保存处理参数
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
    
    def clean_text(self, text: str) -> str:
        """
        清洗单个文本
        
        Args:
            text: 输入文本
            
        Returns:
            清洗后的文本
        """
        return self.text_cleaner.clean_text(text)
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        批量清洗文本
        
        Args:
            texts: 文本列表
            
        Returns:
            清洗后的文本列表
        """
        return self.text_cleaner.clean_batch(texts)
    
    def extract_features(self, text: str) -> Dict[str, float]:
        """
        提取单个文本的特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        return self.feature_extractor.extract_all_features(text)
    
    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            特征字典列表
        """
        return self.feature_extractor.extract_batch_features(texts)
    
    def process_text(self, text: str) -> Tuple[str, Dict[str, float]]:
        """
        处理单个文本：清洗 + 特征提取
        
        Args:
            text: 输入文本
            
        Returns:
            (清洗后的文本, 特征字典)
        """
        cleaned_text = self.clean_text(text)
        features = self.extract_features(cleaned_text)
        return cleaned_text, features
    
    def process_batch(self, texts: List[str]) -> Tuple[List[str], List[Dict[str, float]]]:
        """
        批量处理文本
        
        Args:
            texts: 文本列表
            
        Returns:
            (清洗后的文本列表, 特征字典列表)
        """
        cleaned_texts = self.clean_batch(texts)
        features = self.extract_batch_features(cleaned_texts)
        return cleaned_texts, features

class DataProcessor:
    """数据处理器"""
    
    def __init__(self, 
                 text_column: str = 'text',
                 label_column: str = 'label',
                 min_text_length: int = 10,
                 max_text_length: int = 1000):
        """
        初始化数据处理器
        
        Args:
            text_column: 文本列名
            label_column: 标签列名
            min_text_length: 最小文本长度
            max_text_length: 最大文本长度
        """
        self.text_column = text_column
        self.label_column = label_column
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        
        self.preprocessor = TextPreprocessor(
            text_column=text_column,
            label_column=label_column,
            min_text_length=min_text_length,
            max_text_length=max_text_length
        )
        
        # 保存处理参数
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        self.feature_columns = None  # 新增：可控制使用的特征列
    
    def set_feature_columns(self, cols: List[str]):
        """
        设置要使用的特征列
        
        Args:
            cols: 特征列名列表
        """
        self.feature_columns = cols
        logger.info(f"设置特征列: {len(cols)} 个特征")
    
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            数据框
        """
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() == '.json':
            data = pd.read_json(file_path)
        elif file_path.suffix.lower() == '.txt':
            # 假设是每行一个文本的格式，去掉换行符
            with open(file_path, 'r', encoding='utf-8') as f:
                texts = [line.rstrip('\n').strip() for line in f]
            data = pd.DataFrame({self.text_column: texts})
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"加载数据: {len(data)} 行")
        return data
    
    def save_data(self, data: pd.DataFrame, file_path: Union[str, Path]):
        """
        保存数据
        
        Args:
            data: 数据框
            file_path: 文件路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix.lower() == '.csv':
            data.to_csv(file_path, index=False)
        elif file_path.suffix.lower() == '.json':
            data.to_json(file_path, orient='records', indent=2)
        elif file_path.suffix.lower() == '.pkl':
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        logger.info(f"保存数据到: {file_path}")
    
    def process_social_media_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        处理社交媒体数据
        
        Args:
            data: 原始数据框
            
        Returns:
            处理后的数据框
        """
        logger.info("开始处理社交媒体数据...")
        
        # 检查必要的列
        if self.text_column not in data.columns:
            raise ValueError(f"数据中缺少文本列: {self.text_column}")
        
        # 移除空文本
        initial_count = len(data)
        data = data.dropna(subset=[self.text_column])
        data = data[data[self.text_column].str.strip() != '']
        logger.info(f"移除空文本后: {len(data)} 行 (移除 {initial_count - len(data)} 行)")
        
        # 文本长度过滤
        data['text_length'] = data[self.text_column].str.len()
        data = data[
            (data['text_length'] >= self.min_text_length) & 
            (data['text_length'] <= self.max_text_length)
        ]
        logger.info(f"长度过滤后: {len(data)} 行")
        
        # 清洗文本
        logger.info("清洗文本...")
        data['cleaned_text'] = self.preprocessor.clean_batch(data[self.text_column].tolist())
        
        # 提取特征
        logger.info("提取特征...")
        features_list = self.preprocessor.extract_batch_features(data['cleaned_text'].tolist())
        
        # 将特征添加到数据框
        if features_list:
            features_df = pd.DataFrame(features_list)
            
            # ⭐ 关键修复：确保索引对齐，避免特征与样本错位
            data = data.reset_index(drop=True)
            features_df.index = data.index
            
            # 确保特征列名不重复
            existing_cols = set(data.columns)
            for col in features_df.columns:
                if col in existing_cols:
                    features_df = features_df.rename(columns={col: f"{col}_feature"})
            
            data = pd.concat([data, features_df], axis=1)
            
            # 处理NaN值，避免StandardScaler报错
            data[features_df.columns] = data[features_df.columns].fillna(0.0)
            
            # 保存特征名称
            self.feature_names = list(features_df.columns)
        else:
            logger.warning("没有提取到特征")
            self.feature_names = []
        
        logger.info(f"处理完成: {len(data)} 行, {len(self.feature_names)} 个特征")
        return data
    
    def create_train_test_split(self, data: pd.DataFrame, 
                               test_size: float = 0.2,
                               random_state: int = 42,
                               stratify: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建训练测试分割
        
        Args:
            data: 数据框
            test_size: 测试集比例
            random_state: 随机种子
            stratify: 分层列名
            
        Returns:
            (训练集, 测试集)
        """
        if stratify and stratify in data.columns:
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state,
                stratify=data[stratify]
            )
        else:
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state
            )
        
        logger.info(f"训练集: {len(train_data)} 行, 测试集: {len(test_data)} 行")
        return train_data, test_data
    
    def prepare_features(self, data: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None,
                        label_column: Optional[str] = None,
                        fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        准备特征和标签
        
        Args:
            data: 数据框
            feature_columns: 特征列名列表
            label_column: 标签列名
            fit_scaler: 是否拟合标准化器
            
        Returns:
            (特征数组, 标签数组)
        """
        # 确定使用的特征列
        if feature_columns is None:
            feature_columns = self.feature_columns or self.feature_names or []
        
        if not feature_columns:
            raise ValueError("没有指定特征列")
        
        # 检查特征列是否存在
        missing_cols = [col for col in feature_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少特征列: {missing_cols}")
        
        # 提取特征
        X = data[feature_columns].values
        
        # 处理NaN值
        if np.isnan(X).any():
            logger.warning("发现NaN值，用0填充")
            X = np.nan_to_num(X, nan=0.0)
        
        # 标准化特征
        if fit_scaler or self.scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            logger.info(f"拟合标准化器，特征维度: {X.shape}")
        else:
            X = self.scaler.transform(X)
            logger.info(f"使用已有标准化器，特征维度: {X.shape}")
        
        # 处理标签
        y = None
        if label_column and label_column in data.columns:
            try:
                if self.label_encoder is None:
                    self.label_encoder = LabelEncoder()
                    y = self.label_encoder.fit_transform(data[label_column])
                    logger.info(f"拟合标签编码器，类别: {list(self.label_encoder.classes_)}")
                else:
                    y = self.label_encoder.transform(data[label_column])
                    logger.info(f"使用已有标签编码器")
            except ValueError as e:
                logger.error(f"标签编码失败: {e}")
                # 对于新类别，可以考虑使用OrdinalEncoder
                raise
        
        return X, y
    
    def save_processor(self, file_path: Union[str, Path]):
        """
        保存处理器状态
        
        Args:
            file_path: 文件路径
        """
        processor_state = {
            'text_column': self.text_column,
            'label_column': self.label_column,
            'min_text_length': self.min_text_length,
            'max_text_length': self.max_text_length,
            'feature_names': self.feature_names,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(processor_state, f)
        
        logger.info(f"保存处理器状态到: {file_path}")
        
        # 同时保存可读的JSON格式（用于跨语言/环境）
        json_path = file_path.with_suffix('.json')
        json_state = {
            'text_column': self.text_column,
            'label_column': self.label_column,
            'min_text_length': self.min_text_length,
            'max_text_length': self.max_text_length,
            'feature_names': self.feature_names,
            'feature_columns': self.feature_columns,
            'scaler_mean': self.scaler.mean_.tolist() if self.scaler else None,
            'scaler_scale': self.scaler.scale_.tolist() if self.scaler else None,
            'label_classes': self.label_encoder.classes_.tolist() if self.label_encoder else None
        }
        
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_state, f, indent=2, ensure_ascii=False)
        
        logger.info(f"保存JSON格式到: {json_path}")
    
    def load_processor(self, file_path: Union[str, Path]):
        """
        加载处理器状态
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        
        with open(file_path, 'rb') as f:
            processor_state = pickle.load(f)
        
        self.text_column = processor_state['text_column']
        self.label_column = processor_state['label_column']
        self.min_text_length = processor_state['min_text_length']
        self.max_text_length = processor_state['max_text_length']
        self.feature_names = processor_state['feature_names']
        self.feature_columns = processor_state.get('feature_columns')  # 兼容旧版本
        self.scaler = processor_state['scaler']
        self.label_encoder = processor_state['label_encoder']
        
        # 重新初始化TextPreprocessor
        self.preprocessor = TextPreprocessor(
            text_column=self.text_column,
            label_column=self.label_column,
            min_text_length=self.min_text_length,
            max_text_length=self.max_text_length
        )
        
        logger.info(f"加载处理器状态从: {file_path}")
        logger.info(f"特征列: {self.feature_columns or self.feature_names}")
    
    def get_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取数据摘要
        
        Args:
            data: 数据框
            
        Returns:
            数据摘要字典
        """
        summary = {
            'total_samples': len(data),
            'text_column': self.text_column,
            'label_column': self.label_column,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'text_length_stats': {
                'mean': data['text_length'].mean(),
                'std': data['text_length'].std(),
                'min': data['text_length'].min(),
                'max': data['text_length'].max()
            }
        }
        
        if self.label_column in data.columns:
            summary['label_distribution'] = data[self.label_column].value_counts().to_dict()
        
        return summary
