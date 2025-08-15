#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取器
从清洗后的文本中提取有用的特征
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """文本特征提取器"""
    
    def __init__(self):
        """初始化特征提取器"""
        # 抑郁相关词汇（基于PHQ-9问卷）
        self.depression_keywords = {
            '情绪低落': ['sad', 'depressed', 'down', 'blue', 'unhappy', 'miserable', 'hopeless'],
            '兴趣丧失': ['uninterested', 'bored', 'no_interest', 'nothing_matters', 'empty'],
            '睡眠问题': ['insomnia', 'sleep', 'tired', 'exhausted', 'fatigue', 'restless'],
            '食欲变化': ['appetite', 'hungry', 'not_hungry', 'weight_loss', 'weight_gain'],
            '注意力问题': ['concentrate', 'focus', 'attention', 'distracted', 'mind_wandering'],
            '自我评价': ['worthless', 'failure', 'useless', 'guilty', 'blame_myself'],
            '自杀想法': ['suicide', 'kill_myself', 'death', 'die', 'end_it_all', 'better_off_dead'],
            '焦虑症状': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stress'],
            '身体症状': ['pain', 'ache', 'sick', 'ill', 'headache', 'stomach'],
            '社交退缩': ['alone', 'lonely', 'isolated', 'no_friends', 'avoid_people']
        }
        
        # 情感词汇
        self.emotion_words = {
            'positive': ['happy', 'joy', 'excited', 'love', 'great', 'wonderful', 'amazing', 'fantastic'],
            'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'hate', 'terrible', 'awful', 'horrible'],
            'neutral': ['okay', 'fine', 'normal', 'average', 'usual', 'regular']
        }
        
        # 标点符号模式
        self.punctuation_patterns = {
            'exclamation': r'!+',
            'question': r'\?+',
            'ellipsis': r'\.{3,}',
            'caps_words': r'\b[A-Z]{2,}\b'
        }
        
        # 编译正则表达式
        self.compiled_patterns = {k: re.compile(v) for k, v in self.punctuation_patterns.items()}
    
    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """
        提取语言学特征
        
        Args:
            text: 输入文本
            
        Returns:
            语言学特征字典
        """
        features = {}
        
        # 基本统计特征
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['avg_sentence_length'] = features['word_count'] / features['sentence_count'] if features['sentence_count'] > 0 else 0
        
        # 词汇多样性
        words = text.lower().split()
        if words:
            features['unique_words'] = len(set(words))
            features['lexical_diversity'] = features['unique_words'] / features['word_count']
            features['type_token_ratio'] = features['unique_words'] / features['word_count']
        else:
            features['unique_words'] = 0
            features['lexical_diversity'] = 0
            features['type_token_ratio'] = 0
        
        # 标点符号特征
        features['exclamation_count'] = len(self.compiled_patterns['exclamation'].findall(text))
        features['question_count'] = len(self.compiled_patterns['question'].findall(text))
        features['ellipsis_count'] = len(self.compiled_patterns['ellipsis'].findall(text))
        features['caps_words_count'] = len(self.compiled_patterns['caps_words'].findall(text))
        
        # 大写字母比例
        features['uppercase_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        
        return features
    
    def extract_depression_features(self, text: str) -> Dict[str, float]:
        """
        提取抑郁相关特征
        
        Args:
            text: 输入文本
            
        Returns:
            抑郁相关特征字典
        """
        features = {}
        text_lower = text.lower()
        
        # 计算各类抑郁关键词的出现次数
        for category, keywords in self.depression_keywords.items():
            count = sum(text_lower.count(keyword) for keyword in keywords)
            features[f'depression_{category}_count'] = count
        
        # 总抑郁关键词数量
        all_depression_words = [word for words in self.depression_keywords.values() for word in words]
        features['total_depression_words'] = sum(text_lower.count(word) for word in all_depression_words)
        
        # 抑郁词汇密度
        word_count = len(text.split())
        features['depression_word_density'] = features['total_depression_words'] / word_count if word_count > 0 else 0
        
        # 抑郁词汇类别数
        features['depression_categories'] = sum(1 for category, keywords in self.depression_keywords.items() 
                                              if any(text_lower.count(keyword) > 0 for keyword in keywords))
        
        return features
    
    def extract_emotion_features(self, text: str) -> Dict[str, float]:
        """
        提取情感特征
        
        Args:
            text: 输入文本
            
        Returns:
            情感特征字典
        """
        features = {}
        text_lower = text.lower()
        
        # 计算各类情感词汇的出现次数
        for emotion, words in self.emotion_words.items():
            count = sum(text_lower.count(word) for word in words)
            features[f'{emotion}_emotion_count'] = count
        
        # 情感词汇总数
        all_emotion_words = [word for words in self.emotion_words.values() for word in words]
        features['total_emotion_words'] = sum(text_lower.count(word) for word in all_emotion_words)
        
        # 情感词汇密度
        word_count = len(text.split())
        features['emotion_word_density'] = features['total_emotion_words'] / word_count if word_count > 0 else 0
        
        # 情感极性
        positive_count = features['positive_emotion_count']
        negative_count = features['negative_emotion_count']
        total_emotion = positive_count + negative_count
        
        if total_emotion > 0:
            features['emotion_polarity'] = (positive_count - negative_count) / total_emotion
        else:
            features['emotion_polarity'] = 0
        
        return features
    
    def extract_social_media_features(self, text: str) -> Dict[str, float]:
        """
        提取社交媒体特定特征
        
        Args:
            text: 输入文本
            
        Returns:
            社交媒体特征字典
        """
        features = {}
        
        # 社交媒体标记
        features['hashtag_count'] = len(re.findall(r'#\w+', text))
        features['mention_count'] = len(re.findall(r'@\w+', text))
        features['url_count'] = len(re.findall(r'http[s]?://\S+', text))
        features['emoticon_count'] = len(re.findall(r'[:;=]-?[)(/|\\pPoO]', text))
        
        # 重复字符和单词
        features['repeated_char_count'] = len(re.findall(r'(.)\1{2,}', text))
        features['repeated_word_count'] = len(re.findall(r'\b(\w+)(\s+\1){2,}\b', text))
        
        # 感叹号和问号的使用
        features['exclamation_density'] = text.count('!') / len(text) if text else 0
        features['question_density'] = text.count('?') / len(text) if text else 0
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        提取所有特征
        
        Args:
            text: 输入文本
            
        Returns:
            所有特征的字典
        """
        features = {}
        
        # 合并所有特征
        features.update(self.extract_linguistic_features(text))
        features.update(self.extract_depression_features(text))
        features.update(self.extract_emotion_features(text))
        features.update(self.extract_social_media_features(text))
        
        return features
    
    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            特征字典列表
        """
        return [self.extract_all_features(text) for text in texts]
    
    def get_feature_names(self) -> List[str]:
        """
        获取所有特征名称
        
        Returns:
            特征名称列表
        """
        # 使用示例文本获取特征名称
        sample_text = "This is a sample text for feature extraction."
        features = self.extract_all_features(sample_text)
        return list(features.keys())
    
    def features_to_array(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        """
        将特征字典列表转换为numpy数组
        
        Args:
            features_list: 特征字典列表
            
        Returns:
            numpy数组
        """
        if not features_list:
            return np.array([])
        
        # 获取所有特征名称
        feature_names = list(features_list[0].keys())
        
        # 转换为数组
        array = np.array([[features.get(name, 0.0) for name in feature_names] 
                         for features in features_list])
        
        return array
