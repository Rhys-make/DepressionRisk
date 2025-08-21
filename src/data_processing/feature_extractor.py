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
            '情绪低落': ['sad', 'depressed', 'down', 'blue', 'unhappy', 'miserable', 'hopeless', 'sadness', 'depression'],
            '兴趣丧失': ['uninterested', 'bored', 'no_interest', 'nothing_matters', 'empty', 'meaningless', 'pointless'],
            '睡眠问题': ['insomnia', 'sleep', 'tired', 'exhausted', 'fatigue', 'restless', 'cant_sleep', 'sleepless'],
            '食欲变化': ['appetite', 'hungry', 'not_hungry', 'weight_loss', 'weight_gain', 'eating'],
            '注意力问题': ['concentrate', 'focus', 'attention', 'distracted', 'mind_wandering', 'racing_thoughts'],
            '自我评价': ['worthless', 'failure', 'useless', 'guilty', 'blame_myself', 'hate_myself', 'disappointment'],
            '自杀想法': ['suicide', 'kill_myself', 'death', 'die', 'end_it_all', 'better_off_dead', 'want_to_die', 'end_my_life'],
            '焦虑症状': ['anxious', 'worried', 'nervous', 'panic', 'fear', 'stress', 'anxiety'],
            '身体症状': ['pain', 'ache', 'sick', 'ill', 'headache', 'stomach', 'hurting'],
            '社交退缩': ['alone', 'lonely', 'isolated', 'no_friends', 'avoid_people', 'loneliness']
        }
        
        # 情感词汇（扩展版）
        self.emotion_words = {
            'positive': ['happy', 'joy', 'excited', 'love', 'great', 'wonderful', 'amazing', 'fantastic', 'blessed', 'grateful', 'optimistic', 'good', 'nice', 'perfect', 'awesome', 'brilliant', 'excellent', 'fixed', 'solved', 'better', 'relief', 'cheered', 'helped'],
            'negative': ['sad', 'angry', 'frustrated', 'disappointed', 'hate', 'terrible', 'awful', 'horrible', 'hopeless', 'worthless', 'miserable', 'stressful', 'stress', 'anxiety', 'worried', 'scared', 'afraid', 'tired', 'exhausted', 'lonely', 'alone', 'useless', 'failure'],
            'neutral': ['okay', 'fine', 'normal', 'average', 'usual', 'regular']
        }
        
        # 转折词（新增）
        self.turnaround_words = {
            '但是', '不过', '然而', '可是', '只是', '不过', '但', 'yet', 'but', 'however',
            'although', 'though', 'even though', 'despite', 'in spite of', 'while',
            '虽然', '尽管', '即使', '就算', '哪怕', '纵然', '虽说', '虽说如此',
            'lol', 'haha', '😅', '😊', '😋', '😄', '😂', '😆', '😉', '😎'
        }
        
        # 情感转折模式（新增）
        self.turnaround_patterns = [
            r'但是.*?(开心|快乐|高兴|好|棒|赞|爽|舒服|轻松|放松|解决|修复|治愈|fixed|solved|better|relief)',
            r'不过.*?(开心|快乐|高兴|好|棒|赞|爽|舒服|轻松|放松|解决|修复|治愈|fixed|solved|better|relief)',
            r'然而.*?(开心|快乐|高兴|好|棒|赞|爽|舒服|轻松|放松|解决|修复|治愈|fixed|solved|better|relief)',
            r'but.*?(happy|good|great|amazing|wonderful|fantastic|fixed|solved|better|relief|cheered|helped)',
            r'however.*?(happy|good|great|amazing|wonderful|fantastic|fixed|solved|better|relief|cheered|helped)',
            r'yet.*?(happy|good|great|amazing|wonderful|fantastic|fixed|solved|better|relief|cheered|helped)',
            r'😩.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😢.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😭.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😔.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😞.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😤.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😫.*?(😊|😋|😄|😂|😆|😉|😎)',
            r'😩.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😢.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😭.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😔.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😞.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😤.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)',
            r'😫.*?(lol|haha|😅|😊|😋|😄|😂|😆|😉|😎)'
        ]
        
        # 标点符号模式
        self.punctuation_patterns = {
            'exclamation': r'!+',
            'question': r'\?+',
            'ellipsis': r'\.{3,}',
            'caps_words': r'\b[A-Z]{2,}\b'
        }
        
        # 编译正则表达式
        self.compiled_patterns = {k: re.compile(v) for k, v in self.punctuation_patterns.items()}
        self.compiled_turnaround_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.turnaround_patterns]
    
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
        
        # 计算各类抑郁关键词的出现次数（改进匹配方法）
        for category, keywords in self.depression_keywords.items():
            count = 0
            for keyword in keywords:
                # 使用单词边界匹配，避免部分匹配
                pattern = r'\b' + re.escape(keyword) + r'\b'
                count += len(re.findall(pattern, text_lower))
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
        
        try:
            # 合并所有特征
            features.update(self.extract_linguistic_features(text))
            features.update(self.extract_depression_features(text))
            features.update(self.extract_emotion_features(text))
            features.update(self.extract_social_media_features(text))
            # 添加情感转折特征（新增）
            features.update(self.extract_turnaround_features(text))
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            # 返回默认特征
            features = self._get_default_features()
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """
        获取默认特征（当特征提取失败时使用）
        
        Returns:
            默认特征字典
        """
        # 获取所有特征名称
        sample_text = "This is a sample text for feature extraction."
        try:
            default_features = self.extract_linguistic_features(sample_text)
            default_features.update(self.extract_depression_features(sample_text))
            default_features.update(self.extract_emotion_features(sample_text))
            default_features.update(self.extract_social_media_features(sample_text))
            default_features.update(self.extract_turnaround_features(sample_text))
            # 将所有值设为0
            return {k: 0.0 for k in default_features.keys()}
        except:
            # 如果连默认特征都获取失败，返回基本特征
            return {
                'text_length': 0.0, 'word_count': 0.0, 'char_count': 0.0,
                'avg_word_length': 0.0, 'sentence_count': 0.0, 'avg_sentence_length': 0.0,
                'unique_words': 0.0, 'lexical_diversity': 0.0, 'type_token_ratio': 0.0,
                'exclamation_count': 0.0, 'question_count': 0.0, 'ellipsis_count': 0.0,
                'caps_words_count': 0.0, 'uppercase_ratio': 0.0
            }
    
    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            特征字典列表
        """
        if not texts:
            return []
        
        # 获取标准特征名称
        standard_features = self._get_default_features()
        feature_names = list(standard_features.keys())
        
        features_list = []
        for text in texts:
            try:
                features = self.extract_all_features(text)
                # 确保所有特征都存在
                for name in feature_names:
                    if name not in features:
                        features[name] = 0.0
                features_list.append(features)
            except Exception as e:
                logger.warning(f"特征提取失败: {e}")
                # 返回默认特征字典
                features_list.append(standard_features.copy())
        
        return features_list
    
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
    
    def extract_turnaround_features(self, text: str) -> Dict[str, float]:
        """
        提取情感转折特征（新增）
        
        Args:
            text: 输入文本
            
        Returns:
            情感转折特征字典
        """
        features = {}
        text_lower = text.lower()
        
        # 转折词计数
        turnaround_count = 0
        for word in self.turnaround_words:
            if word in text_lower:
                turnaround_count += 1
        features['turnaround_word_count'] = turnaround_count
        
        # 转折模式匹配
        pattern_matches = 0
        for pattern in self.compiled_turnaround_patterns:
            if pattern.search(text_lower):
                pattern_matches += 1
        features['turnaround_pattern_count'] = pattern_matches
        
        # 转折强度（转折词数量 / 总词数）
        total_words = len(text.split())
        if total_words > 0:
            features['turnaround_intensity'] = turnaround_count / total_words
        else:
            features['turnaround_intensity'] = 0.0
        
        # 情感变化分析
        sentences = re.split(r'[.!?。！？]', text)
        features['sentence_count'] = len([s for s in sentences if s.strip()])
        
        # 分析第一个和最后一个句子的情感
        negative_first = 0
        positive_first = 0
        negative_last = 0
        positive_last = 0
        
        if sentences:
            first_sentence = sentences[0].lower()
            last_sentence = sentences[-1].lower()
            
            # 第一个句子的情感
            for word in self.emotion_words['negative']:
                if word in first_sentence:
                    negative_first += 1
            for word in self.emotion_words['positive']:
                if word in first_sentence:
                    positive_first += 1
            
            # 最后一个句子的情感
            for word in self.emotion_words['negative']:
                if word in last_sentence:
                    negative_last += 1
            for word in self.emotion_words['positive']:
                if word in last_sentence:
                    positive_last += 1
        
        features['negative_first_sentence'] = negative_first
        features['positive_first_sentence'] = positive_first
        features['negative_last_sentence'] = negative_last
        features['positive_last_sentence'] = positive_last
        
        # 情感变化方向
        if negative_first > positive_first and positive_last > negative_last:
            features['sentiment_turnaround'] = 1.0  # 负面转正面
        elif positive_first > negative_first and negative_last > positive_last:
            features['sentiment_turnaround'] = -1.0  # 正面转负面
        else:
            features['sentiment_turnaround'] = 0.0  # 无变化或同向
        
        # 整体情感倾向（考虑转折）
        negative_count = sum(1 for word in self.emotion_words['negative'] if word in text_lower)
        positive_count = sum(1 for word in self.emotion_words['positive'] if word in text_lower)
        
        if negative_count + positive_count > 0:
            base_sentiment = (positive_count - negative_count) / (negative_count + positive_count)
            
            # 如果有转折词，调整情感分数
            if turnaround_count > 0:
                # 转折词越多，情感越偏向中性或正面
                adjustment = min(turnaround_count * 0.2, 0.5)  # 最多调整0.5
                adjusted_sentiment = base_sentiment + adjustment
                features['overall_sentiment'] = max(-1.0, min(1.0, adjusted_sentiment))
            else:
                features['overall_sentiment'] = base_sentiment
        else:
            features['overall_sentiment'] = 0.0
        
        return features
