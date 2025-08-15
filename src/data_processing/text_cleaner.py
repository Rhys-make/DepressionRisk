#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗器
专门处理社交媒体文本的清洗工作
"""

import re
import html
import unicodedata
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """社交媒体文本清洗器"""
    
    def __init__(self):
        """初始化文本清洗器"""
        # 表情符号映射
        self.emoticon_map = {
            ':)': ' [happy] ', ':-)': ' [happy] ', ':D': ' [happy] ', ':-D': ' [happy] ',
            ':(': ' [sad] ', ':-(': ' [sad] ', ':((': ' [sad] ', ':-(((': ' [sad] ',
            ';)': ' [wink] ', ';-)': ' [wink] ',
            ':P': ' [tongue] ', ':-P': ' [tongue] ', ':p': ' [tongue] ', ':-p': ' [tongue] ',
            ':O': ' [surprised] ', ':-O': ' [surprised] ', ':o': ' [surprised] ', ':-o': ' [surprised] ',
            ':*': ' [kiss] ', ':-*': ' [kiss] ',
            '<3': ' [heart] ', '</3': ' [broken_heart] ',
            'xD': ' [laugh] ', 'XD': ' [laugh] ',
            'T_T': ' [crying] ', 'T.T': ' [crying] ', 'T-T': ' [crying] ',
            '^_^': ' [smile] ', '^^': ' [smile] ',
            'o_o': ' [confused] ', 'O_O': ' [confused] ', 'o.O': ' [confused] ',
            '>:(': ' [angry] ', '>:((': ' [angry] ',
            ':|': ' [neutral] ', ':-|': ' [neutral] '
        }
        
        # 社交媒体特定模式
        self.patterns = {
            'url': r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})',
            'date': r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b',
            'hashtag': r'#\w+',
            'mention': r'@\w+',
            'number': r'\b\d+\b',
            'repeated_chars': r'(.)\1{2,}',  # 重复字符超过2次
            'repeated_words': r'\b(\w+)(\s+\1){2,}\b',  # 重复单词
            'extra_spaces': r'\s+',
            'leading_trailing_spaces': r'^\s+|\s+$'
        }
        
        # 编译正则表达式
        self.compiled_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in self.patterns.items()}
    
    def clean_text(self, text: str, 
                   remove_urls: bool = True,
                   remove_emails: bool = True,
                   remove_phones: bool = True,
                   remove_dates: bool = False,
                   remove_times: bool = False,
                   remove_hashtags: bool = False,
                   remove_mentions: bool = False,
                   remove_numbers: bool = False,
                   normalize_emoticons: bool = True,
                   remove_repeated_chars: bool = True,
                   remove_repeated_words: bool = True,
                   lowercase: bool = True) -> str:
        """
        清洗文本
        
        Args:
            text: 输入文本
            remove_urls: 是否移除URL
            remove_emails: 是否移除邮箱
            remove_phones: 是否移除电话号码
            remove_dates: 是否移除日期
            remove_times: 是否移除时间
            remove_hashtags: 是否移除话题标签
            remove_mentions: 是否移除@提及
            remove_numbers: 是否移除数字
            normalize_emoticons: 是否标准化表情符号
            remove_repeated_chars: 是否移除重复字符
            remove_repeated_words: 是否移除重复单词
            lowercase: 是否转换为小写
            
        Returns:
            清洗后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 解码HTML实体
        text = html.unescape(text)
        
        # 标准化Unicode字符
        text = unicodedata.normalize('NFKC', text)
        
        # 移除URL
        if remove_urls:
            text = self.compiled_patterns['url'].sub(' [URL] ', text)
        
        # 移除邮箱
        if remove_emails:
            text = self.compiled_patterns['email'].sub(' [EMAIL] ', text)
        
        # 移除电话号码
        if remove_phones:
            text = self.compiled_patterns['phone'].sub(' [PHONE] ', text)
        
        # 移除日期
        if remove_dates:
            text = self.compiled_patterns['date'].sub(' [DATE] ', text)
        
        # 移除时间
        if remove_times:
            text = self.compiled_patterns['time'].sub(' [TIME] ', text)
        
        # 移除话题标签
        if remove_hashtags:
            text = self.compiled_patterns['hashtag'].sub(' [HASHTAG] ', text)
        
        # 移除@提及
        if remove_mentions:
            text = self.compiled_patterns['mention'].sub(' [MENTION] ', text)
        
        # 移除数字
        if remove_numbers:
            text = self.compiled_patterns['number'].sub(' [NUMBER] ', text)
        
        # 标准化表情符号
        if normalize_emoticons:
            text = self._normalize_emoticons(text)
        
        # 移除重复字符
        if remove_repeated_chars:
            text = self.compiled_patterns['repeated_chars'].sub(r'\1\1', text)
        
        # 移除重复单词
        if remove_repeated_words:
            text = self.compiled_patterns['repeated_words'].sub(r'\1', text)
        
        # 转换为小写
        if lowercase:
            text = text.lower()
        
        # 清理多余空格
        text = self.compiled_patterns['extra_spaces'].sub(' ', text)
        text = self.compiled_patterns['leading_trailing_spaces'].sub('', text)
        
        return text.strip()
    
    def _normalize_emoticons(self, text: str) -> str:
        """标准化表情符号"""
        for emoticon, replacement in self.emoticon_map.items():
            text = text.replace(emoticon, replacement)
        return text
    
    def extract_features(self, text: str) -> Dict[str, any]:
        """
        提取文本特征
        
        Args:
            text: 输入文本
            
        Returns:
            特征字典
        """
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'avg_word_length': 0,
            'has_url': bool(self.compiled_patterns['url'].search(text)),
            'has_email': bool(self.compiled_patterns['email'].search(text)),
            'has_phone': bool(self.compiled_patterns['phone'].search(text)),
            'has_hashtag': bool(self.compiled_patterns['hashtag'].search(text)),
            'has_mention': bool(self.compiled_patterns['mention'].search(text)),
            'has_number': bool(self.compiled_patterns['number'].search(text)),
            'emoticon_count': 0,
            'repeated_char_count': 0,
            'repeated_word_count': 0
        }
        
        # 计算平均单词长度
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        
        # 计算表情符号数量
        for emoticon in self.emoticon_map.keys():
            features['emoticon_count'] += text.count(emoticon)
        
        # 计算重复字符数量
        repeated_chars = self.compiled_patterns['repeated_chars'].findall(text)
        features['repeated_char_count'] = len(repeated_chars)
        
        # 计算重复单词数量
        repeated_words = self.compiled_patterns['repeated_words'].findall(text)
        features['repeated_word_count'] = len(repeated_words)
        
        return features
    
    def clean_batch(self, texts: List[str], **kwargs) -> List[str]:
        """
        批量清洗文本
        
        Args:
            texts: 文本列表
            **kwargs: 传递给clean_text的参数
            
        Returns:
            清洗后的文本列表
        """
        return [self.clean_text(text, **kwargs) for text in texts]
    
    def extract_batch_features(self, texts: List[str]) -> List[Dict[str, any]]:
        """
        批量提取特征
        
        Args:
            texts: 文本列表
            
        Returns:
            特征字典列表
        """
        return [self.extract_features(text) for text in texts]
