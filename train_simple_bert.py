#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版BERT模型训练脚本（CPU友好）
专为CPU训练优化的轻量版本
"""

import sys
import os
import logging
import random
import numpy as np
import pandas as pd
from pathlib import Path

# 创建日志目录
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'simple_training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.preprocessor import DataProcessor
from models.model_factory import ModelFactory


def create_simple_bert_model():
    """创建简化的BERT模型（CPU友好）"""
    logger.info("创建简化BERT模型...")
    
    try:
        # 非常轻量的配置
        model_params = {
            'symptom_capsules': 4,        # 减少到4个胶囊
            'capsule_dim': 8,             # 减少胶囊维度
            'max_length': 64,             # 大幅减少序列长度
            'dropout': 0.1,               # 减少dropout
            'num_iterations': 3           # 减少路由迭代
        }
        
        model = ModelFactory.create_model(
            model_type='bert_capsule',
            bert_model_name='distilbert-base-uncased',  # 使用DistilBERT（更小更快）
            **model_params
        )
        
        # 保存模型参数到模型对象中
        model.model_params = model_params
        model.bert_model_name = 'distilbert-base-uncased'
        
        logger.info("✅ 成功创建简化BERT模型")
        return model
    except Exception as e:
        logger.error(f"❌ 创建模型失败: {e}")
        return None


def load_sample_data():
    """加载少量样本数据进行快速测试"""
    logger.info("加载样本数据...")
    
    # 创建少量示例数据
    sample_texts = [
        "I feel so sad and hopeless today",
        "Had a great day with friends",
        "I can't sleep at night",
        "Feeling really happy today",
        "I feel like a burden",
        "Today was productive",
        "Everything feels meaningless",
        "Just had an amazing day",
        "I feel so alone",
        "Life is beautiful"
    ] * 5  # 重复5次得到50个样本
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5
    
    data = pd.DataFrame({
        'text': sample_texts,
        'label': labels
    })
    
    # 简单分割
    train_data = data[:30]  # 30个训练样本
    val_data = data[30:40]   # 10个验证样本
    test_data = data[40:]    # 10个测试样本
    
    return {
        'train_texts': train_data['text'].tolist(),
        'val_texts': val_data['text'].tolist(),
        'test_texts': test_data['text'].tolist(),
        'y_train': train_data['label'].values,
        'y_val': val_data['label'].values,
        'y_test': test_data['label'].values
    }


def load_real_data():
    """加载真实数据集"""
    logger.info("加载真实数据集...")
    
    try:
        # 尝试加载真实数据集
        data_path = Path('data/raw/simple_sample_data_5000.csv')
        if data_path.exists():
            logger.info(f"📁 找到数据集: {data_path}")
            data = pd.read_csv(data_path)
            logger.info(f"📊 数据集信息: {len(data)} 行, 列: {list(data.columns)}")
            
            # 检查必要的列
            if 'text' not in data.columns or 'label' not in data.columns:
                logger.error("❌ 数据集缺少必要的列: text 或 label")
                return None
            
            # 数据预处理
            processor = DataProcessor()
            processed_data = processor.process_social_media_data(data)
            
            logger.info(f"✅ 数据预处理完成: {len(processed_data)} 行")
            
            # 分割数据
            from sklearn.model_selection import train_test_split
            
            # 首先分割出测试集
            train_val_data, test_data = train_test_split(
                processed_data, 
                test_size=0.2, 
                random_state=42, 
                stratify=processed_data['label']
            )
            
            # 然后分割训练集和验证集
            train_data, val_data = train_test_split(
                train_val_data, 
                test_size=0.2, 
                random_state=42, 
                stratify=train_val_data['label']
            )
            
            logger.info(f"📊 数据分割完成:")
            logger.info(f"  训练集: {len(train_data)} 行")
            logger.info(f"  验证集: {len(val_data)} 行") 
            logger.info(f"  测试集: {len(test_data)} 行")
            
            # 检查标签分布
            logger.info("📈 标签分布:")
            logger.info(f"  训练集: {train_data['label'].value_counts().to_dict()}")
            logger.info(f"  验证集: {val_data['label'].value_counts().to_dict()}")
            logger.info(f"  测试集: {test_data['label'].value_counts().to_dict()}")
            
            return {
                'train_texts': train_data['text'].tolist(),
                'val_texts': val_data['text'].tolist(),
                'test_texts': test_data['text'].tolist(),
                'y_train': train_data['label'].values,
                'y_val': val_data['label'].values,
                'y_test': test_data['label'].values
            }
        else:
            logger.warning("⚠️ 未找到真实数据集，使用示例数据")
            return load_sample_data()
            
    except Exception as e:
        logger.error(f"❌ 加载真实数据集失败: {e}")
        logger.warning("⚠️ 回退到示例数据")
        return load_sample_data()


def train_simple_model():
    """训练简化模型"""
    logger.info("🚀 开始简化BERT模型训练...")
    
    # 1. 创建模型
    model = create_simple_bert_model()
    if model is None:
        logger.error("❌ 无法创建模型")
        return False
    
    # 2. 加载数据
    data = load_real_data() # Changed to load_real_data
    logger.info(f"📊 数据加载完成: 训练{len(data['train_texts'])}, 验证{len(data['val_texts'])}, 测试{len(data['test_texts'])}")
    
    # 3. 训练模型
    logger.info("🔥 开始训练...")
    try:
        history = model.train(
            texts=data['train_texts'],
            y_train=data['y_train'],
            val_texts=data['val_texts'],
            y_val=data['y_val'],
            epochs=30,           # 增加训练轮数
            batch_size=8,       # 适中的批次大小
            learning_rate=2e-5, # 标准BERT学习率
            weight_decay=0.01,
            warmup_steps=100,   # 增加预热步数
            early_stopping_patience=5
        )
        logger.info("✅ 训练完成")
        
        # 4. 评估模型
        logger.info("📊 开始评估...")
        metrics = model.evaluate(data['test_texts'], data['y_test'])
        
        logger.info("📈 评估结果:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1']:.4f}")
        
        # 5. 保存模型
        model_dir = Path('models')
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "simple_bert_model.pkl"
        model.save_model(model_path)
        logger.info(f"💾 模型已保存到: {model_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("简化版BERT模型训练（CPU友好）")
    logger.info("=" * 60)
    
    # 检查PyTorch
    try:
        import torch
        logger.info(f"PyTorch版本: {torch.__version__}")
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("使用CPU训练")
    except ImportError:
        logger.error("❌ PyTorch未安装")
        return
    
    # 开始训练
    success = train_simple_model()
    
    if success:
        logger.info("🎉 训练成功完成！")
        logger.info("💡 现在可以使用 python ensemble_predictor.py 进行预测")
    else:
        logger.error("❌ 训练失败")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
