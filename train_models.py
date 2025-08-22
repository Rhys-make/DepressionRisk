#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT模型训练主脚本
训练抑郁风险预测BERT模型
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
        logging.FileHandler(log_dir / 'training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 设置随机种子以确保可重现性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# 设置PyTorch随机种子（如果可用）
try:
    import torch
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("未检测到GPU，将使用CPU训练（可能很慢）")
except ImportError:
    logger.error("PyTorch未安装，无法使用BERT模型")
    sys.exit(1)

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.preprocessor import DataProcessor
from models.model_factory import ModelFactory, ModelManager


def load_and_preprocess_data(data_path: str = "data/raw/simple_sample_data_5000.csv"):
    """
    加载和预处理数据
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        处理后的数据
    """
    logger.info("开始加载和预处理数据...")
    
    # 初始化数据处理器
    processor = DataProcessor(
        text_column='text',
        label_column='label',
        min_text_length=10,
        max_text_length=500
    )
    
    # 加载数据
    if os.path.exists(data_path):
        data = processor.load_data(data_path)
    else:
        # 创建示例数据
        logger.warning(f"数据文件 {data_path} 不存在，创建示例数据")
        data = create_sample_data()
    
    # 处理数据
    processed_data = processor.process_social_media_data(data)
    
    # 创建训练/验证/测试分割 (60/20/20)
    train_data, temp_data = processor.create_train_test_split(processed_data, test_size=0.4, random_state=42)
    val_data, test_data = processor.create_train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # 获取文本数据（用于BERT模型）
    train_texts = train_data['text'].tolist()
    val_texts = val_data['text'].tolist()
    test_texts = test_data['text'].tolist()
    
    # 获取标签
    y_train = train_data['label'].values
    y_val = val_data['label'].values
    y_test = test_data['label'].values
    
    logger.info(f"数据预处理完成:")
    logger.info(f"  训练集: {len(train_texts)} 样本")
    logger.info(f"  验证集: {len(val_texts)} 样本")
    logger.info(f"  测试集: {len(test_texts)} 样本")
    
    return {
        'train_texts': train_texts,
        'val_texts': val_texts,
        'test_texts': test_texts,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'processor': processor
    }


def create_sample_data():
    """创建示例数据"""
    sample_texts = [
        "I feel so sad and hopeless today. Nothing seems to matter anymore. :(",
        "Had a great day with friends! Everything is wonderful! :D",
        "I can't sleep at night. My mind keeps racing with negative thoughts.",
        "Feeling really happy and grateful for all the good things in my life!",
        "Sometimes I think about hurting myself. Life is just too hard.",
        "Excited about the new opportunities coming my way!",
        "I feel like I'm a burden to everyone around me.",
        "Today was productive and I accomplished a lot!",
        "I don't see the point in living anymore.",
        "Looking forward to spending time with family this weekend!",
        "Everything feels meaningless and I'm tired of pretending to be okay.",
        "Just had an amazing workout session! Feeling energized!",
        "I wish I could just disappear sometimes.",
        "Grateful for the support of my friends and family.",
        "I feel so alone even when I'm surrounded by people.",
        "Had a wonderful conversation with a friend today!",
        "I can't stop thinking about all my failures.",
        "Excited about my new job opportunity!",
        "I feel like I'm drowning in sadness.",
        "Life is beautiful and full of possibilities!"
    ]
    
    # 创建标签（0=正常，1=抑郁风险）
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    return pd.DataFrame({
        'text': sample_texts,
        'label': labels
    })


def create_models():
    """创建要训练的BERT模型"""
    logger.info("创建BERT模型...")
    
    # 检查PyTorch和GPU可用性
    torch_available = False
    gpu_available = False
    try:
        import torch
        torch_available = True
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("GPU不可用，BERT训练将很慢")
    except ImportError:
        logger.error("PyTorch未安装，无法使用BERT模型")
        return {}
    
    model_configs = [
        {
            'name': 'bert_capsule',
            'model_type': 'bert_capsule',
            'bert_model_name': 'bert-base-uncased',
            'symptom_capsules': 12,        # 症状胶囊数量
            'capsule_dim': 32,             # 胶囊维度
            'max_length': 512,             # 最大长度
            'dropout': 0.2,                # dropout
            'num_iterations': 5            # 路由迭代次数
        },
        {
            'name': 'bert_capsule_advanced',
            'model_type': 'bert_capsule',
            'bert_model_name': 'bert-base-uncased',
            'symptom_capsules': 16,        # 更多症状胶囊
            'capsule_dim': 64,             # 更大胶囊维度
            'max_length': 512,             # 最大长度
            'dropout': 0.3,                # 更高dropout
            'num_iterations': 7,           # 更多路由迭代
            'use_contrastive_loss': True   # 启用对比学习
        }
    ]
    
    models = {}
    for config in model_configs:
        try:
            # 如果没有GPU，使用更轻量的配置
            if not gpu_available:
                logger.warning(f"在CPU上创建BERT模型 {config['name']}（训练将很慢）")
                # 为CPU训练调整配置
                config = config.copy()  # 复制配置避免修改原始配置
                config['max_length'] = 128  # 减少序列长度
                config['symptom_capsules'] = min(config.get('symptom_capsules', 8), 8)  # 减少胶囊数
                config['capsule_dim'] = min(config.get('capsule_dim', 16), 16)  # 减少胶囊维度
                
            model = ModelFactory.create_model(
                model_type=config['model_type'],
                **{k: v for k, v in config.items() if k not in ['name', 'model_type']}
            )
            models[config['name']] = model
            logger.info(f"成功创建模型: {config['name']}")
        except Exception as e:
            logger.warning(f"创建模型 {config['name']} 失败: {e}")
    
    return models


def train_models(models, data):
    """训练所有BERT模型"""
    logger.info("开始训练BERT模型...")
    
    training_results = {}
    
    for name, model in models.items():
        logger.info(f"训练模型: {name}")
        try:
            # 检查是否有GPU，调整训练参数
            import torch
            gpu_available = torch.cuda.is_available()
            
            if gpu_available:
                # GPU训练参数
                epochs = 15
                batch_size = 16
                learning_rate = 3e-5
            else:
                # CPU训练参数（更轻量）
                epochs = 5  # 减少训练轮数
                batch_size = 4  # 减少批次大小
                learning_rate = 5e-5  # 稍微提高学习率
                logger.info(f"使用CPU训练参数: epochs={epochs}, batch_size={batch_size}")
            
            history = model.train(
                texts=data['train_texts'],
                y_train=data['y_train'],
                val_texts=data['val_texts'],
                y_val=data['y_val'],
                epochs=epochs,               # 动态调整的训练轮数
                batch_size=batch_size,       # 动态调整的批次大小
                learning_rate=learning_rate, # 动态调整的学习率
                weight_decay=0.01,           # 权重衰减
                warmup_steps=50,             # 减少预热步数
                early_stopping_patience=5    # 减少早停耐心
            )
            
            training_results[name] = {
                'success': True,
                'history': history
            }
            logger.info(f"模型 {name} 训练成功")
            
        except Exception as e:
            logger.error(f"训练模型 {name} 失败: {e}")
            training_results[name] = {
                'success': False,
                'error': str(e)
            }
    
    return training_results


def evaluate_models(models, data):
    """评估所有BERT模型"""
    logger.info("开始评估BERT模型...")
    
    evaluation_results = {}
    
    for name, model in models.items():
        if not model.is_trained:
            logger.warning(f"模型 {name} 尚未训练，跳过评估")
            continue
        
        logger.info(f"评估模型: {name}")
        try:
            metrics = model.evaluate(data['test_texts'], data['y_test'])
            
            evaluation_results[name] = {
                'success': True,
                'metrics': metrics
            }
            logger.info(f"模型 {name} 评估完成")
            
        except Exception as e:
            logger.error(f"评估模型 {name} 失败: {e}")
            evaluation_results[name] = {
                'success': False,
                'error': str(e)
            }
    
    return evaluation_results


def print_results(training_results, evaluation_results):
    """打印训练和评估结果"""
    logger.info("=" * 60)
    logger.info("BERT模型训练和评估结果汇总")
    logger.info("=" * 60)
    
    # 训练结果
    logger.info("\n训练结果:")
    for name, result in training_results.items():
        if result['success']:
            history = result['history']
            # 统一处理训练历史格式
            train_acc = None
            val_acc = None
            
            # 处理不同的历史格式
            if isinstance(history, dict):
                if 'train_accuracy' in history:
                    train_acc = history['train_accuracy']
                elif 'accuracy' in history:
                    train_acc = history['accuracy']
                
                if 'val_accuracy' in history and history['val_accuracy'] is not None:
                    val_acc = history['val_accuracy']
                elif 'validation_accuracy' in history:
                    val_acc = history['validation_accuracy']
            
            # 打印结果
            if train_acc is not None:
                logger.info(f"  {name}: 训练准确率 = {train_acc:.4f}")
            if val_acc is not None:
                logger.info(f"  {name}: 验证准确率 = {val_acc:.4f}")
        else:
            logger.info(f"  {name}: 训练失败 - {result['error']}")
    
    # 评估结果
    logger.info("\n评估结果:")
    for name, result in evaluation_results.items():
        if result['success']:
            metrics = result['metrics']
            logger.info(f"  {name}:")
            logger.info(f"    准确率: {metrics['accuracy']:.4f}")
            logger.info(f"    精确率: {metrics['precision']:.4f}")
            logger.info(f"    召回率: {metrics['recall']:.4f}")
            logger.info(f"    F1分数: {metrics['f1']:.4f}")
            logger.info(f"    ROC AUC: {metrics['roc_auc']:.4f}")
        else:
            logger.info(f"  {name}: 评估失败 - {result['error']}")
    
    # 找出最佳模型（基于验证集性能）
    best_model = None
    best_score = -1
    
    for name, result in training_results.items():
        if result['success']:
            history = result['history']
            score = None
            
            # 优先使用验证集准确率
            if isinstance(history, dict):
                if 'val_accuracy' in history and history['val_accuracy'] is not None:
                    score = history['val_accuracy']
                elif 'validation_accuracy' in history:
                    score = history['validation_accuracy']
                elif 'train_accuracy' in history:
                    score = history['train_accuracy']
                elif 'accuracy' in history:
                    score = history['accuracy']
            
            if score is not None and score > best_score:
                best_score = score
                best_model = name
    
    if best_model:
        logger.info(f"\n最佳模型: {best_model} (验证准确率: {best_score:.4f})")
    else:
        logger.info(f"\n无法确定最佳模型（缺少验证准确率）")


def save_models(models, processor, output_dir: str = "models"):
    """保存训练好的BERT模型和数据处理器"""
    logger.info("保存BERT模型...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # 保存数据处理器
    try:
        processor_path = output_path / "data_processor.pkl"
        processor.save_processor(processor_path)
        logger.info(f"数据处理器已保存到 {processor_path}")
    except Exception as e:
        logger.error(f"保存数据处理器失败: {e}")
    
    # 保存模型
    for name, model in models.items():
        if model.is_trained:
            try:
                model_path = output_path / f"{name}_model.pkl"
                model.save_model(model_path)
                logger.info(f"模型 {name} 已保存到 {model_path}")
            except Exception as e:
                logger.error(f"保存模型 {name} 失败: {e}")


def main():
    """主函数"""
    logger.info("开始BERT模型训练流程")
    
    # 1. 加载和预处理数据
    data = load_and_preprocess_data()
    
    # 2. 创建BERT模型
    models = create_models()
    
    if not models:
        logger.error("没有成功创建任何BERT模型")
        return
    
    # 3. 训练模型
    training_results = train_models(models, data)
    
    # 4. 评估模型
    evaluation_results = evaluate_models(models, data)
    
    # 5. 打印结果
    print_results(training_results, evaluation_results)
    
    # 6. 保存模型
    save_models(models, data['processor'])
    
    logger.info("BERT模型训练流程完成")


if __name__ == "__main__":
    main()
