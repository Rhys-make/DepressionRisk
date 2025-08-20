#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型训练主脚本
训练和比较不同类型的抑郁风险预测模型
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.preprocessor import DataProcessor
from models.model_factory import ModelFactory, ModelManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def load_and_preprocess_data(data_path: str = "data/raw/precise_training_data.csv"):
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
    
    # 创建训练测试分割
    train_data, test_data = processor.create_train_test_split(processed_data, test_size=0.2)
    
    # 准备特征和标签
    X_train, y_train = processor.prepare_features(train_data, label_column='label', fit_scaler=True)
    X_test, y_test = processor.prepare_features(test_data, label_column='label', fit_scaler=False)
    
    # 获取文本数据（用于BERT模型）
    train_texts = train_data['text'].tolist()
    test_texts = test_data['text'].tolist()
    
    logger.info(f"数据预处理完成:")
    logger.info(f"  训练集: {X_train.shape}")
    logger.info(f"  测试集: {X_test.shape}")
    logger.info(f"  特征数量: {X_train.shape[1]}")
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'train_texts': train_texts,
        'test_texts': test_texts,
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
    """创建要训练的模型"""
    logger.info("创建模型...")
    
    model_configs = [
        # 传统机器学习模型
        {
            'name': 'random_forest',
            'model_type': 'random_forest',
            'model_category': 'traditional',
            'n_estimators': 100,
            'max_depth': 10
        },
        {
            'name': 'svm',
            'model_type': 'svm',
            'model_category': 'traditional',
            'C': 1.0,
            'kernel': 'rbf'
        },
        {
            'name': 'logistic_regression',
            'model_type': 'logistic_regression',
            'model_category': 'traditional',
            'C': 1.0
        },
        {
            'name': 'gradient_boosting',
            'model_type': 'gradient_boosting',
            'model_category': 'traditional',
            'n_estimators': 100,
            'learning_rate': 0.1
        },
        # BERT-Capsule模型（如果可用）
        {
            'name': 'bert_capsule',
            'model_type': 'bert_capsule',
            'model_category': 'deep',
            'bert_model_name': 'bert-base-uncased',
            'symptom_capsules': 9,
            'capsule_dim': 16,
            'max_length': 256
        }
    ]
    
    models = {}
    for config in model_configs:
        try:
            model = ModelFactory.create_model(
                model_type=config['model_type'],
                model_category=config['model_category'],
                **{k: v for k, v in config.items() if k not in ['name', 'model_type', 'model_category']}
            )
            models[config['name']] = model
            logger.info(f"成功创建模型: {config['name']}")
        except Exception as e:
            logger.warning(f"创建模型 {config['name']} 失败: {e}")
    
    return models


def train_models(models, data):
    """训练所有模型"""
    logger.info("开始训练模型...")
    
    training_results = {}
    
    for name, model in models.items():
        logger.info(f"训练模型: {name}")
        try:
            # 检查模型类型，决定训练方式
            if hasattr(model, 'bert_model_name'):  # BERT模型
                history = model.train(
                    texts=data['train_texts'],
                    y_train=data['y_train'],
                    val_texts=data['test_texts'],
                    y_val=data['y_test'],
                    epochs=5,
                    batch_size=8,
                    learning_rate=2e-5
                )
            else:  # 传统模型
                history = model.train(
                    X_train=data['X_train'],
                    y_train=data['y_train'],
                    X_val=data['X_test'],
                    y_val=data['y_test']
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
    """评估所有模型"""
    logger.info("开始评估模型...")
    
    evaluation_results = {}
    
    for name, model in models.items():
        if not model.is_trained:
            logger.warning(f"模型 {name} 尚未训练，跳过评估")
            continue
        
        logger.info(f"评估模型: {name}")
        try:
            # 检查模型类型，决定评估方式
            if hasattr(model, 'bert_model_name'):  # BERT模型
                metrics = model.evaluate(data['test_texts'], data['y_test'])
            else:  # 传统模型
                metrics = model.evaluate(data['X_test'], data['y_test'])
            
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
    logger.info("训练和评估结果汇总")
    logger.info("=" * 60)
    
    # 训练结果
    logger.info("\n训练结果:")
    for name, result in training_results.items():
        if result['success']:
            history = result['history']
            if 'train_accuracy' in history:
                logger.info(f"  {name}: 训练准确率 = {history['train_accuracy']:.4f}")
            if 'val_accuracy' in history and history['val_accuracy'] is not None:
                logger.info(f"  {name}: 验证准确率 = {history['val_accuracy']:.4f}")
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
    
    # 找出最佳模型
    best_model = None
    best_score = -1
    
    for name, result in evaluation_results.items():
        if result['success']:
            score = result['metrics']['accuracy']
            if score > best_score:
                best_score = score
                best_model = name
    
    if best_model:
        logger.info(f"\n最佳模型: {best_model} (准确率: {best_score:.4f})")


def save_models(models, processor, output_dir: str = "models"):
    """保存训练好的模型和数据处理器"""
    logger.info("保存模型...")
    
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
                # 若模型尚未记录特征名，则从处理器补齐
                if getattr(model, 'feature_names', None) in (None, []):
                    model.feature_names = processor.feature_columns or processor.feature_names
                model_path = output_path / f"{name}_model.pkl"
                model.save_model(model_path)
                logger.info(f"模型 {name} 已保存到 {model_path}")
            except Exception as e:
                logger.error(f"保存模型 {name} 失败: {e}")


def main():
    """主函数"""
    logger.info("开始模型训练流程")
    
    # 1. 加载和预处理数据
    data = load_and_preprocess_data()
    
    # 2. 创建模型
    models = create_models()
    
    if not models:
        logger.error("没有成功创建任何模型")
        return
    
    # 3. 训练模型
    training_results = train_models(models, data)
    
    # 4. 评估模型
    evaluation_results = evaluate_models(models, data)
    
    # 5. 打印结果
    print_results(training_results, evaluation_results)
    
    # 6. 保存模型
    save_models(models, data['processor'])
    
    logger.info("模型训练流程完成")


if __name__ == "__main__":
    main()
