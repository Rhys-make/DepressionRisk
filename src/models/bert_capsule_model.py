"""
BERT-Capsule-Contrastive Learning模型
基于论文复现的抑郁风险预测模型
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch或Transformers未安装，BERT-Capsule模型不可用")

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class CapsuleLayer(nn.Module):
    """胶囊网络层"""
    
    def __init__(self, num_capsules: int, in_channels: int, out_channels: int, 
                 num_iterations: int = 3):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_iterations = num_iterations
        
        # 权重矩阵
        self.W = nn.Parameter(torch.randn(num_capsules, in_channels, out_channels))
        
    def forward(self, x):
        # x shape: (batch_size, in_channels)
        batch_size = x.size(0)
        
        # 扩展维度以匹配权重矩阵
        x = x.unsqueeze(1).unsqueeze(3)  # (batch_size, 1, in_channels, 1)
        x = x.expand(batch_size, self.num_capsules, self.in_channels, self.out_channels)
        
        # 计算预测向量
        u_hat = torch.sum(x * self.W, dim=2)  # (batch_size, num_capsules, out_channels)
        
        # 动态路由
        b = torch.zeros(batch_size, self.num_capsules, 1).to(x.device)
        
        for i in range(self.num_iterations):
            # 计算耦合系数
            c = F.softmax(b, dim=1)
            
            # 计算输出向量
            s = torch.sum(c * u_hat, dim=1, keepdim=True)  # (batch_size, 1, out_channels)
            v = self.squash(s)  # (batch_size, 1, out_channels)
            
            # 更新b
            if i < self.num_iterations - 1:
                delta_b = torch.sum(u_hat * v, dim=2, keepdim=True)
                b = b + delta_b
        
        return v.squeeze(1)  # (batch_size, out_channels)
    
    def squash(self, x):
        """Squashing函数"""
        norm = torch.norm(x, dim=2, keepdim=True)
        return (norm ** 2 / (1 + norm ** 2)) * (x / norm)


class SymptomCapsule(nn.Module):
    """症状胶囊网络"""
    
    def __init__(self, input_size: int, symptom_capsules: int = 9, 
                 capsule_dim: int = 16):
        super(SymptomCapsule, self).__init__()
        self.input_size = input_size
        self.symptom_capsules = symptom_capsules
        self.capsule_dim = capsule_dim
        
        # 症状胶囊层
        self.symptom_capsules_layer = CapsuleLayer(
            num_capsules=symptom_capsules,
            in_channels=input_size,
            out_channels=capsule_dim
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=capsule_dim,
            num_heads=4,
            batch_first=True
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        batch_size = x.size(0)
        
        # 生成症状胶囊
        symptom_embs = self.symptom_capsules_layer(x)  # (batch_size, capsule_dim)
        
        # 扩展为多个症状胶囊
        symptom_embs = symptom_embs.unsqueeze(1).expand(
            batch_size, self.symptom_capsules, self.capsule_dim
        )
        
        # 自注意力机制
        attended_embs, _ = self.attention(symptom_embs, symptom_embs, symptom_embs)
        
        return attended_embs  # (batch_size, symptom_capsules, capsule_dim)


class DepressionCapsule(nn.Module):
    """抑郁胶囊网络"""
    
    def __init__(self, input_size: int, num_classes: int = 2, 
                 capsule_dim: int = 16, num_iterations: int = 3):
        super(DepressionCapsule, self).__init__()
        self.input_size = input_size
        self.num_classes = num_classes
        self.capsule_dim = capsule_dim
        
        # 抑郁胶囊层
        self.depression_capsules = CapsuleLayer(
            num_capsules=num_classes,
            in_channels=input_size,
            out_channels=capsule_dim,
            num_iterations=num_iterations
        )
        
        # 分类层
        self.classifier = nn.Linear(capsule_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, input_size)
        capsule_output = self.depression_capsules(x)  # (batch_size, capsule_dim)
        logits = self.classifier(capsule_output)  # (batch_size, num_classes)
        return logits


class ContrastiveLoss(nn.Module):
    """对比学习损失"""
    
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features, labels):
        # 计算相似度矩阵
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签矩阵
        labels = labels.unsqueeze(0)
        label_matrix = (labels == labels.T).float()
        
        # 计算对比损失
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # 只考虑正样本对
        mean_log_prob = (label_matrix * log_prob).sum(dim=1) / label_matrix.sum(dim=1)
        loss = -mean_log_prob.mean()
        
        return loss


class BertCapsuleModel(nn.Module):
    """BERT-Capsule-Contrastive Learning模型"""
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", 
                 symptom_capsules: int = 9, capsule_dim: int = 16,
                 num_classes: int = 2, dropout: float = 0.1):
        super(BertCapsuleModel, self).__init__()
        
        # BERT编码器
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        
        # 冻结BERT参数（可选）
        for param in self.bert.parameters():
            param.requires_grad = False
        
        bert_output_dim = self.bert.config.hidden_size
        
        # 特征投影层
        self.feature_projection = nn.Linear(bert_output_dim, 768)
        
        # 症状胶囊网络
        self.symptom_capsule = SymptomCapsule(
            input_size=768,
            symptom_capsules=symptom_capsules,
            capsule_dim=capsule_dim
        )
        
        # 抑郁胶囊网络
        self.depression_capsule = DepressionCapsule(
            input_size=symptom_capsules * capsule_dim,
            num_classes=num_classes,
            capsule_dim=capsule_dim
        )
        
        # 对比学习损失
        self.contrastive_loss = ContrastiveLoss(temperature=0.1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT编码
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 处理不同类型的BERT输出
        if hasattr(bert_outputs, 'pooler_output') and bert_outputs.pooler_output is not None:
            # 标准BERT有pooler_output
            pooled_output = bert_outputs.pooler_output  # (batch_size, hidden_size)
        else:
            # DistilBERT等模型没有pooler_output，使用平均池化
            last_hidden_state = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
            # 使用attention_mask进行平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask  # (batch_size, hidden_size)
        
        # 特征投影
        features = self.feature_projection(pooled_output)
        features = self.dropout(features)
        
        # 症状胶囊
        symptom_embs = self.symptom_capsule(features)  # (batch_size, symptom_capsules, capsule_dim)
        
        # 展平症状胶囊
        flattened_symptoms = symptom_embs.reshape(symptom_embs.size(0), -1)
        
        # 抑郁胶囊
        logits = self.depression_capsule(flattened_symptoms)
        
        outputs = {'logits': logits, 'features': features, 'symptom_embs': symptom_embs}
        
        if labels is not None:
            # 分类损失
            classification_loss = F.cross_entropy(logits, labels)
            
            # 对比学习损失
            contrastive_loss = self.contrastive_loss(features, labels)
            
            # 总损失
            total_loss = classification_loss + 0.1 * contrastive_loss
            
            outputs['loss'] = total_loss
            outputs['classification_loss'] = classification_loss
            outputs['contrastive_loss'] = contrastive_loss
        
        return outputs


class BertCapsuleWrapper(BaseModel):
    """BERT-Capsule模型的包装器"""
    
    def __init__(self, bert_model_name: str = "bert-base-uncased", 
                 symptom_capsules: int = 9, capsule_dim: int = 16,
                 num_classes: int = 2, max_length: int = 512, **kwargs):
        """
        初始化BERT-Capsule模型
        
        Args:
            bert_model_name: BERT模型名称
            symptom_capsules: 症状胶囊数量
            capsule_dim: 胶囊维度
            num_classes: 类别数量
            max_length: 最大序列长度
            **kwargs: 其他参数
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch或Transformers未安装，无法使用BERT-Capsule模型")
        
        super().__init__(model_name="bert_capsule_model")
        
        self.bert_model_name = bert_model_name
        self.symptom_capsules = symptom_capsules
        self.capsule_dim = capsule_dim
        self.num_classes = num_classes
        self.max_length = max_length
        self.model_params = kwargs
        
        # 标记为文本模型
        self.uses_text = True
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化模型"""
        # 确保使用正确的BERT模型名称
        if 'distilbert' in self.bert_model_name.lower():
            # 对于DistilBERT，确保使用正确的模型名称
            if self.bert_model_name == 'distilbert-base-uncased':
                bert_model_name = 'distilbert-base-uncased'
            else:
                bert_model_name = self.bert_model_name
        else:
            # 对于标准BERT，使用标准名称
            bert_model_name = self.bert_model_name
        
        self.model = BertCapsuleModel(
            bert_model_name=bert_model_name,
            symptom_capsules=self.symptom_capsules,
            capsule_dim=self.capsule_dim,
            num_classes=self.num_classes,
            dropout=self.model_params.get('dropout', 0.1)
        )
        
        self.model.to(self.device)
        logger.info(f"初始化BERT-Capsule模型，设备: {self.device}")
    
    def _tokenize_texts(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """对文本进行分词"""
        tokenizer = self.model.tokenizer
        
        # 批量分词
        encoded = tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        return input_ids, attention_mask
    
    def train(self, texts: List[str], y_train: np.ndarray,
              val_texts: Optional[List[str]] = None,
              y_val: Optional[np.ndarray] = None,
              **kwargs) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            texts: 训练文本列表
            y_train: 训练标签
            val_texts: 验证文本列表
            y_val: 验证标签
            **kwargs: 训练参数
        """
        logger.info(f"开始训练BERT-Capsule模型")
        
        # 训练参数
        epochs = kwargs.get('epochs', 10)
        batch_size = kwargs.get('batch_size', 16)
        learning_rate = kwargs.get('learning_rate', 2e-5)
        early_stopping_patience = kwargs.get('early_stopping_patience', 5)
        
        # 准备数据
        train_input_ids, train_attention_mask = self._tokenize_texts(texts)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        train_dataset = TensorDataset(train_input_ids, train_attention_mask, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # 验证数据
        val_loader = None
        if val_texts is not None and y_val is not None:
            val_input_ids, val_attention_mask = self._tokenize_texts(val_texts)
            y_val_tensor = torch.LongTensor(y_val).to(self.device)
            val_dataset = TensorDataset(val_input_ids, val_attention_mask, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # 训练历史
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_input_ids, batch_attention_mask, batch_labels in train_loader:
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    labels=batch_labels
                )
                
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs['logits'], 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # 验证阶段
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_input_ids, batch_attention_mask, batch_labels in val_loader:
                        outputs = self.model(
                            input_ids=batch_input_ids,
                            attention_mask=batch_attention_mask,
                            labels=batch_labels
                        )
                        
                        val_loss += outputs['loss'].item()
                        _, predicted = torch.max(outputs['logits'], 1)
                        val_total += batch_labels.size(0)
                        val_correct += (predicted == batch_labels).sum().item()
                
                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    logger.info(f"早停在第 {epoch+1} 轮")
                    break
                
                self.model.train()
            
            # 打印进度
            if (epoch + 1) % 2 == 0:
                log_msg = f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}"
                if val_loader is not None:
                    log_msg += f", Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
                logger.info(log_msg)
        
        self.is_trained = True
        self.training_history = {
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'model_params': self.model_params
        }
        
        logger.info(f"BERT-Capsule模型训练完成")
        return self.training_history
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """预测"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        input_ids, attention_mask = self._tokenize_texts(texts)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            _, predicted = torch.max(outputs['logits'], 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """预测概率"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        input_ids, attention_mask = self._tokenize_texts(texts)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = F.softmax(outputs['logits'], dim=1)
        
        return probabilities.cpu().numpy()
    
    def save_model(self, file_path):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存PyTorch模型权重
        weights_path = file_path.with_suffix('.pt')
        torch.save(self.model.state_dict(), weights_path)
        
        # 保存模型元数据（训练历史、特征名等）
        meta_path = file_path.with_suffix('.meta.json')
        meta_data = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_params': getattr(self, 'model_params', {}),
            'bert_model_name': getattr(self, 'bert_model_name', 'distilbert-base-uncased'),
            'weights_path': str(weights_path)
        }
        
        import json
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        # 同时保存pickle格式以保持兼容性
        pkl_data = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_params': getattr(self, 'model_params', {}),
            'bert_model_name': getattr(self, 'bert_model_name', 'distilbert-base-uncased'),
            'weights_path': str(weights_path)
        }
        
        import pickle
        with open(file_path, 'wb') as f:
            pickle.dump(pkl_data, f)
        
        logger.info(f"BERT模型已保存到: {weights_path} 和 {meta_path} 和 {file_path}")
    
    def load_model(self, file_path):
        """加载模型"""
        file_path = Path(file_path)
        
        # 尝试加载.pt + .meta.json格式
        meta_path = file_path.with_suffix('.meta.json')
        weights_path = file_path.with_suffix('.pt')
        
        if meta_path.exists() and weights_path.exists():
            # 加载模型元数据
            import json
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            self.model_name = meta_data['model_name']
            self.is_trained = meta_data['is_trained']
            self.feature_names = meta_data.get('feature_names')
            self.training_history = meta_data.get('training_history', {})
            self.model_params = meta_data.get('model_params', {})
            self.bert_model_name = meta_data.get('bert_model_name', 'distilbert-base-uncased')
            
            # 重新初始化模型以确保结构一致
            self._initialize_model()
            
            # 加载PyTorch模型权重
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                # 尝试加载权重，如果失败则使用strict=False
                try:
                    self.model.load_state_dict(state_dict, strict=True)
                    logger.info(f"BERT模型已从 {weights_path} 加载（严格模式）")
                except Exception as e:
                    logger.warning(f"严格加载失败，尝试宽松加载: {e}")
                    # 尝试宽松加载，忽略不匹配的键
                    missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                    if missing_keys:
                        logger.warning(f"缺失的键: {missing_keys[:5]}...")  # 只显示前5个
                    if unexpected_keys:
                        logger.warning(f"意外的键: {unexpected_keys[:5]}...")  # 只显示前5个
                    logger.info(f"BERT模型已从 {weights_path} 加载（宽松模式）")
                
                self.is_trained = True
            except Exception as e:
                logger.warning(f"加载权重失败，可能需要重新训练: {e}")
                self.is_trained = False
            return
        
        # 尝试加载.pkl格式（兼容旧版本）
        if file_path.exists():
            import pickle
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # 从pickle数据中提取信息
            self.model_name = model_data.get('model_name', 'bert_capsule_model')
            self.is_trained = model_data.get('is_trained', False)
            self.training_history = model_data.get('training_history', {})
            self.feature_names = model_data.get('feature_names')
            self.model_params = model_data.get('model_params', {})
            self.bert_model_name = model_data.get('bert_model_name', 'distilbert-base-uncased')
            
            # 重新初始化模型以确保结构一致
            self._initialize_model()
            
            # 检查是否有权重文件路径
            weights_path = model_data.get('weights_path')
            if weights_path and Path(weights_path).exists():
                try:
                    # 加载PyTorch模型权重
                    state_dict = torch.load(weights_path, map_location=self.device)
                    # 尝试加载权重，如果失败则使用strict=False
                    try:
                        self.model.load_state_dict(state_dict, strict=True)
                        logger.info(f"BERT模型已从 {weights_path} 加载（严格模式）")
                    except Exception as e:
                        logger.warning(f"严格加载失败，尝试宽松加载: {e}")
                        # 尝试宽松加载，忽略不匹配的键
                        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
                        if missing_keys:
                            logger.warning(f"缺失的键: {missing_keys[:5]}...")  # 只显示前5个
                        if unexpected_keys:
                            logger.warning(f"意外的键: {unexpected_keys[:5]}...")  # 只显示前5个
                        logger.info(f"BERT模型已从 {weights_path} 加载（宽松模式）")
                    
                    self.is_trained = True
                except Exception as e:
                    logger.warning(f"加载权重失败，可能需要重新训练: {e}")
                    self.is_trained = False
            else:
                logger.warning("没有找到权重文件，模型可能需要重新训练")
            
            return
        
        raise FileNotFoundError(f"模型文件不存在: {file_path}")
