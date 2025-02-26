import json
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path
import re
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        # 基本配置
        self.model_type = config.get('model_type', 'hybrid-bert')
        self.bert_base_model = config.get('bert_base_model', 'bert-base-uncased')
        self.hidden_size = config.get('hidden_size', 768)
        self.n_classes = config.get('n_classes', 6)
        self.n_features = config.get('n_features', 46)
        self.combined_dim = config.get('combined_dim', 814)
        self.model_version = config.get('model_version', 'hybrid-bert-base-v1.0')
        self.max_length = config.get('max_length', 512)  # 添加 max_length 参数
        
        # 分类器配置
        classifier_config = config.get('classifier_config', {})
        self.hidden_layers = classifier_config.get('hidden_layers', [512, 256])
        self.dropout = classifier_config.get('dropout', 0.3)
        self.output_dim = classifier_config.get('output_dim', 6)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FeatureConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.label_dict = config.get('label_dict', {})
        self.emotion_words = config.get('emotion_words', {})

class ImprovedAdComplaintFeatures:
    def __init__(self, feature_config):
        self.label_dict = feature_config.label_dict
        self.emotion_words = feature_config.emotion_words

    # [Previous methods remain the same]
    def clean_text(self, text):
        """文本清理"""
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\s!?.,-:;\'"()]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_phrase_features(self, text, phrases):
        """提取短语特征"""
        count = 0
        for phrase in phrases:
            count += text.count(phrase)
        return count

    def extract_label_features(self, text):
        """提取标签相关特征"""
        text = self.clean_text(text)
        words = text.split()
        total_words = len(words) if words else 1

        features = {}
        
        for label, word_sets in self.label_dict.items():
            primary_count = sum(word in text for word in word_sets['primary'])
            phrase_count = self.extract_phrase_features(text, word_sets['phrases'])
            
            features.update({
                f'{label}_primary_count': primary_count,
                f'{label}_phrase_count': phrase_count,
                f'{label}_total_count': primary_count + phrase_count,
                f'{label}_density': (primary_count + phrase_count) / total_words,
                f'{label}_phrase_ratio': phrase_count / (primary_count + phrase_count + 1e-10)
            })

        return features

    def extract_emotion_features(self, text):
        """提取情感特征"""
        text = self.clean_text(text)
        words = text.split()
        total_words = len(words) if words else 1
        sentences = [s.strip() for s in text.split('.') if s.strip()]

        features = {}

        for emotion, words_list in self.emotion_words.items():
            count = sum(word in text for word in words_list)
            features[f'{emotion}_count'] = count
            features[f'{emotion}_ratio'] = count / total_words

        features.update({
            'exclamation_density': text.count('!') / len(sentences) if sentences else 0,
            'question_density': text.count('?') / len(sentences) if sentences else 0,
            'emphasis_punctuation': (text.count('!') + text.count('?')) / len(sentences) if sentences else 0,
            'caps_sentence_ratio': sum(1 for s in sentences if any(c.isupper() for c in s)) / len(sentences) if sentences else 0
        })

        return features

    def extract_structural_features(self, text):
        """提取结构特征"""
        text = str(text)
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()

        return {
            'sentence_count': len(sentences),
            'avg_sentence_length': np.mean([len(s.split()) for s in sentences]) if sentences else 0,
            'max_sentence_length': max([len(s.split()) for s in sentences]) if sentences else 0,
            'min_sentence_length': min([len(s.split()) for s in sentences]) if sentences else 0,
            'word_count': len(words),
            'unique_word_ratio': len(set(words)) / len(words) if words else 0,
            'comma_per_sentence': sum(s.count(',') for s in sentences) / len(sentences) if sentences else 0
        }

    def create_features_single_text(self, text):
        """为单个文本创建所有特征"""
        features = {}
        features.update(self.extract_label_features(text))
        features.update(self.extract_emotion_features(text))
        features.update(self.extract_structural_features(text))

        for i in range(6):
            features[f'keyword_match_{i}'] = 0

        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        return feature_vector

class HybridBERTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_base_model)
        
        # Load from config
        self.n_classes = config.n_classes
        self.combined_dim = config.combined_dim
        
        # 构建分类器层
        layers = []
        input_dim = self.combined_dim
        
        # 添加隐藏层
        for hidden_dim in config.hidden_layers:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = hidden_dim
        
        # 添加输出层
        layers.extend([
            nn.Linear(input_dim, config.output_dim),
            nn.Softmax(dim=1)
        ])
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        combined_features = torch.cat((pooled_output, features), dim=1)
        return self.classifier(combined_features)

class InferenceEngine:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.config = ModelConfig(self.model_dir / 'config.json')
        self.feature_config = FeatureConfig(self.model_dir / 'feature_config.json')
        
        # Initialize components
        self.tokenizer = self._load_tokenizer()
        self.feature_engineer = ImprovedAdComplaintFeatures(self.feature_config)
        self.model = self._load_model()
        self.scaler = self._initialize_scaler()
        
    def _load_tokenizer(self):
        """Load tokenizer from local files"""
        tokenizer_files = {
            'vocab_file': str(self.model_dir / 'vocab.txt'),
            'special_tokens_map_file': str(self.model_dir / 'special_tokens_map.json'),
            'tokenizer_config_file': str(self.model_dir / 'tokenizer_config.json')
        }
        return BertTokenizer.from_pretrained(str(self.model_dir), **tokenizer_files)
        
    def _load_model(self):
        model = HybridBERTModel(self.config)
        model.load_state_dict(torch.load(self.model_dir / 'pytorch_model.pt', 
                                       map_location=self.config.device))
        model.to(self.config.device)
        model.eval()
        return model
        
    def _initialize_scaler(self):
        scaler = StandardScaler()
        sample_features = self.feature_engineer.create_features_single_text("Sample text")
        scaler.fit(sample_features.reshape(1, -1))
        return scaler
        
    def analyze_text(self, text):
        try:
            # Create features
            features = self.feature_engineer.create_features_single_text(text)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=self.config.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.config.device)
            attention_mask = encoding['attention_mask'].to(self.config.device)
            features = torch.FloatTensor(features_scaled).to(self.config.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, features)
                predictions = outputs.cpu().numpy()[0]
            
            # Process results
            category_mapping = {
                0: "misleading",
                1: "social responsibility",
                2: "placement",
                3: "children issues",
                4: "taste and decency",
                5: "other"
            }
            
            top_indices = np.argsort(predictions)[-4:][::-1]
            analysis_results = [
                {
                    "category": category_mapping[idx],
                    "confidence": float(predictions[idx])
                }
                for idx in top_indices
            ]
            
            return {
                "request_id": str(uuid.uuid4()),
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "data": {
                    "analysis_results": analysis_results,
                    "metadata": {
                        "model_version": self.config.model_version,
                        "language": "eng",
                        "word_count": len(text.split())
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {
                "request_id": str(uuid.uuid4()),
                "status": "error",
                "error": str(e)
            }

# Usage example
if __name__ == "__main__":
    MODEL_DIR = "model_files"
    
    # Initialize inference engine
    engine = InferenceEngine(MODEL_DIR)
    
    # Test text
    test_text = "Seen at the Coromandel Ketic Fair on 2 January..."
    
    # Get analysis results
    result = engine.analyze_text(test_text)
    print(result)