import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.preprocessing import StandardScaler
import logging
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from datetime import datetime
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedAdComplaintFeatures:
    def __init__(self):
        # 标签词典
        self.label_dict = {
            'misleading': {
                'primary': [
                    'misleading', 'false', 'incorrect', 'inaccurate', 'untrue',
                    'deceptive', 'misrepresent', 'exaggerate', 'misleads',
                    'unsubstantiated', 'wrong', 'dishonest', 'no proof'
                ],
                'phrases': [
                    'not true', 'false claim', 'misleading information',
                    'wrong information', 'cannot be substantiated'
                ]
            },
            'social_responsibility': {
                'primary': [
                    'unsafe', 'dangerous', 'harmful', 'irresponsible', 'hazard',
                    'risk', 'safety', 'health', 'alcohol', 'gambling'
                ],
                'phrases': [
                    'social responsibility', 'public safety', 'health risk',
                    'safety concern', 'unsafe practice'
                ]
            },
            'placement': {
                'primary': [
                    'location', 'place', 'position', 'display', 'billboard',
                    'visible', 'screen', 'site', 'area', 'distance'
                ],
                'phrases': [
                    'near school', 'close to', 'in front of', 'next to',
                    'wrong place'
                ]
            },
            'children': {
                'primary': [
                    'child', 'children', 'kid', 'minor', 'young', 'youth',
                    'teen', 'teenage', 'student', 'school', 'parent'
                ],
                'phrases': [
                    'target children', 'appeal to children', 'child safety',
                    'protect children', 'school area'
                ]
            },
            'taste_decency': {
                'primary': [
                    'offensive', 'inappropriate', 'vulgar', 'explicit', 'sexual',
                    'violent', 'disturbing', 'graphic', 'crude', 'tasteless'
                ],
                'phrases': [
                    'sexually suggestive', 'offensive content', 'bad taste',
                    'inappropriate content', 'adult content'
                ]
            }
        }

        # 情感词典
        self.emotion_words = {
            'strong_negative': [
                'very', 'extremely', 'absolutely', 'totally', 'completely',
                'highly', 'seriously', 'strongly', 'deeply', 'gravely'
            ],
            'concern': [
                'worry', 'concern', 'afraid', 'fear', 'alarming',
                'dangerous', 'risky', 'threat', 'problem', 'issue'
            ]
        }

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
        
        # 合并所有特征
        features.update(self.extract_label_features(text))
        features.update(self.extract_emotion_features(text))
        features.update(self.extract_structural_features(text))

        # 添加关键词匹配结果
        for i in range(6):
            features[f'keyword_match_{i}'] = 0  # 默认为0

        # 将字典转换为向量
        feature_names = sorted(features.keys())
        feature_vector = np.array([features[name] for name in feature_names])
        
        return feature_vector

class HybridBERTModel(nn.Module):
    def __init__(self, n_classes=6, n_features=None, bert_model='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        
        # BERT输出维度 (768) + 手工特征维度
        combined_dim = self.bert.config.hidden_size + n_features

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, input_ids, attention_mask, features):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output
        combined_features = torch.cat((pooled_output, features), dim=1)
        return self.classifier(combined_features)

class SingleTextDataset(Dataset):
    def __init__(self, text, features, tokenizer, max_length=512):
        self.text = [text]  # Wrap single text in list
        self.features = features
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.FloatTensor(self.features[idx])
        }

def load_models(model_paths, device):
    """加载所有模型"""
    models = []
    for path in model_paths:
        try:
            state_dict = torch.load(path, map_location=device)
            n_features = state_dict['classifier.0.weight'].shape[1] - 768
            model = HybridBERTModel(n_classes=6, n_features=n_features)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            models.append(model)
            logger.info(f"Successfully loaded model from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {str(e)}")
            raise
    return models

def analyze_single_text(text, models, tokenizer, feature_engineer, scaler, device='cuda'):
    """分析单个文本"""
    try:
        # 创建特征
        features = feature_engineer.create_features_single_text(text)
        features_scaled = scaler.transform(features.reshape(1, -1))

        # 创建数据集和加载器
        dataset = SingleTextDataset(text, features_scaled, tokenizer)
        dataloader = DataLoader(dataset, batch_size=1)

        # 获取所有模型的预测
        all_predictions = []
        for model in models:
            with torch.no_grad():
                batch = next(iter(dataloader))
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                
                outputs = model(input_ids, attention_mask, features)
                predictions = outputs.cpu().numpy()
                all_predictions.append(predictions)

        # 平均所有模型的预测
        final_prediction = np.mean(all_predictions, axis=0)[0]

        # 创建结果
        category_mapping = {
            0: "misleading",
            1: "social responsibility",
            2: "placement",
            3: "children issues",
            4: "taste and decency",
            5: "other"
        }

        # 获取前4个最高置信度的预测
        top_indices = np.argsort(final_prediction)[-4:][::-1]
        analysis_results = [
            {
                "category": category_mapping[idx],
                "confidence": float(final_prediction[idx])
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
                    "model_version": "bert-base-uncased-v1.0",
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

def initialize_model(model_paths):
    """初始化所有必要的组件"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型
    models = load_models(model_paths, device)
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # 初始化特征工程组件
    feature_engineer = ImprovedAdComplaintFeatures()
    
    # 初始化scaler
    scaler = StandardScaler()
    # 创建一个样例特征以适配scaler
    sample_features = feature_engineer.create_features_single_text("Sample text")
    scaler.fit(sample_features.reshape(1, -1))
    
    return models, tokenizer, feature_engineer, scaler, device

# 使用示例
if __name__ == "__main__":
    # 设置模型路径
    model_paths = [
        'best_model/best_hybrid_model_fold_5.pt'
    ]
    
    # 初始化模型和组件
    models, tokenizer, feature_engineer, scaler, device = initialize_model(model_paths)
    
    # 测试文本
    test_text = "Seen at the Coromandel Ketic Fair on 2 January. I was gobsmacked with the |utter non -scientific rubbish. As a Medical Laboratory Scientist| I do have some knowledge in |this field. Th is Naturopath could be endangering lives| and a lot of people seemed to be |interested in what he was doing. This may be more than against ASA| but also illegal. More |info on his website www.bem.nz"
    # 获取分析结果
    result = analyze_single_text(test_text, models, tokenizer, feature_engineer, scaler, device)
    print(result)