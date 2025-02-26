import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report
import logging
import re
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

logging.basicConfig(level=logging.INFO)

class ImprovedAdComplaintFeatures:
    def __init__(self):
        # 重新设计的标签词典
        self.label_dict = {
            'misleading': {  # 标签1
                'primary': [
                    'misleading', 'false', 'incorrect', 'inaccurate', 'untrue',
                    'deceptive', 'misrepresent', 'exaggerate', 'misleads',
                    'unsubstantiated', 'wrong', 'dishonest', 'no proof',
                    'no evidence', 'claim', 'unclear', 'confusing'
                ],
                'phrases': [
                    'not true', 'false claim', 'misleading information',
                    'wrong information', 'cannot be substantiated',
                    'no scientific evidence', 'misleading comparison',
                    'not factual', 'factually incorrect'
                ]
            },
            'social_responsibility': {  # 标签2
                'primary': [
                    'unsafe', 'dangerous', 'harmful', 'irresponsible', 'hazard',
                    'risk', 'safety', 'health', 'alcohol', 'gambling', 'tobacco',
                    'addiction', 'accident', 'injury', 'public', 'society'
                ],
                'phrases': [
                    'social responsibility', 'public safety', 'health risk',
                    'safety concern', 'unsafe practice', 'dangerous behavior',
                    'public health', 'community concern', 'harmful effect',
                    'risk to public'
                ]
            },
            'placement': {  # 标签3
                'primary': [
                    'location', 'place', 'position', 'display', 'billboard',
                    'visible', 'screen', 'site', 'area', 'distance', 'near',
                    'close', 'proximity', 'street', 'road', 'outside'
                ],
                'phrases': [
                    'near school', 'close to', 'in front of', 'next to',
                    'wrong place', 'inappropriate location', 'public space',
                    'residential area', 'too close to', 'placement issue',
                    'visible from'
                ]
            },
            'children': {  # 标签4
                'primary': [
                    'child', 'children', 'kid', 'minor', 'young', 'youth',
                    'teen', 'teenage', 'student', 'school', 'parent', 'family',
                    'playground', 'juvenile', 'underage', 'baby', 'infant'
                ],
                'phrases': [
                    'target children', 'appeal to children', 'child safety',
                    'protect children', 'school area', 'young people',
                    'young audience', 'children content', 'kids zone',
                    'family viewing'
                ]
            },
            'taste_decency': {  # 标签5
                'primary': [
                    'offensive', 'inappropriate', 'vulgar', 'explicit', 'sexual',
                    'violent', 'disturbing', 'graphic', 'crude', 'tasteless',
                    'indecent', 'obscene', 'inappropriate', 'distasteful',
                    'provocative', 'uncomfortable'
                ],
                'phrases': [
                    'sexually suggestive', 'offensive content', 'bad taste',
                    'inappropriate content', 'adult content', 'mature content',
                    'explicit material', 'offensive language', 'disturbing image',
                    'sexual innuendo'
                ]
            }
        }

        # 标签映射词典 - 就在label_dict后面添加
        self.label_mapping = {
            'misleading': '1',
            'social_responsibility': '2',
            'placement': '3',
            'children': '4',
            'taste_decency': '5'
        }

        # 情感标记词典 (原有的代码)
        self.emotion_words = {
            'strong_negative': [
                'very', 'extremely', 'absolutely', 'totally', 'completely',
                'highly', 'seriously', 'strongly', 'deeply', 'gravely',
                'outrageous', 'unacceptable', 'inappropriate', 'disgusting'
            ],
            'concern': [
                'worry', 'concern', 'afraid', 'fear', 'alarming',
                'dangerous', 'risky', 'threat', 'problem', 'issue',
                'serious', 'severe', 'significant'
            ]
        }

    def clean_text(self, text):
        """改进的文本清理，保留更多有意义的标点和格式"""
        text = str(text).lower()
        # 保留更多标点符号和格式特征
        text = re.sub(r'[^a-z0-9\s!?.,-:;\'"()]', ' ', text)
        # 规范化空白字符
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_phrase_features(self, text, phrases):
        """提取短语特征"""
        count = 0
        for phrase in phrases:
            count += text.count(phrase)
        return count
    def direct_keyword_classification(self, text):
        """
        直接基于关键词进行分类的方法
        返回包含标签的列表
        """
        text = self.clean_text(text)
        matched_labels = set()

        # 检查每个类别的关键词
        for category, word_sets in self.label_dict.items():
            # 检查主要关键词
            if any(word in text for word in word_sets['primary']):
                matched_labels.add(self.label_mapping[category])
                continue

            # 检查短语
            if any(phrase in text for phrase in word_sets['phrases']):
                matched_labels.add(self.label_mapping[category])

        # 如果没有匹配到任何标签，返回'0'（其他类别）
        return list(matched_labels) if matched_labels else ['0']

    def test_classification(self, text):
        """
        测试文本的关键词匹配结果
        """
        text = self.clean_text(text)
        results = {}

        for category, word_sets in self.label_dict.items():
            # 找到匹配的关键词
            matched_primary = [word for word in word_sets['primary'] if word in text]
            matched_phrases = [phrase for phrase in word_sets['phrases'] if phrase in text]

            if matched_primary or matched_phrases:
                results[self.label_mapping[category]] = {
                    'matched_keywords': matched_primary,
                    'matched_phrases': matched_phrases
                }

        return results

    def test_keyword_matching(self, text):
        """测试关键词匹配功能的便捷方法"""
        # 获取关键词分类结果
        labels = self.direct_keyword_classification(text)
        print("分类标签:", labels)

        # 获取详细匹配信息
        matches = self.test_classification(text)
        print("匹配详情:", matches)

        return labels, matches

    def extract_label_features(self, text):
        """提取标签相关特征"""
        text = self.clean_text(text)
        words = text.split()
        total_words = len(words) if words else 1

        features = {}

        for label, word_sets in self.label_dict.items():
            # 单词级别特征
            primary_count = sum(word in text for word in word_sets['primary'])
            # 短语级别特征
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

        # 情感词特征
        for emotion, words_list in self.emotion_words.items():
            count = sum(word in text for word in words_list)
            features[f'{emotion}_count'] = count
            features[f'{emotion}_ratio'] = count / total_words

        # 标点符号特征
        features.update({
            'exclamation_density': text.count('!') / len(sentences) if sentences else 0,
            'question_density': text.count('?') / len(sentences) if sentences else 0,
            'emphasis_punctuation': (text.count('!') + text.count('?') + text.count('!!')) / len(sentences) if sentences else 0
        })

        # 大写字母特征（从原始文本提取）
        original_sentences = str(text).split('.')
        features['caps_sentence_ratio'] = sum(1 for s in original_sentences if any(c.isupper() for c in s)) / len(sentences) if sentences else 0

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

    def create_features(self, df, text_column='complaints'):
        """生成所有特征"""
        logging.info("开始生成改进后的特征...")

        all_features = []
        for text in df[text_column]:
            features = {}

            # 合并所有特征
            features.update(self.extract_label_features(text))
            features.update(self.extract_emotion_features(text))
            features.update(self.extract_structural_features(text))

            # 添加关键词匹配结果作为独热编码特征
            keyword_matches = self.direct_keyword_classification(text)
            # 为每个可能的标签（0-5）创建一个二进制特征
            for i in range(6):  # 包括0标签
                features[f'keyword_match_{i}'] = 1 if str(i) in keyword_matches else 0

            all_features.append(features)

        features_df = pd.DataFrame(all_features)
        logging.info(f"特征生成完成。总特征数: {features_df.shape[1]}")

        return features_df

    def analyze_features(self, features_df, labels, top_n=20):
        """分析特征重要性"""
        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(features_df, labels)

        importance = pd.Series(
            rf.feature_importances_,
            index=features_df.columns
        ).sort_values(ascending=False)

        return {
            'top_features': importance.head(top_n),
            'feature_correlations': features_df.corr()
        }

class HybridDataset(Dataset):
    def __init__(self, texts, features, labels, tokenizer, max_length=512):
        self.texts = texts
        self.features = features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'features': torch.FloatTensor(self.features[idx]),
            'labels': torch.FloatTensor(self.labels[idx])
        }

class HybridBERTModel(nn.Module):
    def __init__(self, n_classes, n_features, bert_model='bert-base-uncased'):
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
            nn.Softmax(dim=1)  # 改为 Softmax，因为是单标签分类
        )

    def forward(self, input_ids, attention_mask, features):
        # BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output

        # 连接BERT输出和手工特征
        combined_features = torch.cat((pooled_output, features), dim=1)

        # 分类
        return self.classifier(combined_features)
import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
import numpy as np

class TextAugmenter:
    def __init__(self, num_augmentations=2, p_synonym=0.3, p_back_translation=0.3):
        """
        Initialize TextAugmenter with configurable parameters.

        Args:
            num_augmentations (int): Number of augmented versions to generate per text
            p_synonym (float): Probability of applying synonym replacement
            p_back_translation (float): Probability of applying back translation
        """
        self.num_augmentations = num_augmentations
        self.p_synonym = p_synonym
        self.p_back_translation = p_back_translation

        # NLTK resources are downloaded at script startup
        pass

    def get_synonyms(self, word):
        """Get synonyms for a word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word and '_' not in lemma.name():
                    synonyms.add(lemma.name())
        return list(synonyms)

    def synonym_replacement(self, text):
        """Replace random words with their synonyms."""
        words = word_tokenize(text)
        num_to_replace = max(1, int(len(words) * 0.15))  # Replace up to 15% of words

        # Get indices of words that have synonyms
        valid_indices = []
        for i, word in enumerate(words):
            if len(self.get_synonyms(word)) > 0:
                valid_indices.append(i)

        # Randomly select indices to replace
        if valid_indices:
            indices_to_replace = random.sample(valid_indices,
                                            min(num_to_replace, len(valid_indices)))

            for idx in indices_to_replace:
                synonyms = self.get_synonyms(words[idx])
                if synonyms:
                    words[idx] = random.choice(synonyms)

        return ' '.join(words)

    def word_deletion(self, text):
        """Randomly delete words from the text."""
        words = word_tokenize(text)
        if len(words) <= 3:  # Don't delete from very short texts
            return text

        num_to_delete = max(1, int(len(words) * 0.1))  # Delete up to 10% of words
        indices_to_delete = random.sample(range(len(words)), num_to_delete)

        return ' '.join([word for i, word in enumerate(words)
                        if i not in indices_to_delete])

    def augment(self, text):
        """Generate multiple augmented versions of the input text."""
        augmented_texts = []

        for _ in range(self.num_augmentations):
            aug_text = text

            # Apply synonym replacement
            if random.random() < self.p_synonym:
                aug_text = self.synonym_replacement(aug_text)

            # Apply word deletion
            if random.random() < 0.2:  # 20% chance of word deletion
                aug_text = self.word_deletion(aug_text)

            if aug_text != text:  # Only add if the text was actually modified
                augmented_texts.append(aug_text)

        return augmented_texts

def train_model(model, train_loader, val_loader, device, train_labels, epochs=5):
    # 使用 CrossEntropyLoss，适用于单标签分类
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)
            labels = batch['labels'].to(device)

            # 获取标签的索引（因为是单标签分类）
            labels = torch.argmax(labels, dim=1)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask, features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        logging.info(f'Epoch {epoch+1}:')
        logging.info(f'Average training loss: {avg_train_loss:.4f}')
        logging.info(f'Average validation loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_hybrid_model.pt')

def prepare_labels(df):
    """
    将标签转换为单标签格式（只取第一个数字）
    参数：
        df: 包含 'label' 列的 DataFrame
    返回：
        label_matrix: one-hot编码的标签矩阵
        all_labels: 标签列表
    """
    # 提取第一个数字作为标签
    labels = df['label'].apply(lambda x: int(x.split(',')[0]))

    # 获取唯一标签值
    all_labels = sorted(labels.unique())

    # 创建 one-hot 编码矩阵
    label_matrix = np.zeros((len(df), len(all_labels)))
    for i, label in enumerate(labels):
        label_matrix[i, all_labels.index(label)] = 1

    return label_matrix, all_labels

def get_label_weights(label_matrix):
    pos_weights = len(label_matrix) / (np.sum(label_matrix, axis=0) + 1e-5)
    return torch.FloatTensor(pos_weights)



from sklearn.model_selection import KFold

def main():
    # 读取数据
    df = pd.read_csv('/content/drive/MyDrive/combination_excel_withoutnull_labeled.csv')

    # 只保留需要的列并处理标签
    df = df[['complaint_id', 'complaints', 'label']]

    # 提取特征
    feature_engineer = ImprovedAdComplaintFeatures()
    features_df = feature_engineer.create_features(df)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # 准备标签（单标签）
    labels, all_labels = prepare_labels(df)

    # 初始化 5-fold 交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # 对每个fold进行训练和评估
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df['complaints'].values)):
        logging.info(f"\n开始训练 Fold {fold + 1}/5")

        # 分割数据
        X_train, X_val = df['complaints'].values[train_idx], df['complaints'].values[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        f_train, f_val = features_scaled[train_idx], features_scaled[val_idx]

        # 初始化 text augmenter
        augmenter = TextAugmenter()

        # 数据增强
        logging.info(f"Fold {fold + 1}: 正在进行数据增强...")
        X_train_aug, y_train_aug, f_train_aug = augment_training_data(
            X_train, y_train, f_train, augmenter
        )

        # 初始化tokenizer和数据加载器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = HybridDataset(X_train_aug, f_train_aug, y_train_aug, tokenizer)
        val_dataset = HybridDataset(X_val, f_val, y_val, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # 初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridBERTModel(
            n_classes=len(all_labels),
            n_features=features_scaled.shape[1]
        ).to(device)

        # 训练模型
        train_model(model, train_loader, val_loader, device, y_train_aug)

        # 评估当前fold的模型
        logging.info(f"评估 Fold {fold + 1} 的结果:")
        fold_result = evaluate_model(model, val_loader, device)
        fold_results.append(fold_result)

        # 保存每个fold的模型
        torch.save(model.state_dict(), f'best_hybrid_model_fold_{fold+1}.pt')

    # 计算并输出平均结果
    logging.info("\n所有Fold的平均结果:")
    # 这里需要修改evaluate_model函数来返回具体的指标值，以便计算平均值

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            features = batch['features'].to(device)

            outputs = model(input_ids, attention_mask, features)
            preds = torch.argmax(outputs, dim=1)
            labels = torch.argmax(batch['labels'], dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算评估指标
    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        output_dict=True  # 返回字典格式，便于后续计算平均值
    )

    # 打印当前fold的结果
    print("\n分类报告:")
    print(classification_report(all_labels, all_preds, digits=4))

    return report
def augment_training_data(X_train, y_train, f_train, augmenter):
    """
    使用文本增强技术来增强训练数据

    参数：
        X_train: 原始训练文本
        y_train: 原始训练标签（one-hot编码）
        f_train: 原始训练特征
        augmenter: TextAugmenter实例

    返回：
        元组 (augmented_texts, augmented_labels, augmented_features)
    """
    augmented_texts = []
    augmented_labels = []
    augmented_features = []

    # 首先添加原始数据
    augmented_texts.extend(X_train)
    augmented_labels.extend(y_train)
    augmented_features.extend(f_train)

    # 然后添加增强版本
    for idx, (text, label, features) in enumerate(zip(X_train, y_train, f_train)):
        # 获取文本的增强版本
        aug_versions = augmenter.augment(text)

        # 为每个增强版本添加相应的标签和特征
        for aug_text in aug_versions:
            augmented_texts.append(aug_text)
            augmented_labels.append(label)  # 使用原始标签
            augmented_features.append(features)  # 使用原始特征

    # 转换为numpy数组
    return (
        np.array(augmented_texts),
        np.array(augmented_labels),
        np.array(augmented_features)
    )

def main():
    # 读取数据
    df = pd.read_csv('/content/drive/MyDrive/combination_excel_withoutnull_labeled.csv')

    # 只保留需要的列并处理标签
    df = df[['complaint_id', 'complaints', 'label']]

    # 提取特征
    feature_engineer = ImprovedAdComplaintFeatures()
    features_df = feature_engineer.create_features(df)

    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)

    # 准备标签（单标签）
    labels, all_labels = prepare_labels(df)

    # 初始化 5-fold 交叉验证
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    # 对每个fold进行训练和评估
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df['complaints'].values)):
        logging.info(f"\n开始训练 Fold {fold + 1}/5")

        # 分割数据
        X_train, X_val = df['complaints'].values[train_idx], df['complaints'].values[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        f_train, f_val = features_scaled[train_idx], features_scaled[val_idx]

        # 初始化 text augmenter
        augmenter = TextAugmenter()

        # 数据增强
        logging.info(f"Fold {fold + 1}: 正在进行数据增强...")
        X_train_aug, y_train_aug, f_train_aug = augment_training_data(
            X_train, y_train, f_train, augmenter
        )

        # 初始化tokenizer和数据加载器
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        train_dataset = HybridDataset(X_train_aug, f_train_aug, y_train_aug, tokenizer)
        val_dataset = HybridDataset(X_val, f_val, y_val, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16)

        # 初始化模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HybridBERTModel(
            n_classes=len(all_labels),
            n_features=features_scaled.shape[1]
        ).to(device)

        # 训练模型
        train_model(model, train_loader, val_loader, device, y_train_aug)

        # 评估当前fold的模型
        logging.info(f"评估 Fold {fold + 1} 的结果:")
        fold_result = evaluate_model(model, val_loader, device)
        fold_results.append(fold_result)

        # 保存每个fold的模型
        torch.save(model.state_dict(), f'best_hybrid_model_fold_{fold+1}.pt')

    # 计算并输出平均结果
    logging.info("\n所有Fold的平均结果:")


if __name__ == "__main__":
    main()