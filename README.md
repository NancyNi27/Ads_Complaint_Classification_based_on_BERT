# 🧠 Hybrid-BERT Advertising Complaint Classification (NZASA Project)

> **Keywords:** NLP • BERT • Feature Engineering • Data Augmentation • Advertising Regulation • Deep Learning

---

## 📘 Overview

This project introduces an **automated complaint classification platform** for the **New Zealand Advertising Standards Authority (NZASA)**.
It integrates **domain-specific handcrafted features** with a **Hybrid-BERT deep learning architecture** to categorize advertising complaints across six categories.

The model achieved **81.37% accuracy**, outperforming traditional machine-learning baselines (SVM, Random Forest).
It demonstrates how **AI can enhance regulatory efficiency**, consistency, and transparency in real-world compliance systems.

---

## 🧩 System Architecture

```
Frontend:  React + Material UI + Recharts + React-WordCloud  
Backend:   FastAPI + Python-jose + Uvicorn + Node.js  
Database:  MongoDB (document-based storage)  
Modeling:  PyTorch + Scikit-learn + NLTK + Pandas
Deployment: Flask (API) + Gunicorn + Docker + AWS EC2
```

* **Frontend**: Complaint submission + dashboard for citizens and regulators
* **Backend**: RESTful API with NLP & LLM pipelines (Hybrid-BERT + ChatGPT API)
* **Data Layer**: Stores complaint metadata, ad content, and classification results
* **Cloud Deployment**: Containerized (Docker) with AWS scalability and monitoring

---

## 🧮 Methodology

1. **Data Acquisition** – Scraped ASA decisions with paginated POST requests + retry logic
2. **Cleaning & Preprocessing** – Regex extraction, emoji/OCR noise removal, UTF-8 normalization
3. **Feature Engineering** – TF-IDF, keyword frequencies, sentiment markers, structural metrics
4. **Hybrid-BERT Model**

   * Base: `bert-base-uncased` (768-dim embeddings)
   * Fusion: Concatenate BERT features + handcrafted domain features
   * Classifier: Multi-layer dense network with ReLU + Dropout regularization
5. **Augmentation Strategies**

   * ChatGPT synthetic complaint generation
   * WordNet synonym replacement (15%)
   * SMOTE oversampling for minority classes

---

## 📊 Key Results

| Metric                  | Best Score  | Notes                                              |
| :---------------------- | :---------- | :------------------------------------------------- |
| **Accuracy**            | **81.37 %** | 5-fold cross-validation                            |
| **F1-Score (Weighted)** | 0.808       | High for *Misleading* & *Taste/Decency* categories |
| **Architecture**        | Hybrid-BERT | Combines semantic + handcrafted features           |
| **Deployment**          | AWS EC2     | Dockerized Flask API with Gunicorn                 |

> ✅ The Hybrid-BERT approach outperformed SVM and Random Forest baselines while remaining interpretable and extensible.

---

## 💡 Learnings & Reflections

* Data quality and preprocessing are **critical foundations** — poor data undermines even the best models.
* Cross-disciplinary understanding (data science + backend + product design) improves practical impact.
* Real-world AI systems must balance **technical accuracy**, **cost efficiency**, and **user-side transparency**.

---

## 🚀 Future Work

* **Lightweight architectures:** DistilBERT / ALBERT for faster inference
* **Multi-label classification:** Handle overlapping complaint categories
* **Generative augmentation:** GPT-3/4-based realistic text synthesis
* **Knowledge integration:** Fuse with advertising codes and industry guidelines
