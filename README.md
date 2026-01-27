#Aspect-Based Hate Speech Detection using Transformers

This project implements an **aspect-based hate speech detection system** using a Transformer-based deep learning model.  
Unlike binary toxic/non-toxic classification, this system predicts **multiple hate-related aspects simultaneously**.

The model is trained and evaluated on the **Jigsaw Toxic Comment Classification Dataset** and supports **GPU-accelerated training and inference**.

---

## 🔍 Problem Statement
Online moderation systems must identify not only whether text is harmful, but **what kind of harm it contains**.  
A single comment may include insults, obscenity, threats, or identity-based hate.

This project addresses this by performing **multi-label classification**, where each comment can belong to multiple hate aspects.

---

## 🧠 Model & Approach

- **Model**: RoBERTa (Transformer-based)
- **Task**: Multi-label text classification
- **Loss Function**: Binary Cross-Entropy with Logits
- **Activation**: Sigmoid (per-label probability)
- **Training**: GPU-accelerated (CUDA)
- **Optimization**:
  - Mixed Precision Training (AMP)
  - Gradient Clipping
  - Validation-based checkpointing

---

## 📊 Dataset

**Jigsaw Toxic Comment Classification Dataset**
- ~160,000 user comments
- Aspect labels:
  - `toxic`
  - `severe_toxic`
  - `obscene`
  - `threat`
  - `insult`
  - `identity_hate`

Only `train.csv` is used.

---

## 📁 Repository Structure

hate-speech-aspect-detection/
│
├── data/
│ └── train.csv
│
├── eval_results/
│ ├── classification_report.json
│ ├── confusion_matrices.png
│ ├── evaluation_summary.txt
│ ├── full_results.json
│ ├── metrics.json
│ ├── per_label_metrics.png
│ └── threshold_analysis.png
│
├── models/
│ ├── best_model.pt
│ ├── config.json
│ └── metrics.json
│
├── src/
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│ └── predict.py
│
├── .gitignore
├── README.md
└── requirements.txt


---

## ⚙️ Environment Setup

### Recommended
- Python **3.9 – 3.11**
- CUDA-enabled GPU (tested on RTX 3060)
- Conda or virtual environment

### Install dependencies
```bash
pip install -r requirements.txt
▶️ Run Order
1. (Optional) Data sanity check
python src/preprocess.py
2. Train the model
python src/train.py
Best model saved to models/best_model.pt

3. Evaluate performance
python src/evaluate.py
Metrics and plots saved in eval_results/

4. Run inference (demo)
python src/predict.py
📈 Results Summary
F1-score (Weighted): ~0.76

Precision (Weighted): ~0.77

Recall (Weighted): ~0.78

ROC-AUC (Weighted): ~0.99

Hamming Loss: ~0.016

High-frequency labels (toxic, obscene, insult) show strong performance.
Lower recall on rare labels (threat, identity_hate) is primarily due to class imbalance, not under-training.

🎯 Key Observations
The model accurately distinguishes profanity from identity-based hate

Clean inputs are not falsely flagged

Threshold tuning confirms optimal performance near 0.5

Additional epochs show diminishing returns

🚀 Future Work
Class-imbalance handling (Focal Loss, reweighting)

Class-wise threshold tuning

Multilingual hate speech detection

Speech-to-text hate detection

Web deployment (Streamlit / Flask)