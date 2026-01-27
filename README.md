# Aspect-Based Hate Speech Detection

This project implements an aspect-based hate speech detection system using a Transformer-based deep learning model. Instead of performing only binary toxic or non-toxic classification, the system predicts multiple hate-related aspects present in a single piece of text.

The task is formulated as a multi-label classification problem, where each comment can belong to more than one hate category at the same time. This allows the system to distinguish between different types of harmful content such as insults, obscenity, threats, and identity-based hate.

The model is based on RoBERTa, a Transformer architecture pretrained on large-scale text corpora. Input text is tokenized and passed through the Transformer encoder, followed by a classification head that outputs independent probabilities for each label using a sigmoid activation function. Binary Cross-Entropy with Logits loss is used during training to handle the multi-label nature of the task. Training and inference are performed using GPU acceleration with CUDA, along with mixed precision training and gradient clipping for efficiency and stability.

The dataset used for this project is the Jigsaw Toxic Comment Classification dataset, which contains approximately 160,000 user-generated comments. Each comment is annotated with six labels: toxic, severe_toxic, obscene, threat, insult, and identity_hate. Only the training portion of the dataset is used for model development and evaluation.

After training, the model achieves strong performance on the validation set, with a weighted F1-score of approximately 0.76, weighted precision of approximately 0.77, weighted recall of approximately 0.78, and a weighted ROC-AUC close to 0.99. The model performs particularly well on frequent labels such as toxic, obscene, and insult. Lower recall on rare labels such as threat and identity_hate is primarily due to class imbalance in the dataset rather than insufficient training.

The system supports inference on custom user input, producing per-label probabilities and identifying the primary toxic aspect when present. Clean text inputs are correctly classified as non-toxic, while harmful inputs are accurately flagged with appropriate aspect labels.

This project demonstrates the effectiveness of Transformer-based models for fine-grained hate speech detection and highlights the advantages of aspect-based analysis over simple binary toxicity classification.

Academic project on Aspect-Based Hate Speech Detection using Deep Learning and Transformers.
