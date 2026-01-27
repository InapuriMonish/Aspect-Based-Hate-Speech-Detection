import json
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, set_seed
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    jaccard_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============= LOGGING =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= CONFIG =============
@dataclass
class EvalConfig:
    """Configuration for evaluation pipeline."""
    data_path: str = "../data/train.csv"
    model_path: Optional[str] = None  # Will auto-discover if None
    model_dir: str = "../models"
    model_name: str = "roberta-base"
    output_dir: str = "../eval_results"
    max_len: int = 128
    batch_size: int = 64
    seed: int = 42
    val_split: float = 0.1
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    label_cols: List[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ])
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-discover model path if not provided
        if self.model_path is None:
            self.model_path = self._discover_model()
    
    def _discover_model(self) -> str:
        """Discover model checkpoint in model directory."""
        model_dir = Path(self.model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Search order: best_model.pt, roberta_aspect.pt, any .pt file
        candidates = [
            "best_model.pt",
            "roberta_aspect.pt",
            "model.pt",
        ]
        
        for candidate in candidates:
            path = model_dir / candidate
            if path.exists():
                logger.info(f"Auto-discovered model: {path}")
                return str(path)
        
        # Fallback: find any .pt file
        pt_files = list(model_dir.glob("*.pt"))
        if pt_files:
            discovered = str(pt_files[0])
            logger.warning(f"No standard model name found. Using: {discovered}")
            return discovered
        
        raise FileNotFoundError(
            f"No .pt model files found in {model_dir}. "
            f"Expected one of: {candidates}"
        )


# ============= DATASET =============
class InferenceDataset(torch.utils.data.Dataset):
    """Efficient dataset for inference."""
    
    def __init__(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        tokenizer=None,
        max_len: int = 128
    ):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        self.tokenizer = tokenizer
        self.max_len = max_len
        logger.info(f"Dataset created: {len(texts)} samples")

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.texts[idx]).strip()
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "text": text,
        }
        
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        
        return item


# ============= METRICS CALCULATOR =============
class ComprehensiveMetricsCalculator:
    """Calculate extensive metrics for multi-label classification."""
    
    @staticmethod
    def calculate_all(
        y_true: np.ndarray,
        y_proba: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        label_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict:
        """Calculate comprehensive metrics."""
        if y_pred is None:
            y_pred = (y_proba >= threshold).astype(int)
        
        n_labels = y_true.shape[1]
        
        overall_metrics = {
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "subset_accuracy": float(accuracy_score(y_true, y_pred)),
            "jaccard_score": float(jaccard_score(y_true, y_pred, average='samples')),
        }
        
        for avg_type in ["macro", "micro", "weighted"]:
            overall_metrics[f"f1_{avg_type}"] = float(
                f1_score(y_true, y_pred, average=avg_type, zero_division=0)
            )
            overall_metrics[f"precision_{avg_type}"] = float(
                precision_score(y_true, y_pred, average=avg_type, zero_division=0)
            )
            overall_metrics[f"recall_{avg_type}"] = float(
                recall_score(y_true, y_pred, average=avg_type, zero_division=0)
            )
        
        try:
            overall_metrics["roc_auc_macro"] = float(
                roc_auc_score(y_true, y_proba, average="macro", multi_class='ovr')
            )
            overall_metrics["roc_auc_weighted"] = float(
                roc_auc_score(y_true, y_proba, average="weighted", multi_class='ovr')
            )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            overall_metrics["roc_auc_macro"] = None
            overall_metrics["roc_auc_weighted"] = None
        
        per_label_metrics = {}
        
        if label_names is None:
            label_names = [f"label_{i}" for i in range(n_labels)]
        
        for i, label_name in enumerate(label_names):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            
            per_label_metrics[label_name] = {
                "f1": float(f1_score(y_true_i, y_pred_i, zero_division=0)),
                "precision": float(precision_score(y_true_i, y_pred_i, zero_division=0)),
                "recall": float(recall_score(y_true_i, y_pred_i, zero_division=0)),
                "support": int(y_true_i.sum()),
                "true_positives": int(((y_pred_i == 1) & (y_true_i == 1)).sum()),
                "false_positives": int(((y_pred_i == 1) & (y_true_i == 0)).sum()),
                "false_negatives": int(((y_pred_i == 0) & (y_true_i == 1)).sum()),
                "true_negatives": int(((y_pred_i == 0) & (y_true_i == 0)).sum()),
            }
            
            try:
                per_label_metrics[label_name]["roc_auc"] = float(
                    roc_auc_score(y_true_i, y_proba[:, i])
                )
            except:
                per_label_metrics[label_name]["roc_auc"] = None
        
        return {
            "overall": overall_metrics,
            "per_label": per_label_metrics,
            "threshold": threshold
        }


# ============= EVALUATOR =============
class ModelEvaluator:
    """Main evaluation class with comprehensive reporting."""
    
    def __init__(
        self,
        model,
        tokenizer,
        config: EvalConfig,
        device: torch.device
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        logger.info(f"Evaluator initialized on device: {device}")

    def predict_batch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, List[str]]:
        """Predict on batches efficiently."""
        self.model.eval()
        all_probs = []
        all_texts = []
        
        with torch.no_grad():
            loop = tqdm(dataloader, desc="Predicting", leave=False)
            
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_probs.append(probs)
                all_texts.extend(batch["text"])
        
        predictions = np.vstack(all_probs)
        logger.info(f"Generated predictions for {len(all_texts)} samples")
        
        return predictions, all_texts

    def evaluate(
        self,
        X_val: List[str],
        y_val: np.ndarray
    ) -> Dict:
        """Full evaluation with metrics and reporting."""
        logger.info(f"Evaluating on {len(X_val)} samples")
        
        dataset = InferenceDataset(
            X_val, y_val, self.tokenizer, self.config.max_len
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        y_proba, texts = self.predict_batch(dataloader)
        y_pred = (y_proba >= self.config.threshold).astype(int)
        
        logger.info("Calculating metrics...")
        metrics = ComprehensiveMetricsCalculator.calculate_all(
            y_val, y_proba, y_pred,
            self.config.label_cols,
            self.config.threshold
        )
        
        class_report = classification_report(
            y_val, y_pred,
            target_names=self.config.label_cols,
            output_dict=True,
            zero_division=0
        )
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "model_path": self.config.model_path,
                "model_name": self.config.model_name,
                "threshold": self.config.threshold,
                "max_len": self.config.max_len,
                "labels": self.config.label_cols
            },
            "metrics": metrics,
            "classification_report": class_report,
            "predictions": {
                "probabilities": y_proba.tolist(),
                "binary": y_pred.tolist(),
                "texts": texts
            }
        }
        
        return results

    def generate_reports(self, results: Dict) -> None:
        """Generate comprehensive text and visual reports."""
        output_dir = Path(self.config.output_dir)
        
        logger.info(f"Generating reports in {output_dir}")
        
        # Save metrics as JSON
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(results["metrics"], f, indent=2)
        logger.info(f"✓ Metrics saved to {metrics_file}")
        
        # Save classification report
        report_file = output_dir / "classification_report.json"
        with open(report_file, "w") as f:
            json.dump(results["classification_report"], f, indent=2)
        logger.info(f"✓ Classification report saved to {report_file}")
        
        # Text summary
        self._save_text_report(results, output_dir / "evaluation_summary.txt")
        
        # Visualizations
        try:
            self._plot_per_label_metrics(results, output_dir / "per_label_metrics.png")
            self._plot_confusion_matrices(results, output_dir / "confusion_matrices.png")
            self._plot_threshold_analysis(results, output_dir / "threshold_analysis.png")
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")

    def _save_text_report(self, results: Dict, filepath: Path) -> None:
        """Save human-readable report."""
        metrics = results["metrics"]
        
        with open(filepath, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL EVALUATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Timestamp: {results['timestamp']}\n")
            f.write(f"Model: {results['config']['model_path']}\n")
            f.write(f"Threshold: {metrics['threshold']}\n\n")
            
            f.write("OVERALL METRICS\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics["overall"].items():
                if value is not None:
                    f.write(f"{key:.<40} {value:.4f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("PER-LABEL METRICS\n")
            f.write("=" * 80 + "\n\n")
            
            for label, metrics_dict in metrics["per_label"].items():
                f.write(f"{label.upper()}\n")
                f.write("-" * 80 + "\n")
                for key, value in metrics_dict.items():
                    if isinstance(value, float):
                        f.write(f"  {key:.<35} {value:.4f}\n")
                    else:
                        f.write(f"  {key:.<35} {value}\n")
                f.write("\n")
        
        logger.info(f"✓ Text report saved to {filepath}")

    def _plot_per_label_metrics(self, results: Dict, filepath: Path) -> None:
        """Plot per-label F1, precision, recall."""
        metrics = results["metrics"]["per_label"]
        labels = list(metrics.keys())
        
        f1_scores = [metrics[l]["f1"] for l in labels]
        precisions = [metrics[l]["precision"] for l in labels]
        recalls = [metrics[l]["recall"] for l in labels]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(labels))
        width = 0.25
        
        ax.bar(x - width, f1_scores, width, label="F1", alpha=0.8)
        ax.bar(x, precisions, width, label="Precision", alpha=0.8)
        ax.bar(x + width, recalls, width, label="Recall", alpha=0.8)
        
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Per-Label Performance Metrics", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✓ Per-label metrics plot saved to {filepath}")

    def _plot_confusion_matrices(self, results: Dict, filepath: Path) -> None:
        """Plot confusion matrices for each label."""
        y_true = np.array(results["predictions"]["binary"])
        y_pred = np.array(results["predictions"]["binary"])
        
        n_labels = len(self.config.label_cols)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, label in enumerate(self.config.label_cols):
            cm = multilabel_confusion_matrix(y_true, y_pred)[i]
            
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues",
                ax=axes[i], cbar=False,
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"]
            )
            axes[i].set_title(f"{label}", fontweight="bold")
            axes[i].set_ylabel("True")
            axes[i].set_xlabel("Predicted")
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✓ Confusion matrices plot saved to {filepath}")

    def _plot_threshold_analysis(self, results: Dict, filepath: Path) -> None:
        """Analyze metrics across different thresholds."""
        y_true = np.array(results["predictions"]["binary"])
        y_proba = np.array(results["predictions"]["probabilities"])
        
        thresholds = np.linspace(0.1, 0.9, 20)
        f1_scores = []
        precisions = []
        recalls = []
        
        for thresh in thresholds:
            y_pred = (y_proba >= thresh).astype(int)
            f1_scores.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            precisions.append(precision_score(y_true, y_pred, average="weighted", zero_division=0))
            recalls.append(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(thresholds, f1_scores, marker="o", label="F1-Score", linewidth=2)
        ax.plot(thresholds, precisions, marker="s", label="Precision", linewidth=2)
        ax.plot(thresholds, recalls, marker="^", label="Recall", linewidth=2)
        
        ax.axvline(x=self.config.threshold, color="red", linestyle="--", 
                   label=f"Current ({self.config.threshold})")
        
        ax.set_xlabel("Threshold", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title("Threshold Impact on Performance Metrics", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"✓ Threshold analysis plot saved to {filepath}")


# ============= MAIN PIPELINE =============
def main(config: EvalConfig = None):
    """Main evaluation pipeline."""
    if config is None:
        config = EvalConfig()
    
    logger.info("=" * 80)
    logger.info("STARTING EVALUATION PIPELINE")
    logger.info("=" * 80)
    
    # Set seeds
    set_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device(config.device)
    logger.info(f"Device: {device}")
    logger.info(f"Model path: {config.model_path}")
    
    # Load data
    logger.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    texts = df["comment_text"].astype(str).values
    labels = df[config.label_cols].values
    
    logger.info(f"Dataset shape: {labels.shape}")
    logger.info(f"Class balance:\n{df[config.label_cols].sum()}")
    
    # Split data
    _, X_val, _, y_val = train_test_split(
        texts, labels,
        test_size=config.val_split,
        random_state=config.seed
    )
    
    logger.info(f"Validation set size: {len(X_val)}")
    
    # Load model and tokenizer
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    
    logger.info(f"Loading base model: {config.model_name}")
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.label_cols),
        problem_type="multi_label_classification"
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {config.model_path}")
    try:
        state_dict = torch.load(config.model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info("✓ Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise
    
    model = model.to(device)
    
    # Evaluate
    evaluator = ModelEvaluator(model, tokenizer, config, device)
    results = evaluator.evaluate(X_val, y_val)
    
    # Generate reports
    evaluator.generate_reports(results)
    
    # Save all results
    results_file = Path(config.output_dir) / "full_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"✓ Full results saved to {results_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    overall = results["metrics"]["overall"]
    print(f"F1 (Weighted):     {overall['f1_weighted']:.4f}")
    print(f"Precision (Weighted): {overall['precision_weighted']:.4f}")
    print(f"Recall (Weighted): {overall['recall_weighted']:.4f}")
    print(f"Hamming Loss:      {overall['hamming_loss']:.4f}")
    if overall.get('roc_auc_weighted'):
        print(f"ROC-AUC (Weighted): {overall['roc_auc_weighted']:.4f}")
    print("=" * 80)
    logger.info("Evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()