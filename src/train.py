import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
    set_seed
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

# ============= LOGGING SETUP =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= CONFIG =============
@dataclass
class TrainingConfig:
    """Configuration for training pipeline."""
    data_path: str = "../data/train.csv"
    model_name: str = "roberta-base"
    output_dir: str = "../models"
    batch_size: int = 32  # Increased for better training
    max_len: int = 128
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    lr: float = 2e-5
    seed: int = 42
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    val_split: float = 0.1
    test_split: float = 0.1
    
    label_cols: List[str] = None
    
    def __post_init__(self):
        if self.label_cols is None:
            self.label_cols = [
                "toxic", "severe_toxic", "obscene",
                "threat", "insult", "identity_hate"
            ]
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


# ============= DATASET =============
class TextClassificationDataset(Dataset):
    """Efficient dataset for multi-label text classification."""
    
    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        tokenizer,
        max_len: int = 128
    ):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        logger.info(f"Dataset created with {len(texts)} samples")

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
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx]
        }


# ============= TRAINING UTILITIES =============
class MetricsCalculator:
    """Calculate comprehensive metrics for multi-label classification."""
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            y_proba: Prediction probabilities
            threshold: Classification threshold
        """
        y_pred_binary = (y_proba >= threshold).astype(int) if y_proba is not None else y_pred
        
        metrics = {
            "hamming_loss": hamming_loss(y_true, y_pred_binary),
            "f1_macro": f1_score(y_true, y_pred_binary, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred_binary, average="micro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred_binary, average="weighted", zero_division=0),
            "precision_macro": precision_score(y_true, y_pred_binary, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred_binary, average="macro", zero_division=0),
        }
        
        if y_proba is not None:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba, average="macro")
        
        return metrics


class Trainer:
    """Main training loop with checkpointing and early stopping."""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config: TrainingConfig,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config
        self.device = device
        
        self.best_val_metric = float('inf')
        self.best_model_path = Path(config.output_dir) / "best_model.pt"
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_metrics": []
        }
        
        logger.info(f"Trainer initialized on device: {device}")

    def train_epoch(self) -> float:
        """Train one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(self.train_loader, desc="Training", leave=False)
        
        for step, batch in enumerate(loop):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.criterion(outputs.logits, labels)
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate model and return loss + metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation", leave=False)
            
            for batch in loop:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        metrics = MetricsCalculator.calculate(all_labels, None, all_preds)
        
        return avg_loss, metrics

    def train(self) -> Dict:
        """Run full training loop."""
        logger.info("Starting training...")
        
        for epoch in range(self.config.epochs):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["val_loss"].append(val_loss)
            self.metrics_history["val_metrics"].append(val_metrics)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"F1 (weighted): {val_metrics['f1_weighted']:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_metric:
                self.best_val_metric = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"✓ Best model saved (val_loss: {val_loss:.4f})")
        
        logger.info("Training complete!")
        return self.metrics_history


# ============= MAIN PIPELINE =============
def main(config: TrainingConfig = None):
    """Main training pipeline."""
    if config is None:
        config = TrainingConfig()
    
    # Set seeds for reproducibility
    set_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Load data
    logger.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    texts = df["comment_text"].astype(str).values
    labels = df[config.label_cols].values
    
    logger.info(f"Dataset shape: {labels.shape}")
    logger.info(f"Class distribution:\n{df[config.label_cols].sum()}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        texts, labels,
        test_size=(config.test_split + config.val_split),
        random_state=config.seed
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=config.test_split / (config.test_split + config.val_split),
        random_state=config.seed
    )
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Load model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(config.label_cols),
        problem_type="multi_label_classification"
    ).to(device)
    
    # Create datasets and loaders
    train_dataset = TextClassificationDataset(X_train, y_train, tokenizer, config.max_len)
    val_dataset = TextClassificationDataset(X_val, y_val, tokenizer, config.max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    total_steps = len(train_loader) * config.epochs // config.gradient_accumulation_steps
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=1e-8
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    # Train
    trainer = Trainer(
        model, train_loader, val_loader,
        optimizer, scheduler, criterion,
        config, device
    )
    
    metrics_history = trainer.train()
    
    # Save config and metrics
    with open(Path(config.output_dir) / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)
    
    with open(Path(config.output_dir) / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)
    
    logger.info(f"Results saved to {config.output_dir}")


if __name__ == "__main__":
    main()