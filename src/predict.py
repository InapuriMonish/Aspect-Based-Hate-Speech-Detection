import json
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, set_seed
from pydantic import BaseModel, Field, field_validator

# ============= LOGGING =============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============= CONFIG =============
@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    model_path: Optional[str] = None  # Auto-discover if None
    model_dir: str = "../models"
    model_name: str = "roberta-base"
    max_len: int = 128
    batch_size: int = 32
    threshold: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    enable_cache: bool = True
    cache_size: int = 1000
    
    label_cols: List[str] = field(default_factory=lambda: [
        "toxic", "severe_toxic", "obscene",
        "threat", "insult", "identity_hate"
    ])
    
    def __post_init__(self):
        """Auto-discover model if not provided."""
        if self.model_path is None:
            self.model_path = self._discover_model()
    
    def _discover_model(self) -> str:
        """Discover model checkpoint in model directory."""
        model_dir = Path(self.model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Search order: best_model.pt, roberta_aspect.pt, model.pt
        candidates = ["best_model.pt", "roberta_aspect.pt", "model.pt"]
        
        for candidate in candidates:
            path = model_dir / candidate
            if path.exists():
                logger.info(f"✓ Auto-discovered model: {path}")
                return str(path)
        
        # Fallback: find any .pt file
        pt_files = list(model_dir.glob("*.pt"))
        if pt_files:
            discovered = str(pt_files[0])
            logger.warning(f"No standard model name found. Using: {discovered}")
            return discovered
        
        raise FileNotFoundError(
            f"No .pt model files found in {model_dir}. "
            f"Expected one of: {candidates}\n"
            f"Available files: {list(model_dir.iterdir())}"
        )


# ============= PYDANTIC MODELS (PYDANTIC V2) =============
class PredictionRequest(BaseModel):
    """Input validation for prediction requests."""
    text: str = Field(..., min_length=1, max_length=10000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_probs: bool = True
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()


class PredictionResult(BaseModel):
    """Output format for predictions."""
    text: str
    predictions: Dict[str, float]
    binary_predictions: Dict[str, int]
    max_label: str
    max_confidence: float
    all_above_threshold: List[str]
    inference_time_ms: float
    model_version: str = "roberta-base"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str] = Field(..., min_length=1, max_length=1000)
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    return_probs: bool = True


class BatchPredictionResult(BaseModel):
    """Batch prediction results."""
    predictions: List[PredictionResult]
    total_time_ms: float
    num_samples: int


# ============= MODEL WRAPPER =============
class TextClassificationModel:
    """Thread-safe model wrapper with caching and batching."""
    
    _instance = None
    
    def __new__(cls, config: InferenceConfig = None):
        """Singleton pattern for single model instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config: InferenceConfig = None):
        """Initialize model (only once due to singleton)."""
        if self._initialized:
            logger.info("Using cached model instance")
            return
        
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        logger.info("=" * 80)
        logger.info("INITIALIZING MODEL")
        logger.info("=" * 80)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
        logger.info("✓ Tokenizer loaded")
        
        # Load base model
        logger.info(f"Loading base model: {self.config.model_name}")
        self.model = RobertaForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.config.label_cols),
            problem_type="multi_label_classification"
        )
        logger.info("✓ Base model loaded")
        
        # Load checkpoint
        logger.info(f"Loading checkpoint from: {self.config.model_path}")
        
        if not Path(self.config.model_path).exists():
            raise FileNotFoundError(
                f"Model checkpoint not found: {self.config.model_path}\n"
                f"Please ensure the file exists."
            )
        
        try:
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            logger.info("✓ Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
        
        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"✓ Model moved to device: {self.device}")
        
        # Optimization (torch.compile requires Triton, skip if not available)
        try:
            # Only compile if on newer PyTorch with full support
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model, backend="eager")
                logger.info("✓ Model compiled with torch.compile (eager backend)")
        except Exception as e:
            logger.debug(f"torch.compile skipped: {e}")
        
        # Cache for repeated predictions
        self._prediction_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"Labels: {self.config.label_cols}")
        logger.info(f"Cache enabled: {self.config.enable_cache}")
        logger.info("=" * 80)
        
        self._initialized = True
    
    @torch.no_grad()
    def predict_single(
        self,
        text: str,
        threshold: Optional[float] = None
    ) -> Dict:
        """Predict on single text with caching."""
        threshold = threshold or self.config.threshold
        
        # Check cache
        cache_key = f"{text}_{threshold}"
        if self.config.enable_cache and cache_key in self._prediction_cache:
            self._cache_hits += 1
            return self._prediction_cache[cache_key]
        
        self._cache_misses += 1
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_len,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict with mixed precision
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
            outputs = self.model(**encoding)
        
        probs = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()
        
        # Handle single label case (output is scalar)
        if probs.ndim == 0:
            probs = np.array([probs.item()])
        
        result = {
            "probs": dict(zip(self.config.label_cols, probs.astype(float))),
            "binary": {
                label: int(prob >= threshold)
                for label, prob in zip(self.config.label_cols, probs)
            },
            "max_label": self.config.label_cols[int(np.argmax(probs))],
            "max_confidence": float(np.max(probs))
        }
        
        # Cache result
        if self.config.enable_cache:
            if len(self._prediction_cache) >= self.config.cache_size:
                self._prediction_cache.clear()
                logger.debug("Cache cleared (size limit reached)")
            self._prediction_cache[cache_key] = result
        
        return result
    
    @torch.no_grad()
    def predict_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """Predict on batch of texts efficiently."""
        threshold = threshold or self.config.threshold
        
        # Tokenize batch
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.config.max_len,
            return_tensors="pt"
        ).to(self.device)
        
        # Predict with mixed precision
        with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu'):
            outputs = self.model(**encodings)
        
        probs = torch.sigmoid(outputs.logits).cpu().numpy()
        
        results = []
        for i, text in enumerate(texts):
            result = {
                "probs": dict(zip(self.config.label_cols, probs[i].astype(float))),
                "binary": {
                    label: int(prob >= threshold)
                    for label, prob in zip(self.config.label_cols, probs[i])
                },
                "max_label": self.config.label_cols[int(np.argmax(probs[i]))],
                "max_confidence": float(np.max(probs[i]))
            }
            results.append(result)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = (self._cache_hits / total * 100) if total > 0 else 0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_size": len(self._prediction_cache)
        }
    
    def clear_cache(self):
        """Clear prediction cache."""
        self._prediction_cache.clear()
        logger.info("✓ Prediction cache cleared")


# ============= INFERENCE ENGINE =============
class InferenceEngine:
    """High-level inference API with validation and formatting."""
    
    def __init__(self, config: InferenceConfig = None):
        self.config = config or InferenceConfig()
        self.model = TextClassificationModel(self.config)
    
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """Single text prediction with validation and timing."""
        start_time = time.time()
        
        threshold = request.threshold or self.config.threshold
        result = self.model.predict_single(request.text, threshold)
        
        inference_time_ms = (time.time() - start_time) * 1000
        
        return PredictionResult(
            text=request.text[:100],
            predictions=result["probs"],
            binary_predictions=result["binary"],
            max_label=result["max_label"],
            max_confidence=result["max_confidence"],
            all_above_threshold=[
                label for label, prob in result["probs"].items()
                if prob >= threshold
            ],
            inference_time_ms=round(inference_time_ms, 2)
        )
    
    def predict_batch(self, request: BatchPredictionRequest) -> BatchPredictionResult:
        """Batch prediction with optimal batching."""
        start_time = time.time()
        threshold = request.threshold or self.config.threshold
        
        results = []
        for i in range(0, len(request.texts), self.config.batch_size):
            batch_texts = request.texts[i:i + self.config.batch_size]
            batch_results = self.model.predict_batch(batch_texts, threshold)
            
            for text, result in zip(batch_texts, batch_results):
                results.append(
                    PredictionResult(
                        text=text[:100],
                        predictions=result["probs"],
                        binary_predictions=result["binary"],
                        max_label=result["max_label"],
                        max_confidence=result["max_confidence"],
                        all_above_threshold=[
                            label for label, prob in result["probs"].items()
                            if prob >= threshold
                        ],
                        inference_time_ms=0
                    )
                )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResult(
            predictions=results,
            total_time_ms=round(total_time_ms, 2),
            num_samples=len(request.texts)
        )
    
    def get_model_info(self) -> Dict:
        """Get model information and cache stats."""
        return {
            "model_name": self.config.model_name,
            "model_path": self.config.model_path,
            "device": str(self.config.device),
            "labels": self.config.label_cols,
            "default_threshold": self.config.threshold,
            "cache_enabled": self.config.enable_cache,
            "cache_stats": self.model.get_cache_stats()
        }


# ============= INTERACTIVE CLI =============
class InteractiveCLI:
    """Command-line interface for inference."""
    
    def __init__(self, config: InferenceConfig = None):
        self.engine = InferenceEngine(config)
        self.config = config or InferenceConfig()
    
    def run(self):
        """Main interactive loop."""
        self._print_welcome()
        
        while True:
            try:
                user_input = input("\n➜ Enter text (or command): ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == "quit":
                    print("\n👋 Exiting...")
                    break
                
                if user_input.lower() == "info":
                    self._print_info()
                    continue
                
                if user_input.lower() == "batch":
                    self._batch_mode()
                    continue
                
                if user_input.lower() == "clear":
                    self.engine.model.clear_cache()
                    continue
                
                if user_input.lower() == "help":
                    self._print_help()
                    continue
                
                # Single prediction
                try:
                    request = PredictionRequest(text=user_input)
                    result = self.engine.predict(request)
                    self._print_result(result)
                except ValueError as e:
                    print(f"❌ Input error: {e}\n")
            
            except KeyboardInterrupt:
                print("\n\n👋 Exiting...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"❌ Error: {e}\n")
    
    def _print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 80)
        print(" " * 15 + "TOXIC COMMENT CLASSIFICATION - INTERACTIVE MODE")
        print("=" * 80)
        print("Commands: 'info' | 'batch' | 'clear' | 'help' | 'quit'\n")
    
    def _print_help(self):
        """Print help information."""
        print("\n" + "-" * 80)
        print("COMMANDS:")
        print("-" * 80)
        print("  'info'    - Show model information and cache statistics")
        print("  'batch'   - Batch prediction mode (multiple texts)")
        print("  'clear'   - Clear prediction cache")
        print("  'help'    - Show this help message")
        print("  'quit'    - Exit the application")
        print("-" * 80 + "\n")
    
    def _print_result(self, result: PredictionResult):
        """Format and print single result."""
        print("\n" + "-" * 80)
        print(f"📝 Text: {result.text}...")
        print(f"⏱️  Inference Time: {result.inference_time_ms:.2f}ms")
        print("-" * 80)
        
        print("📊 Probabilities:")
        for label, prob in sorted(
            result.predictions.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar = "█" * int(prob * 20)
            status = "🔴" if prob >= self.config.threshold else "⚪"
            print(f"  {status} {label:.<20} {prob:.4f} {bar}")
        
        if result.all_above_threshold:
            print(f"\n⚠️  FLAGGED: {', '.join(result.all_above_threshold)}")
        else:
            print(f"\n✅ CLEAN")
        
        print(f"🎯 Primary: {result.max_label} ({result.max_confidence:.4f})")
        print()
    
    def _batch_mode(self):
        """Batch prediction mode."""
        print("\n📋 Enter texts (one per line, empty line to finish):")
        texts = []
        
        while True:
            text = input().strip()
            if not text:
                break
            texts.append(text)
        
        if not texts:
            print("No texts provided.\n")
            return
        
        try:
            request = BatchPredictionRequest(texts=texts)
            results = self.engine.predict_batch(request)
            
            print(f"\n✓ Processed {results.num_samples} texts in {results.total_time_ms:.2f}ms")
            print(f"  Average: {results.total_time_ms / results.num_samples:.2f}ms per text\n")
            
            for i, result in enumerate(results.predictions, 1):
                flags = f" [{', '.join(result.all_above_threshold)}]" if result.all_above_threshold else " [CLEAN]"
                print(f"{i}. {result.text}... {flags}")
                print(f"   {result.max_label}: {result.max_confidence:.4f}\n")
        
        except ValueError as e:
            print(f"❌ Error: {e}\n")
    
    def _print_info(self):
        """Print model info and cache stats."""
        info = self.engine.get_model_info()
        
        print("\n" + "=" * 80)
        print("ℹ️  MODEL INFORMATION")
        print("=" * 80)
        print(f"Model Name:        {info['model_name']}")
        print(f"Model Path:        {info['model_path']}")
        print(f"Device:            {info['device']}")
        print(f"Default Threshold: {info['default_threshold']}")
        print(f"Labels ({len(info['labels'])}):")
        for label in info['labels']:
            print(f"  • {label}")
        
        print("\n📦 Cache Statistics:")
        stats = info['cache_stats']
        print(f"  Enabled:       {info['cache_enabled']}")
        print(f"  Hits:          {stats['cache_hits']}")
        print(f"  Misses:        {stats['cache_misses']}")
        print(f"  Hit Rate:      {stats['hit_rate_percent']}%")
        print(f"  Cached Items:  {stats['cache_size']}")
        print("=" * 80 + "\n")


# ============= MAIN =============
def main():
    """Entry point with flexible usage."""
    try:
        set_seed(InferenceConfig().seed)
        
        config = InferenceConfig()
        
        # Interactive mode
        cli = InteractiveCLI(config)
        cli.run()
    
    except FileNotFoundError as e:
        logger.error(f"Initialization failed: {e}")
        print(f"\n❌ Error: {e}\n")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Error: {e}\n")
        exit(1)


if __name__ == "__main__":
    main()