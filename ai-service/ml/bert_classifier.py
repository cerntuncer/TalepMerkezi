import os
from typing import List, Dict, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


class BertTextClassifier:
    """Thin wrapper around a Hugging Face sequence classification model.

    Expects a locally available fine-tuned model directory that contains
    a compatible config and weights for sequence classification.
    """

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._pipeline: Optional[TextClassificationPipeline] = None

    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self._tokenizer = tokenizer
        self._model = model
        self._pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

    def predict(self, text: str) -> Dict[str, object]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        outputs = self._pipeline(text)
        # HF returns a list per input; we pass single string -> outputs[0]
        scores = outputs[0]
        # Pick top label by score
        top = max(scores, key=lambda s: float(s["score"])) if scores else {"label": "", "score": 0.0}
        return {
            "label": str(top.get("label", "")),
            "confidence": float(top.get("score", 0.0)),
            "scores": {str(s["label"]): float(s["score"]) for s in scores},
        }


class ModelManager:
    """Manage lifecycle of a singleton BERT classifier.

    Environment variables:
      - SEQ_CLS_MODEL_DIR: absolute/relative path to fine-tuned model directory
                           default: /workspace/ai-service/models/bert
    """

    def __init__(self) -> None:
        model_dir = os.getenv("SEQ_CLS_MODEL_DIR", "/workspace/ai-service/models/bert")
        self.model_dir = model_dir
        self.classifier: Optional[BertTextClassifier] = None
        self._ensure_loaded_if_available()

    def _ensure_loaded_if_available(self) -> None:
        if not os.path.isdir(self.model_dir):
            return
        config_path = os.path.join(self.model_dir, "config.json")
        if not os.path.exists(config_path):
            return
        try:
            self.classifier = BertTextClassifier(self.model_dir)
            self.classifier.load()
        except Exception:
            # Leave classifier as None if loading fails; caller can handle gracefully
            self.classifier = None

    def ready(self) -> bool:
        return self.classifier is not None and self.classifier.is_loaded()

    def classify(self, text: str) -> Dict[str, object]:
        if not self.ready():
            return {"label": "", "confidence": 0.0, "scores": {}, "ok": False}
        result = self.classifier.predict(text)
        result["ok"] = True
        return result