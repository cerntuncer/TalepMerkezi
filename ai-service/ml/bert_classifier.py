# ai-service/ml/bert_classifier.py
import os
from typing import Dict, List, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
)


def _device_index() -> int:
    # GPU varsa 0, yoksa CPU (-1)
    return 0 if torch.cuda.is_available() else -1


class BertTextClassifier:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.device = _device_index()
        self.pipeline: TextClassificationPipeline | None = None

    def load(self) -> None:
        tok = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self.model_dir, local_files_only=True
        )
        self.pipeline = TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            device=self.device,               # GPU:0, CPU:-1
            return_all_scores=True,
            function_to_apply="softmax",
        )

    def ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, text: str) -> Tuple[str, float, List[Dict]]:
        if not self.ready():
            raise RuntimeError("Model not loaded")
        scores = self.pipeline(text, truncation=True, max_length=256)[0]
        best = max(scores, key=lambda x: x["score"])
        return best["label"], float(best["score"]), scores


class ModelManager:
    """
    FastAPI'nin kullanacağı ince sarmalayıcı.
    """
    def __init__(self, model_dir: str | None = None):
        self.model_dir = model_dir or os.getenv(
            "SEQ_CLS_MODEL_DIR",
            # Senin klasörün:
            "ai-service/models/model_tr_support2/model_tr_support"
        )
        self._clf: BertTextClassifier | None = None

    def ready(self) -> bool:
        return self._clf is not None and self._clf.ready()

    def load(self) -> None:
        self._clf = BertTextClassifier(self.model_dir)
        self._clf.load()

    def classify(self, text: str) -> Dict:
        if not self.ready():
            return {"ok": False, "label": "", "confidence": 0.0, "scores": []}
        label, conf, dist = self._clf.predict(text or "")
        return {"ok": True, "label": label, "confidence": conf, "scores": dist}
