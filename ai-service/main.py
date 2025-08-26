from fastapi import FastAPI
from pydantic import BaseModel
import re
import unicodedata

from ml.bert_classifier import ModelManager

app = FastAPI(title="AI Classifier", version="0.5.0")

class ClassifyReq(BaseModel):
    text: str

class ClassifyResp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str


ml_manager = ModelManager()


def normalize_text(text: str) -> str:
    """Lowercase, strip diacritics and collapse whitespace for robust matching."""
    t = (text or "").lower().strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"\s+", " ", t)
    return t


def contains_keyword(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            return True
    return False


def contains_phrase(text: str, phrases: list[str]) -> bool:
    for ph in phrases:
        if ph in text:
            return True
    return False


@app.get("/ping")
def ping():
    return {"ok": True}


@app.post("/classify", response_model=ClassifyResp)
def classify(req: ClassifyReq):
    orig = (req.text or "")
    t = orig.lower()
    tn = normalize_text(orig)

    # Phrase-based rules (more specific → higher confidence)
    phrase_rules = [
        ( [
            "islem yapamiyorum", "islem basarisiz", "islem hatasi", "islem gerceklesmiyor", "islem olmuyor",
            "calismiyor", "baglanamiyorum", "uygulama cokuyor", "hata veriyor", "error"
          ], "TeknikDestek", 0.84),
        ( ["parolami unuttum", "sifre sifirla", "sifre yenile", "hesap kilitlendi"], "SifreIslemleri", 0.87),
        ( ["uyeligi iptal", "uyeligimi iptal", "abonelik iptal", "hesabi kapat"], "Iptal", 0.83),
        ( ["siparis vermek", "urun satin almak", "yeni urun", "stok var mi"], "UrunTalebi", 0.78),
        ( ["fatura odeme", "odeme basarisiz", "e-fatura", "tutar cok yuksek", "faturam"], "Fatura", 0.82),
        ( ["garanti kapsami", "garanti kapsaminda", "iade etmek", "onarim"], "Garanti", 0.72),
    ]

    for phrases, label, conf in phrase_rules:
        if contains_phrase(tn, phrases):
            return ClassifyResp(ok=True, label=label, confidence=conf, model="rule-v2")

    # Keyword-based rules (single tokens)
    keyword_rules = [
        (["sifre", "parola", "unut", "reset", "giris", "kilitlendi", "acilmiyor"], "SifreIslemleri", 0.85),
        (["fatura", "e-fatura", "irsaliye", "belge", "tutar", "ucret"], "Fatura", 0.82),
        (["iptal", "fesih", "dondur", "uyelik bitir", "abonelik"], "Iptal", 0.80),
        (["baglanmiyor", "coktu", "hata", "error", "donuyor", "takiliyor", "yavasladi"], "TeknikDestek", 0.78),
        (["siparis", "urun", "stok", "almak istiyorum"], "UrunTalebi", 0.75),
        (["garanti", "iade", "bozuldu", "onarim", "servis"], "Garanti", 0.70),
    ]

    for keywords, label, conf in keyword_rules:
        if contains_keyword(tn, keywords) or contains_keyword(t, keywords):
            return ClassifyResp(ok=True, label=label, confidence=conf, model="rule-v2")

    # Fallback: keep lower confidence to indicate uncertainty
    return ClassifyResp(ok=True, label="GenelTalep", confidence=0.60, model="rule-v2")


@app.post("/classify-ml")
def classify_ml(req: ClassifyReq):
    if not ml_manager.ready():
        return {"ok": False, "message": "ML model not loaded. Set SEQ_CLS_MODEL_DIR to fine-tuned dir.", "model": "bert-ml"}
    result = ml_manager.classify(req.text)
    return {"ok": result.get("ok", True), "label": result.get("label", ""), "confidence": result.get("confidence", 0.0), "model": "bert-ml", "scores": result.get("scores", {})}