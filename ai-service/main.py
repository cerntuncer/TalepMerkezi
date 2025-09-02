# ai-service/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import re
import unicodedata

from ml.bert_classifier import ModelManager

app = FastAPI(title="AI Classifier", version="0.5.0")

ml_manager_support2 = ModelManager(model_dir="models/model_tr_support2")

# --- İsteğe bağlı CORS (MVC/Frontend için aç) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "https://localhost:5001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==== Şema ====
class ClassifyReq(BaseModel):
    text: str

class ClassifyResp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str

class ClassifyMLResp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str
    scores: Optional[List[Dict]] = None


# ==== Model Yöneticisi ====
ml_manager = ModelManager()  # ENV: SEQ_CLS_MODEL_DIR ile override edebilirsin


@app.on_event("startup")
def _startup():
    try:
        ml_manager.load()
        print("[AI] ML model loaded successfully.")
    except Exception as e:
        # Model yüklenemezse servis yine çalışır; /classify-ml kurala düşer.
        print(f"[AI] ML model load FAILED: {e}")


# ==== Yardımcılar (senin kural fonksiyonların) ====
def normalize_text(text: str) -> str:
    """Lowercase, diacritics strip, whitespace collapse."""
    t = (text or "").lower().strip()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"\s+", " ", t)
    return t

def contains_keyword(text: str, keywords: List[str]) -> bool:
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text, flags=re.IGNORECASE):
            return True
    return False

def contains_phrase(text: str, phrases: List[str]) -> bool:
    for ph in phrases:
        if ph in text:
            return True
    return False


# ==== Health ====
@app.get("/ping")
def ping():
    return {"ok": True}


# ==== Kural Tabanlı ====
@app.post("/classify", response_model=ClassifyResp)
def classify(req: ClassifyReq):
    orig = (req.text or "")
    t = orig.lower()
    tn = normalize_text(orig)

    # Phrase-based rules (daha spesifik → yüksek güven)
    phrase_rules = [
        ([
            "islem yapamiyorum", "islem basarisiz", "islem hatasi", "islem gerceklesmiyor", "islem olmuyor",
            "calismiyor", "baglanamiyorum", "uygulama cokuyor", "hata veriyor", "error"
         ], "TeknikDestek", 0.84),
        (["parolami unuttum", "sifre sifirla", "sifre yenile", "hesap kilitlendi"], "SifreIslemleri", 0.87),
        (["uyeligi iptal", "uyeligimi iptal", "abonelik iptal", "hesabi kapat"], "Iptal", 0.83),
        (["siparis vermek", "urun satin almak", "yeni urun", "stok var mi"], "UrunTalebi", 0.78),
        (["fatura odeme", "odeme basarisiz", "e-fatura", "tutar cok yuksek", "faturam"], "Fatura", 0.82),
        (["garanti kapsami", "garanti kapsaminda", "iade etmek", "onarim"], "Garanti", 0.72),
    ]
    for phrases, label, conf in phrase_rules:
        if contains_phrase(tn, phrases):
            return ClassifyResp(ok=True, label=label, confidence=conf, model="rule-v2")

    # Keyword-based rules
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

    # Fallback (belirsizlik)
    return ClassifyResp(ok=True, label="GenelTalep", confidence=0.60, model="rule-v2")


# ==== ML + Fallback ====
@app.post("/classify-ml", response_model=ClassifyMLResp)
def classify_ml(req: ClassifyReq):
    if ml_manager.ready():
        r = ml_manager.classify(req.text)
        return ClassifyMLResp(
            ok=r.get("ok", True),
            label=r.get("label", ""),
            confidence=float(r.get("confidence", 0.0)),
            model="bert-ml",
            scores=r.get("scores", []),
        )

    # Model hazır değil → kurala düş
    rb = classify(req)
    return ClassifyMLResp(
        ok=True,
        label=rb.label,
        confidence=rb.confidence,
        model=rb.model,
        scores=[],
    )

@app.post("/classify-support2", response_model=ClassifyMLResp)
def classify_support2(req: ClassifyReq):
    if ml_manager_support2.ready():
        r = ml_manager_support2.classify(req.text)
        return ClassifyMLResp(
            ok=r.get("ok", True),
            label=r.get("label", ""),
            confidence=float(r.get("confidence", 0.0)),
            model="bert-support2",
            scores=r.get("scores", []),
        )
    # Model hazır değilse fallback
    return ClassifyMLResp(ok=False, label="", confidence=0.0, model="bert-support2", scores=[])