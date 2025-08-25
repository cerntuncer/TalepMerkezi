from fastapi import FastAPI
from pydantic import BaseModel
import re

app = FastAPI(title="AI Classifier", version="0.3.1")

class ClassifyReq(BaseModel):
    text: str

class ClassifyResp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str

def contains_keyword(text: str, keywords: list[str]) -> bool:
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            return True
    return False

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/classify", response_model=ClassifyResp)
def classify(req: ClassifyReq):
    t = (req.text or "").lower()

    rules = [
        (["şifre", "parola", "unut", "reset", "giriş", "kilitlendi", "açılmıyor"], "SifreIslemleri", 0.85),
        (["fatura", "e-fatura", "irsaliye", "belge", "tutar", "ücret", "faturam", "çok yüksek", "yüksek geldi"], "Fatura", 0.82),
        (["iptal", "fesih", "dondur", "üyelik bitir", "üyeliğimi iptal", "hesabımı kapat", "abonelik iptal"], "Iptal", 0.80),
        (["bağlanmıyor", "çöktü", "hata", "error", "donuyor", "takılıyor", "yavaşladı"], "TeknikDestek", 0.78),
        (["sipariş", "ürün", "stok", "almak istiyorum", "yeni ürün", "ürün satın almak istiyorum"], "UrunTalebi", 0.75),
        (["garanti", "iade", "bozuldu", "onarım", "servis", "kapsamında mı", "garanti kapsamında"], "Garanti", 0.70),
    ]

    for keywords, label, conf in rules:
        if contains_keyword(t, keywords):
            return ClassifyResp(ok=True, label=label, confidence=conf, model="rule-v1")

    return ClassifyResp(ok=True, label="GenelTalep", confidence=0.60, model="rule-v1")
