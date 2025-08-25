from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AI Classifier", version="0.1.0")

class ClassifyReq(BaseModel):
    text: str

class ClassifyResp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str

@app.get("/ping")
def ping():
    return {"ok": True}

@app.post("/classify", response_model=ClassifyResp)
def classify(req: ClassifyReq):
    t = (req.text or "").lower()
    if any(k in t for k in ["şifre","parola","unut","reset"]):
        return ClassifyResp(ok=True, label="SifreIslemleri", confidence=0.85, model="rule-v0")
    if any(k in t for k in ["fatura","e-fatura","irsaliye"]):
        return ClassifyResp(ok=True, label="Fatura", confidence=0.80, model="rule-v0")
    if any(k in t for k in ["iptal","fesih"]):
        return ClassifyResp(ok=True, label="Iptal", confidence=0.75, model="rule-v0")
    if any(k in t for k in ["hata","bağlanmıyor","çöktü","çalışmıyor"]):
        return ClassifyResp(ok=True, label="TeknikDestek", confidence=0.70, model="rule-v0")
    return ClassifyResp(ok=True, label="GenelTalep", confidence=0.60, model="rule-v0")
