from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

app = FastAPI(title="AI Service", version="1.0.0")

# --- Model yolları ---
MODEL_DIR = "./models/bert_balanced_final"
TRANS_MODEL = "Helsinki-NLP/opus-mt-tr-en"

# --- Çeviri pipeline ---
translator = pipeline("translation", model=TRANS_MODEL)

# --- BERT sınıflandırıcı yükleme ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)

# --- Request/Response modelleri ---
class Req(BaseModel):
    text: str

class Resp(BaseModel):
    ok: bool
    label: str
    confidence: float
    model: str
    translated_text: str   # 👈 çeviriyi de döndürüyoruz

@app.get("/")
async def root():
    return {"status": "ok", "msg": "AI Service is running"}

@app.post("/classify", response_model=Resp)
async def classify(req: Req):
    # 1) Türkçe → İngilizce çeviri
    translated = translator(req.text, max_length=512)[0]['translation_text']
    print(f"[Çeviri] TR: {req.text} -> EN: {translated}")

    # 2) İngilizce metni BERT'e gönder
    inputs = tokenizer(translated, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        conf, pred = torch.max(probs, dim=1)

    label = model.config.id2label[pred.item()]

    return Resp(
        ok=True,
        label=label,
        confidence=conf.item(),
        model="bert_balanced_final",
        translated_text=translated   # 👈 Swagger çıktısına eklendi
    )
