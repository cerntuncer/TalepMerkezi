# AI Worker for TalepMerkezi

SQLite'taki `Talepler` tablosunda AI etiketi olmayan kayıtları bulur, Zero‑Shot sınıflandırma ile etiketler ve `PredictedLabel` alanını günceller.

## Kurulum

```bash
cd /workspace/ai_worker
# (Opsiyonel) Sanal ortam
python3 -m venv .venv || true
source .venv/bin/activate || true
pip install -r requirements.txt || true
# Eğer venv olmazsa:
python3 -m pip install --break-system-packages -r requirements.txt
cp .env.example .env
```

## Çalışma Modları

### 1) Uzak API (varsayılan)
- `.env` içinde `HUGGINGFACE_API_TOKEN` girin (Read token).
```bash
MODE=once python3 worker.py
```

### 2) Yerel Transformers (internet/ücret olmadan)
- `.env` içinde `USE_LOCAL_MODEL=1` yapın.
- `LOCAL_MODEL`'i indirdiğiniz model adı veya yerel klasör yoluna ayarlayın.
- `requirements.txt` ile birlikte `torch` kurun (platformunuza uygun):
```bash
# CPU örnek (Linux):
pip install --index-url https://download.pytorch.org/whl/cpu torch
```
- Sonra çalıştırın:
```bash
MODE=once python3 worker.py
```

## Ayarlar (.env)
- `SQLITE_PATH`: `/workspace/TalepMerkezi/app.db`
- `CANDIDATE_LABELS`: Virgülle ayrılmış Türkçe etiketler (veya `labels.tr.txt`).
- `WORKER_POLL_SECONDS`: Watch modunda bekleme süresi.
- `BATCH_SIZE`: Her turda işlenecek kayıt sayısı.

## Sürekli İzleme
```bash
python3 worker.py
```

## Notlar
- İlk çağrıda 503 gelebilir (model yüklenmesi); otomatik yeniden dener.
- Varsayılan olarak, etiket yazarken `Status = InProgress` yapar. İsterseniz değiştirebiliriz.
