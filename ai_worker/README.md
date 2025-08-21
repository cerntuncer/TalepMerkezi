# AI Worker for TalepMerkezi

Python işçi, SQLite içindeki `Talepler` tablosunda AI etiketi olmayan kayıtları alır, Hugging Face Zero-Shot sınıflandırma API'siyle etiketler ve `PredictedLabel` alanını günceller.

## Kurulum

```bash
cd /workspace/ai_worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # düzenleyin
```

`.env` içindeki önemli değişkenler:
- `SQLITE_PATH`: `/workspace/TalepMerkezi/app.db`
- `HUGGINGFACE_API_TOKEN`: (opsiyonel, ücret/limit için gerebilir)
- `CANDIDATE_LABELS`: Virgülle ayrılmış Türkçe etiketler, boşsa `labels.tr.txt` okunur
- `WORKER_POLL_SECONDS`: watch modunda bekleme süresi
- `BATCH_SIZE`: her döngüde işlenecek kayıt sayısı

## Çalıştırma

- Tek sefer çalıştırma:
```bash
MODE=once python worker.py
```

- Sürekli izleme (varsayılan):
```bash
python worker.py
```

## Notlar
- API kullanımı ilk istekte 503 döndürebilir; otomatik yeniden deneme yapılır.
- `PredictedLabel` yazılırken `Status` alanı `InProgress` olarak güncellenir. İsterseniz bunu kaldırabilir veya farklı mantık ekleyebilirsiniz.
