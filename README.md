# 🛡 WAF Shield — Transformer-based Web Application Firewall

> SIH25172 · Bade Rupali Ramesh · PRN 2124UCSF2023

---

## Architecture

```
waf_project/
├── app.py          ← FastAPI REST backend  (all API endpoints)
├── model.py        ← Transformer model     (encoding, inference)
├── train.py        ← Training pipeline     (synthetic dataset + loop)
├── log_parser.py   ← Apache / Nginx parser (stream logs → WAF input)
├── requirements.txt
└── index.html      ← Dashboard UI          (real-time monitoring)
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model  *(~2 minutes on CPU)*
```bash
python train.py --epochs 15
# Saves model.pt with ~95%+ accuracy on synthetic data
```

### 3. Start the backend
```bash
uvicorn app:app --reload --port 8000
```

### 4. Open the dashboard
```
http://localhost:8000
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/inspect` | Classify a single HTTP request |
| POST | `/inspect/batch` | Classify up to 50 requests at once |
| POST | `/inspect/log` | Upload Apache/Nginx log file |
| GET  | `/dashboard/stats` | Aggregate statistics |
| GET  | `/dashboard/recent` | Recent request log |
| POST | `/demo/simulate` | Generate demo traffic |
| DELETE | `/dashboard/reset` | Reset all stats |

### Example — inspect a request
```bash
curl -X POST http://localhost:8000/inspect \
  -H "Content-Type: application/json" \
  -d '{"request_text": "GET /search?q='\'' OR 1=1-- HTTP/1.1", "ip": "1.2.3.4"}'
```

### Response
```json
{
  "id": "a1b2c3d4",
  "label": "sqli",
  "is_attack": true,
  "confidence": 0.9821,
  "action": "BLOCKED",
  "latency_ms": 4.2,
  "probabilities": {
    "normal": 0.003,
    "sqli": 0.982,
    "xss": 0.008,
    "path_traversal": 0.004,
    "cmd_injection": 0.003
  }
}
```

---

## Model Details

| Component | Value |
|-----------|-------|
| Architecture | Transformer Encoder (BERT-style) |
| Embedding | Character-level + [CLS] token |
| d_model | 128 |
| Attention heads | 4 |
| Encoder layers | 3 |
| Max sequence length | 256 chars |
| Classes | normal · sqli · xss · path_traversal · cmd_injection |
| Parameters | ~1.2 M |
| Training data | Synthetic (3,000 samples, 5 classes) |

---

## Attack Classes Detected

| Label | Example |
|-------|---------|
| `sqli` | `' OR 1=1--` · `UNION SELECT` |
| `xss`  | `<script>alert(1)</script>` · `onerror=` |
| `path_traversal` | `../../etc/passwd` |
| `cmd_injection` | `; cat /etc/passwd` · `| whoami` |
| `normal` | Regular GET/POST traffic |

---

## Extending the Project

- **Use real labelled data** — replace synthetic samples in `train.py` with CSIC 2010 HTTP dataset or similar.
- **Streaming WAF** — hook `app.py` as a FastAPI middleware to proxy real traffic.
- **Online learning** — collect mis-classified samples and fine-tune `model.pt` incrementally.
- **Rule fusion** — combine model score with a lightweight rule layer (ModSecurity CRS) for defence-in-depth.
