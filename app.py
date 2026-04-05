"""
app.py  –  FastAPI WAF Backend
Run:  uvicorn app:app --reload --port 8000
"""

import asyncio
import os
import random
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from model import WAFInferencer, LABELS
from log_parser import parse_apache_line, parse_nginx_error_line

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Transformer WAF API",
    description="AI-powered Web Application Firewall using a Transformer model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model.pt"
inferencer: Optional[WAFInferencer] = None
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ROOT_INDEX_FILE = BASE_DIR / "index.html"
STATIC_INDEX_FILE = STATIC_DIR / "index.html"


def _dashboard_file_path() -> Path:
    """Resolve dashboard file location with backward compatibility."""
    if STATIC_INDEX_FILE.exists():
        return STATIC_INDEX_FILE
    return ROOT_INDEX_FILE

# ── In-Memory State ──────────────────────────────────────────────────────────

MAX_HISTORY = 500
request_log: deque = deque(maxlen=MAX_HISTORY)

stats = {
    "total": 0,
    "blocked": 0,
    "allowed": 0,
    "by_label": {l: 0 for l in LABELS},
}


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    global inferencer
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    inferencer = WAFInferencer(model_path=model_path)
    if model_path:
        print(f"✓ Loaded trained model from {MODEL_PATH}")
    else:
        print("⚠ No model.pt found — using untrained model. Run: python train.py")


# ── Schemas ──────────────────────────────────────────────────────────────────

class InspectRequest(BaseModel):
    request_text: str
    ip: Optional[str] = "unknown"
    path: Optional[str] = "/"
    method: Optional[str] = "GET"


class InspectResponse(BaseModel):
    id: str
    timestamp: str
    ip: str
    method: str
    path: str
    request_text: str
    label: str
    is_attack: bool
    confidence: float
    probabilities: dict
    action: str          # "BLOCKED" | "ALLOWED"
    latency_ms: float


class BatchRequest(BaseModel):
    requests: list[InspectRequest]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _inspect(req: InspectRequest) -> InspectResponse:
    t0 = time.perf_counter()
    result = inferencer.predict(req.request_text)
    latency = round((time.perf_counter() - t0) * 1000, 2)

    action = "BLOCKED" if result["is_attack"] else "ALLOWED"
    entry = InspectResponse(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.utcnow().isoformat() + "Z",
        ip=req.ip,
        method=req.method,
        path=req.path,
        request_text=req.request_text[:200],
        label=result["label"],
        is_attack=result["is_attack"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        action=action,
        latency_ms=latency,
    )

    # Update state
    request_log.appendleft(entry.dict())
    stats["total"] += 1
    if result["is_attack"]:
        stats["blocked"] += 1
    else:
        stats["allowed"] += 1
    stats["by_label"][result["label"]] += 1

    return entry


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    dashboard = _dashboard_file_path()
    if not dashboard.exists():
        raise HTTPException(500, "Dashboard file not found (expected index.html)")
    return FileResponse(dashboard)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": inferencer is not None}


@app.post("/inspect", response_model=InspectResponse, tags=["WAF"])
async def inspect_request(req: InspectRequest):
    """Classify a single HTTP request string."""
    if not inferencer:
        raise HTTPException(503, "Model not loaded")
    return _inspect(req)


@app.post("/inspect/batch", tags=["WAF"])
async def inspect_batch(batch: BatchRequest):
    """Classify multiple requests at once."""
    if not inferencer:
        raise HTTPException(503, "Model not loaded")
    return [_inspect(r).dict() for r in batch.requests[:50]]


@app.post("/inspect/log", tags=["WAF"])
async def inspect_log_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    fmt: str = "apache",
):
    """Upload an Apache / Nginx log file for bulk analysis."""
    content = await file.read()
    lines = content.decode("utf-8", errors="replace").splitlines()

    parser = parse_apache_line if fmt == "apache" else parse_nginx_error_line
    results = []
    for line in lines[:200]:           # cap at 200 lines
        entry = parser(line)
        if entry:
            req = InspectRequest(
                request_text=entry.to_waf_input(),
                ip=entry.ip,
                method=entry.method,
                path=entry.path,
            )
            results.append(_inspect(req).dict())

    return {"processed": len(results), "results": results}


@app.get("/dashboard/stats", tags=["Dashboard"])
async def dashboard_stats():
    """Aggregate statistics for the dashboard."""
    total = stats["total"] or 1
    return {
        "total": stats["total"],
        "blocked": stats["blocked"],
        "allowed": stats["allowed"],
        "block_rate": round(stats["blocked"] / total * 100, 1),
        "by_label": stats["by_label"],
    }


@app.get("/dashboard/recent", tags=["Dashboard"])
async def recent_requests(limit: int = 50):
    """Most recent inspected requests."""
    return list(request_log)[:limit]


@app.delete("/dashboard/reset", tags=["Dashboard"])
async def reset_stats():
    """Clear all stats and history."""
    request_log.clear()
    stats.update(total=0, blocked=0, allowed=0, by_label={l: 0 for l in LABELS})
    return {"message": "Reset complete"}


# ── Demo Simulator (populates dashboard without a real WAF) ──────────────────

DEMO_REQUESTS = [
    # normal
    ("GET /index.html HTTP/1.1", "GET", "/index.html"),
    ("GET /api/products?page=2 HTTP/1.1", "GET", "/api/products"),
    ("POST /login username=alice HTTP/1.1", "POST", "/login"),
    ("GET /static/app.js HTTP/1.1", "GET", "/static/app.js"),
    # attacks
    ("GET /search?q=' OR '1'='1--  HTTP/1.1", "GET", "/search"),
    ("GET /page?id=<script>alert(1)</script> HTTP/1.1", "GET", "/page"),
    ("GET /../../../etc/passwd HTTP/1.1", "GET", "/../../../etc/passwd"),
    ("POST /cmd?run=; cat /etc/passwd HTTP/1.1", "POST", "/cmd"),
]

IPS = ["203.0.113.5", "198.51.100.42", "10.0.0.17", "192.168.1.99", "172.16.0.3"]


@app.post("/demo/simulate", tags=["Demo"])
async def simulate_traffic(n: int = 20):
    """Generate n random requests for demo purposes."""
    results = []
    for _ in range(min(n, 100)):
        text, method, path = random.choice(DEMO_REQUESTS)
        req = InspectRequest(
            request_text=text,
            ip=random.choice(IPS),
            method=method,
            path=path,
        )
        results.append(_inspect(req).dict())
    return {"simulated": len(results), "results": results}


# ── Static Files ─────────────────────────────────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
