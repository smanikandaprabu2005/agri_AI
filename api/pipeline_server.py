"""
api/pipeline_server.py
========================
FastAPI server for the multi-model advisory pipeline.

Endpoints:
  POST /advisory    — full pipeline (crop + fertilizer + pesticide + LLM)
  POST /chat        — conversational Q&A (existing chatbot mode)
  GET  /weather     — weather data
  GET  /profile     — farmer profile
  POST /profile     — update farmer profile
  GET  /health      — health check
  POST /reset       — reset session

This wraps the existing SageStorm V2 backend with the new pipeline.
"""

import os, sys, time
from typing import Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Request / Response models ─────────────────────────────────
class AdvisoryRequest(BaseModel):
    # Required context
    query:       str
    location:    Optional[str] = ""
    soil_type:   Optional[str] = ""
    season:      Optional[str] = ""
    # Optional sensor/test data
    temperature: Optional[float] = 28.0
    humidity:    Optional[float] = 65.0
    nitrogen:    Optional[float] = 50.0
    phosphorus:  Optional[float] = 30.0
    potassium:   Optional[float] = 30.0
    growth_stage: Optional[str] = ""
    symptoms:    Optional[str]  = ""
    crop_name:   Optional[str]  = ""    # if user already knows crop


class ChatRequest(BaseModel):
    query:   str
    city:    Optional[str] = ""
    verbose: Optional[bool] = False


class AdvisoryResponse(BaseModel):
    advisory:    str
    source:      str
    crop_name:   str
    crop_conf:   float
    fertilizer:  str
    pest_name:   str
    pesticide:   str
    warnings:    list
    latency_ms:  float
    timestamp:   str


class ChatResponse(BaseModel):
    answer:     str
    source:     str
    latency_ms: float
    timestamp:  str


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Strom Sage AI — Pipeline API",
    description="Multi-model agriculture advisory: Crop + Fertilizer + Pesticide + LLM",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy system load ──────────────────────────────────────────
_pipeline = None
_chat_gen  = None
_memory    = None
_weather   = None
_loaded    = False


def _load_system():
    global _pipeline, _chat_gen, _memory, _weather, _loaded
    if _loaded:
        return

    from config import (
        FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY,
        DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE,
    )
    from retrieval.vector_search    import load_retriever
    from memory.memory_manager      import MemoryManager
    from weather.weather_api        import WeatherService
    from rag.context_builder        import ContextBuilder
    from chatbot.response_generator import ResponseGenerator
    from models.tokenizer           import SageTokenizer
    from pipeline.advisory_pipeline import AdvisoryPipeline

    print("[API v3] Loading system...")

    # Existing RAG chatbot
    try:
        ret, _ = load_retriever(RETRIEVER_DIR)
    except Exception as e:
        print(f"  [!] Retriever error: {e}")
        class DummyRet:
            def retrieve(self, q, top_k=5): return []
        ret = DummyRet()

    _memory  = MemoryManager()
    _weather = WeatherService(
        city=DEFAULT_CITY,
        api_key=os.environ.get("OWM_API_KEY", "")
    )
    ctx = ContextBuilder(ret, _memory, _weather, top_k=5)
    tok = SageTokenizer()

    model = None
    try:
        import torch
        if os.path.exists(FINETUNED_CKPT):
            from models.sagestorm_v2 import SageStormV2
            model = SageStormV2.load(FINETUNED_CKPT, DEVICE)
            print(f"  [✓] SageStorm V2 loaded")
    except ImportError:
        print("  [!] PyTorch not available — retrieval-only mode")

    _chat_gen = ResponseGenerator(
        model, tok, ret, ctx,
        max_tokens=GEN_MAX_TOKENS,
        temperature=GEN_TEMPERATURE,
    )

    # New multi-model pipeline
    llm_backend = os.environ.get("STROM_SAGE_LLM", "sagestorm")
    _pipeline = AdvisoryPipeline(
        llm_backend=llm_backend,
        sagestorm_generator=_chat_gen,
        weather_service=_weather,
    )

    _loaded = True
    print("[API v3] System ready.")


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        _load_system()
        return {
            "status": "ok",
            "pipeline": "AdvisoryPipeline v3.0",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/advisory", response_model=AdvisoryResponse)
def advisory(req: AdvisoryRequest):
    """Full multi-model advisory pipeline."""
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")

    _load_system()

    raw = req.dict()
    try:
        result = _pipeline.run(raw)
    except Exception as e:
        raise HTTPException(500, f"Pipeline error: {e}")

    return AdvisoryResponse(
        advisory=result["advisory"],
        source=result["source"],
        crop_name=result["crop"]["crop_name"],
        crop_conf=result["crop"]["confidence"],
        fertilizer=result["fertilizer"]["fertilizer_name"],
        pest_name=result["pesticide"]["pest_name"],
        pesticide=result["pesticide"]["pesticide_name"],
        warnings=result["warnings"],
        latency_ms=result["latency_ms"],
        timestamp=datetime.now().isoformat(),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Conversational Q&A — existing chatbot mode (backward compatible)."""
    if not req.query.strip():
        raise HTTPException(400, "query cannot be empty")

    _load_system()

    t0 = time.perf_counter()

    _memory.process_input(req.query)
    if req.city:
        _weather.city = req.city

    answer, source = _chat_gen.generate(req.query)
    _memory.add_response(answer)

    return ChatResponse(
        answer=answer,
        source=source,
        latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        timestamp=datetime.now().isoformat(),
    )


@app.get("/weather")
def get_weather(city: Optional[str] = None):
    _load_system()
    try:
        w = _weather.get(city)
        return w
    except Exception as e:
        return {"error": str(e), "src": "error"}


@app.get("/profile")
def get_profile():
    _load_system()
    return _memory.long.data


@app.post("/profile")
def update_profile(profile: dict):
    _load_system()
    _memory.long.data.update(profile)
    _memory.long.save()
    return {"status": "updated", "profile": _memory.long.data}


@app.post("/reset")
def reset_session():
    _load_system()
    _memory.reset_session()
    return {"status": "session_cleared"}


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "pipeline_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
