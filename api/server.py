"""
api/server.py
=============
FastAPI backend that bridges the React UI to SageStorm V2.

Exposes:
  POST /chat          — main chat endpoint
  GET  /weather       — weather data
  GET  /profile       — farmer profile
  POST /profile       — update farmer profile
  GET  /health        — health check
  GET  /stats         — session statistics

CORS is enabled for the React dev server (localhost:3000 / localhost:5173).

Usage:
    pip install fastapi uvicorn
    python api/server.py
    # or
    uvicorn api.server:app --reload --port 8000
"""

import os, sys, json, time
from typing import Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Request / Response Models ─────────────────────────────────
class ChatRequest(BaseModel):
    query:   str
    city:    Optional[str] = "Guwahati"
    verbose: Optional[bool] = False

class ChatResponse(BaseModel):
    answer:  str
    source:  str
    latency_ms: float
    timestamp:  str

class ProfileUpdate(BaseModel):
    name:       Optional[str] = None
    crop_type:  Optional[str] = None
    location:   Optional[str] = None
    soil_type:  Optional[str] = None
    farm_size:  Optional[str] = None

class WeatherResponse(BaseModel):
    city:     str
    temp:     float
    humid:    int
    wind:     float
    desc:     str
    rain:     bool
    rain_pct: int
    src:      str

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="Strom Sage AI API",
    description="Agriculture advisory AI powered by SageStorm V2",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy load system (loaded once on first request) ────────────
_system = None
_load_error = None

def _load_system():
    global _system, _load_error
    if _system is not None:
        return _system
    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        from config import FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY, DEVICE, GEN_TEMPERATURE, GEN_MAX_TOKENS
        from retrieval.vector_search import load_retriever
        from memory.memory_manager import MemoryManager
        from weather.weather_api import WeatherService
        from rag.context_builder import ContextBuilder
        from chatbot.response_generator import ResponseGenerator
        from models.tokenizer import SageTokenizer

        print("[API] Loading system components...")
        try:
            ret, vocab = load_retriever(RETRIEVER_DIR)
        except Exception as e:
            print(f"  [!] Retriever not found: {e}")
            ret, vocab = None, None

        if ret is None:
            class DummyRetriever:
                def retrieve(self, query, top_k=5):
                    return []
            ret = DummyRetriever()
            vocab = None
        memory = MemoryManager()
        weather = WeatherService(city=DEFAULT_CITY, api_key=os.environ.get("OWM_API_KEY", ""))
        ctx = ContextBuilder(ret, memory, weather, top_k=5)
        tok = SageTokenizer()

        model = None
        backend = "retrieval-only"
        try:
            import torch
            if os.path.exists(FINETUNED_CKPT):
                from models.sagestorm_v2 import SageStormV2
                model = SageStormV2.load(FINETUNED_CKPT, DEVICE)
                backend = f"SageStorm V2 ({model.param_count()['total_M']}M params)"
            else:
                print(f"  [!] Weights not found: {FINETUNED_CKPT}")
        except ImportError:
            print("  [!] PyTorch not installed — retrieval-only mode")

        gen = ResponseGenerator(
            model, tok, ret, ctx,
            max_tokens=GEN_MAX_TOKENS,
            temperature=GEN_TEMPERATURE
        )

        _system = {
            "gen": gen,
            "memory": memory,
            "weather": weather,
            "backend": backend,
        }
        print(f"[API] Ready — backend: {backend}")
        return _system

    except Exception as e:
        _load_error = str(e)
        print(f"[API] LOAD ERROR: {e}")
        raise


# ── Stats tracker ──────────────────────────────────────────────
_stats = {
    "total_requests": 0,
    "template_hits": 0,
    "retrieval_hits": 0,
    "sagestorm_hits": 0,
    "fallback_hits": 0,
    "start_time": datetime.now().isoformat(),
}


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check — also attempts to load the system."""
    try:
        sys_state = _load_system()
        return {
            "status": "ok",
            "backend": sys_state["backend"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Main chat endpoint called by the React UI."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    t0 = time.perf_counter()

    try:
        sys_state = _load_system()
        gen = sys_state["gen"]
        memory = sys_state["memory"]

        # Process input (updates farmer profile)
        memory.process_input(req.query)

        # Update weather city if provided
        if req.city and req.city != sys_state["weather"].city:
            sys_state["weather"].city = req.city

        # Generate response
        answer, source = gen.generate(req.query)

        # Store response in memory
        memory.add_response(answer)

        # Update stats
        _stats["total_requests"] += 1
        _stats[f"{source}_hits"] = _stats.get(f"{source}_hits", 0) + 1

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)

        return ChatResponse(
            answer=answer,
            source=source,
            latency_ms=latency_ms,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(500, f"Generation error: {str(e)}")


@app.get("/weather")
def get_weather(city: Optional[str] = None):
    """Get weather data for a city."""
    try:
        sys_state = _load_system()
        w = sys_state["weather"].get(city)
        return WeatherResponse(
            city=w["city"],
            temp=w["temp"],
            humid=w["humid"],
            wind=w["wind"],
            desc=w["desc"],
            rain=w["rain"],
            rain_pct=w["rain_pct"],
            src=w["src"],
        )
    except Exception as e:
        # Return mock weather on error
        return WeatherResponse(
            city=city or "Unknown",
            temp=28.0,
            humid=65,
            wind=10.0,
            desc="Data unavailable",
            rain=False,
            rain_pct=0,
            src="error",
        )


@app.get("/profile")
def get_profile():
    """Get current farmer profile."""
    try:
        sys_state = _load_system()
        return sys_state["memory"].long.data
    except Exception:
        return {}


@app.post("/profile")
def update_profile(profile: ProfileUpdate):
    """Update farmer profile fields."""
    try:
        sys_state = _load_system()
        memory = sys_state["memory"]
        updates = {k: v for k, v in profile.dict().items() if v is not None}
        memory.long.data.update(updates)
        memory.long.save()
        return {"status": "updated", "profile": memory.long.data}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/reset")
def reset_session():
    """Reset conversation session (keeps profile)."""
    try:
        sys_state = _load_system()
        sys_state["memory"].reset_session()
        return {"status": "session_cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/stats")
def get_stats():
    """Session statistics."""
    return {
        **_stats,
        "uptime_seconds": (datetime.now() - datetime.fromisoformat(_stats["start_time"])).total_seconds(),
    }


# ── Run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
