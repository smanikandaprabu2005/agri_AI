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
from main_v2 import load_system as load_system_v2
from config import DEFAULT_CITY, GEN_TEMPERATURE, GEN_MAX_TOKENS, FINETUNED_CKPT, RETRIEVER_DIR
from weather.weather_api import WeatherService

# ── VectorDB Configuration ──────────────────────────────────
VECTOR_DB_DIR = os.path.join("saved_models", "vector_index")

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

# ── Load config defaults ──────────────────────────────────────

# ── Lazy load system (loaded once on first request) ────────────
_system = None
_load_error = None
@app.on_event("startup")
def startup_event():
    """Eagerly load the RAG system when the FastAPI server starts."""
    try:
        _load_system()
        print("[API] Startup complete — system loaded")
    except Exception as e:
        print(f"[API] Startup load failed: {e}")
        raise
def _load_system():
    global _system, _load_error
    if _system is not None:
        return _system
    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        print("[API] Loading system via main_v2...")
        
        # Check VectorDB availability
        vdb_config = os.path.join(VECTOR_DB_DIR, "config.json")
        use_vector_db = os.path.exists(vdb_config)
        
        if use_vector_db:
            print(f"[API] VectorDB found at {VECTOR_DB_DIR} — enabling hybrid retrieval")
        else:
            print(f"[API] VectorDB not found — will use Word2Vec fallback")
        
        # Load system with VectorDB support
        gen, memory = load_system_v2(
            weights_path=FINETUNED_CKPT,
            api_key=os.environ.get("OWM_API_KEY", ""),
            city=DEFAULT_CITY,
            temperature=GEN_TEMPERATURE,
            max_tokens=GEN_MAX_TOKENS,
            use_vector_db=use_vector_db  # Use VectorDB if available
        )
        
        # Get backend info from response generator's context builder
        ctx_builder = gen.ctx_builder
        backend_info = "ContextBuilderV2 (VectorDB + Word2Vec Hybrid)" if hasattr(ctx_builder, 'use_vector_db') and ctx_builder.use_vector_db else "ContextBuilder (Word2Vec)"
        
        weather = WeatherService(city=DEFAULT_CITY, api_key=os.environ.get("OWM_API_KEY", ""))
        
        _system = {
            "gen": gen,
            "memory": memory,
            "weather": weather,
            "backend": backend_info,
            "vector_db_enabled": use_vector_db,
            "context_builder": ctx_builder,
        }
        print(f"[API] Ready — backend: {_system['backend']}")
        print(f"[API] Generator sources: template | rag_generated | retrieval_summarized | retrieval_fallback | fallback\n")
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
            "vector_db_enabled": sys_state["vector_db_enabled"],
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/context")
def get_context_info(query: str):
    """Debug endpoint to inspect context builder behavior."""
    try:
        sys_state = _load_system()
        ctx_builder = sys_state["context_builder"]
        
        intent = ctx_builder.get_intent(query)
        retrieved = ctx_builder.get_retrieved(query)
        context_str, confidence = ctx_builder.build_context_str(query)
        
        return {
            "query": query,
            "intent": intent,
            "context_length": len(context_str),
            "confidence": confidence,
            "retrieved_count": len(retrieved),
            "retrieved_docs": [
                {"score": float(score), "text": text[:200]} 
                for score, text in retrieved[:3]
            ],
            "context_preview": context_str[:500] + ("..." if len(context_str) > 500 else ""),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


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
        ctx_builder = sys_state["context_builder"]

        # Process input (updates farmer profile via memory)
        memory.process_input(req.query)

        # Update weather city if provided
        if req.city:
            ctx_builder.weather_svc.city = req.city

        # Generate response (uses ContextBuilderV2 internally)
        answer, source = gen.generate(req.query)

        # Store response in memory
        memory.add_response(answer)

        # Log context builder state if verbose
        if req.verbose:
            intent = ctx_builder.get_intent(req.query)
            retrieved = ctx_builder.get_retrieved(req.query)
            print(f"[Chat] Query: {req.query}")
            print(f"[Chat] Detected intent: {intent}")
            print(f"[Chat] Retrieved {len(retrieved)} documents")
            print(f"[Chat] Response source: {source}")

        # Update stats
        _stats["total_requests"] += 1
        if source not in _stats:
            _stats[f"{source}_hits"] = 0
        _stats[f"{source}_hits"] += 1

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
        ctx_builder = sys_state["context_builder"]
        
        if city:
            ctx_builder.weather_svc.city = city
        
        w = ctx_builder.weather_svc.get(city or DEFAULT_CITY)
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
