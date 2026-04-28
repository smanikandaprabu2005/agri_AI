"""
api/server_v3.py
================
SageStorm V3 FastAPI Backend
=============================
Adds:
  POST /advisory   — Full ML + SLM advisory from field parameters
  POST /predict    — ML-only predictions (no SLM narration)
  GET  /advisory/last — Returns last generated advisory with all details

Existing endpoints (/chat, /weather, /profile, etc.) are unchanged.
"""

import os
import sys
import time
import json
from typing import Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import (
    FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY,
    DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE,
)

# ── Request / Response models ─────────────────────────────────
class FieldInputs(BaseModel):
    """Structured field inputs for ML advisory."""
    soil_type      : str   = Field("Loamy", description="Clay/Loamy/Sandy/Silt/Black/Red/Alluvial")
    ph             : float = Field(6.5,  ge=3.0, le=10.0)
    soil_moisture  : float = Field(30.0, ge=0.0, le=100.0)
    organic_carbon : float = Field(0.8,  ge=0.0, le=10.0)
    ec             : float = Field(0.5,  ge=0.0, le=10.0)
    N              : float = Field(80.0, ge=0.0, le=300.0, description="Nitrogen kg/ha")
    P              : float = Field(40.0, ge=0.0, le=300.0, description="Phosphorus kg/ha")
    K              : float = Field(40.0, ge=0.0, le=300.0, description="Potassium kg/ha")
    temperature    : float = Field(25.0, ge=0.0, le=55.0)
    humidity       : float = Field(65.0, ge=0.0, le=100.0)
    rainfall       : float = Field(100.0, ge=0.0, le=5000.0)
    season         : str   = Field("Kharif", description="Kharif/Rabi/Zaid")
    irrigation     : str   = Field("Sprinkler", description="Canal/Drip/Sprinkler/Rainfed")
    region         : str   = Field("East", description="North/South/East/West/Central")
    previous_crop  : str   = Field("Rice")
    farm_size      : float = Field(2.0, ge=0.1, le=10000.0)


class ChatRequest(BaseModel):
    query  : str
    city   : Optional[str] = DEFAULT_CITY
    verbose: Optional[bool] = False


class AdvisoryRequest(BaseModel):
    inputs : FieldInputs
    verbose: Optional[bool] = False


class PredictRequest(BaseModel):
    inputs : FieldInputs


class AdvisoryResponse(BaseModel):
    crop                 : str
    crop_confidence      : float
    fertilizer           : str
    fertilizer_confidence: float
    pesticide            : str
    pesticide_target     : str
    pesticide_dose       : str
    advisory_text        : str
    source               : str
    latency_ms           : float
    timestamp            : str
    alternatives         : list


class PredictResponse(BaseModel):
    crop                 : str
    crop_confidence      : float
    fertilizer           : str
    fertilizer_confidence: float
    pesticide            : str
    pesticide_target     : str
    crop_top3            : list
    fertilizer_top3      : list
    mode                 : str
    timestamp            : str


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title       = "SageStorm V3 Agriculture AI API",
    description = "ML + SLM agricultural advisory system",
    version     = "3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

# ── System state ───────────────────────────────────────────────
_system     = None
_load_error = None


def _load_system():
    global _system, _load_error
    if _system is not None:
        return _system
    if _load_error is not None:
        raise RuntimeError(_load_error)

    try:
        from retrieval.vector_search    import load_retriever
        from memory.memory_manager      import MemoryManager
        from weather.weather_api        import WeatherService
        from rag.context_builder        import ContextBuilder
        from models.tokenizer           import SageTokenizer
        from models.agri_predictor      import AgriPredictor
        from chatbot.response_generator_v3 import ResponseGeneratorV3

        print("[API V3] Loading system components...")

        # Retriever
        try:
            ret, vocab = load_retriever(RETRIEVER_DIR)
        except Exception as e:
            print(f"  [!] Retriever not found: {e} — using dummy")
            class _DummyRet:
                def retrieve(self, q, top_k=5): return []
            ret, vocab = _DummyRet(), None

        memory  = MemoryManager()
        weather = WeatherService(city=DEFAULT_CITY,
                                  api_key=os.environ.get("OWM_API_KEY", ""))
        ctx     = ContextBuilder(ret, memory, weather, top_k=5)
        tok     = SageTokenizer()

        # SLM model (optional)
        model   = None
        backend = "retrieval+ml-rules"
        try:
            import torch
            if os.path.exists(FINETUNED_CKPT):
                from models.sagestorm_v2 import SageStormV2
                model   = SageStormV2.load(FINETUNED_CKPT, DEVICE)
                backend = f"SageStorm V2 ({model.param_count()['total_M']}M) + ML Advisory"
            else:
                print(f"  [!] Weights not found: {FINETUNED_CKPT}")
        except ImportError:
            print("  [!] PyTorch not installed")

        # ML Predictor
        predictor = AgriPredictor()
        if not predictor.trained:
            # Try training from default data paths
            soil_path = os.path.join("data", "soil_dataset.csv")
            npk_path  = os.path.join("data", "npk_dataset.csv")
            if os.path.exists(soil_path) or os.path.exists(npk_path):
                predictor.train(
                    soil_csv_path = soil_path if os.path.exists(soil_path) else None,
                    npk_csv_path  = npk_path  if os.path.exists(npk_path)  else None,
                )
            else:
                print("  [!] No training data found — using rule-based predictor")

        gen = ResponseGeneratorV3(
            model       = model,
            tokenizer   = tok,
            retriever   = ret,
            ctx_builder = ctx,
            predictor   = predictor,
            max_tokens  = GEN_MAX_TOKENS,
            temperature = GEN_TEMPERATURE,
        )

        _system = {
            "gen"      : gen,
            "memory"   : memory,
            "weather"  : weather,
            "predictor": predictor,
            "backend"  : backend,
        }
        print(f"[API V3] Ready — backend: {backend}")
        return _system

    except Exception as e:
        _load_error = str(e)
        print(f"[API V3] LOAD ERROR: {e}")
        raise


_stats = {
    "total_requests"  : 0,
    "advisory_requests": 0,
    "chat_requests"   : 0,
    "start_time"      : datetime.now().isoformat(),
}


# ── Endpoints ──────────────────────────────────────────────────
@app.get("/health")
def health():
    try:
        sys_ = _load_system()
        return {"status": "ok", "backend": sys_["backend"],
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.post("/advisory", response_model=AdvisoryResponse)
def advisory(req: AdvisoryRequest):
    """
    Full ML + SLM advisory endpoint.

    1. AgriPredictor runs on structured field inputs
    2. SageStorm narrates the predictions as a complete advisory
    """
    t0   = time.perf_counter()
    sys_ = _load_system()

    field_dict = req.inputs.dict()
    result = sys_["gen"].advisory_engine.generate_advisory(field_dict)
    pred   = result["predictions"]

    _stats["total_requests"]   += 1
    _stats["advisory_requests"] += 1

    alts = [{"crop": x.get("crop",""), "confidence": x.get("confidence", 0)}
            for x in pred.get("crop_top3", [])]

    return AdvisoryResponse(
        crop                  = pred["crop"],
        crop_confidence       = pred["crop_confidence"],
        fertilizer            = pred["fertilizer"],
        fertilizer_confidence = pred["fertilizer_confidence"],
        pesticide             = pred.get("pesticide", "N/A"),
        pesticide_target      = pred.get("pesticide_target", "N/A"),
        pesticide_dose        = pred.get("pesticide_dose", "as directed"),
        advisory_text         = result["advisory_text"],
        source                = result["source"],
        latency_ms            = round((time.perf_counter() - t0) * 1000, 1),
        timestamp             = datetime.now().isoformat(),
        alternatives          = alts,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """ML-only predictions — no SLM narration."""
    sys_  = _load_system()
    pred  = sys_["predictor"].predict(req.inputs.dict())
    _stats["total_requests"] += 1
    return PredictResponse(
        crop                  = pred["crop"],
        crop_confidence       = pred["crop_confidence"],
        fertilizer            = pred["fertilizer"],
        fertilizer_confidence = pred["fertilizer_confidence"],
        pesticide             = pred.get("pesticide", "N/A"),
        pesticide_target      = pred.get("pesticide_target", "N/A"),
        crop_top3             = pred.get("crop_top3", []),
        fertilizer_top3       = pred.get("fertilizer_top3", []),
        mode                  = pred.get("mode", "rule"),
        timestamp             = datetime.now().isoformat(),
    )


@app.post("/chat")
def chat(req: ChatRequest):
    """Standard chat endpoint — auto-routes to advisory pipeline when needed."""
    if not req.query.strip():
        raise HTTPException(400, "Query cannot be empty")

    t0   = time.perf_counter()
    sys_ = _load_system()

    sys_["memory"].process_input(req.query)

    if req.city and req.city != sys_["weather"].city:
        sys_["weather"].city = req.city

    answer, source = sys_["gen"].generate(req.query)
    sys_["memory"].add_response(answer)

    _stats["total_requests"] += 1
    _stats["chat_requests"]  += 1

    extra = {}
    if source == "advisory_ml":
        last = sys_["gen"].get_last_advisory()
        if last:
            pred = last["predictions"]
            extra = {
                "ml_predictions": {
                    "crop"                 : pred["crop"],
                    "crop_confidence"      : pred["crop_confidence"],
                    "fertilizer"           : pred["fertilizer"],
                    "fertilizer_confidence": pred["fertilizer_confidence"],
                    "pesticide"            : pred.get("pesticide"),
                }
            }

    return {
        "answer"    : answer,
        "source"    : source,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "timestamp" : datetime.now().isoformat(),
        **extra,
    }


@app.get("/advisory/last")
def get_last_advisory():
    """Returns the full structured result of the last advisory generation."""
    try:
        sys_  = _load_system()
        last  = sys_["gen"].get_last_advisory()
        if not last:
            return {"status": "none", "message": "No advisory generated yet"}
        # Filter out prompt_used to keep response lean
        return {k: v for k, v in last.items() if k != "prompt_used"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/stats")
def stats():
    elapsed = (datetime.now() -
               datetime.fromisoformat(_stats["start_time"])).total_seconds()
    return {**_stats, "uptime_seconds": elapsed}


@app.get("/profile")
def get_profile():
    try:
        sys_ = _load_system()
        return sys_["memory"].long.data
    except Exception:
        return {}


@app.post("/reset")
def reset():
    try:
        sys_ = _load_system()
        sys_["memory"].reset_session()
        return {"status": "session_cleared"}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/weather")
def get_weather(city: Optional[str] = None):
    try:
        sys_ = _load_system()
        w    = sys_["weather"].get(city)
        return w
    except Exception as e:
        return {"city": city or "Unknown", "error": str(e), "src": "error"}


# ── Run ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_v3:app", host="0.0.0.0", port=8000, reload=True)