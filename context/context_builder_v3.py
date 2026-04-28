"""
context/context_builder_v3.py
===============================
ContextBuilderV3 — Structured Feature Extraction

Extracts the structured input dictionary that feeds all three decision
models from:
  1. Free-text user message (regex + keyword extraction)
  2. Farmer long-term memory profile
  3. Live weather data
  4. Explicit key=value form fields (for UI advisory mode)

Output dict keys (all optional — models handle missing values):
  location, soil_type, season, temperature, humidity,
  nitrogen, phosphorus, potassium, growth_stage,
  symptoms, crop_name, month, query
"""

import re
import os
import sys
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Soil type keywords ────────────────────────────────────────
_SOIL_RE = re.compile(
    r"\b(loamy?|sandy|clay(?:ey)?|black|red|alluvial|laterite|silty|"
    r"alkaline|acidic|peaty)\b", re.I
)

# ── Season keywords ───────────────────────────────────────────
_SEASON_RE = re.compile(r"\b(kharif|rabi|zaid|summer|winter|monsoon)\b", re.I)
_SEASON_MAP = {
    "summer": "zaid", "winter": "rabi", "monsoon": "kharif",
    "kharif": "kharif", "rabi": "rabi", "zaid": "zaid",
}

# ── NPK extraction ────────────────────────────────────────────
_N_RE   = re.compile(r"\bn(?:itrogen)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.I)
_P_RE   = re.compile(r"\bp(?:hosphorus|hosphate)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.I)
_K_RE   = re.compile(r"\bk(?:otash|potassium)?\s*[:=]?\s*(\d+(?:\.\d+)?)\b", re.I)

# ── Temperature extraction ────────────────────────────────────
_TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*[cC](?:elsius)?\b")

# ── Humidity extraction ───────────────────────────────────────
_HUMID_RE = re.compile(
    r"(?:humidity|humid|rh)[^0-9]*(\d+(?:\.\d+)?)"   # "humidity 75" or "humid:75"
    r"|(\d+(?:\.\d+)?)\s*%\s*(?:humidity|humid|rh)",  # "75% humidity"
    re.I
)

# ── Location extraction ───────────────────────────────────────
_LOC_RE = re.compile(
    r"\b(?:in|near|at|from|located\s+in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"
)

# ── Crop extraction (comprehensive Indian crop list) ──────────
_CROPS = [
    "rice","paddy","wheat","maize","corn","sorghum","jowar","bajra","millet",
    "barley","oats","ragi","chickpea","gram","lentil","cowpea","moong","urad",
    "arhar","pigeon pea","blackgram","greengram","rajma","soybean","mustard",
    "rapeseed","sunflower","groundnut","peanut","sesame","linseed","castor",
    "potato","tomato","onion","garlic","chilli","chili","pepper","brinjal",
    "eggplant","cauliflower","cabbage","okra","bhindi","cucumber","pumpkin",
    "gourd","radish","carrot","spinach","pea","bean","capsicum","mango",
    "banana","lemon","orange","papaya","guava","coconut","watermelon",
    "muskmelon","grapes","apple","pomegranate","cotton","sugarcane","jute",
    "tea","coffee","tobacco","rubber","arecanut","cashew","cardamom",
    "turmeric","ginger","tapioca",
]
_CROP_RE = re.compile(
    r"\b(" + "|".join(sorted(_CROPS, key=len, reverse=True)) + r")\b", re.I
)

# ── Growth stage extraction ───────────────────────────────────
_STAGE_RE = re.compile(
    r"\b(seedling|vegetative|flowering|fruiting|tillering|heading|"
    r"panicle|maturity|harvest|transplanting|germination|basal|"
    r"booting|boot\s+leaf|grand\s+growth|square|boll)\b", re.I
)

# ── Symptom / pest keywords ───────────────────────────────────
_SYMPTOM_RE = re.compile(
    r"\b(yellowing?|wilting?|rotting?|blight|spots?|rust|mildew|aphid|borer|"
    r"caterpillar|worm|fly|beetle|mite|thrip|whitefly|hopper|"
    r"dead\s+heart|white\s+ear|leaf\s+curl|mosaic|lesion|discolor)\w*", re.I
)


def _extract_from_text(text: str) -> Dict:
    """Extract structured features from free-text input."""
    ctx = {}

    # Soil
    m = _SOIL_RE.search(text)
    if m:
        soil = m.group(1).lower().rstrip("y")
        ctx["soil_type"] = soil if soil != "loam" else "loamy"

    # Season
    m = _SEASON_RE.search(text)
    if m:
        ctx["season"] = _SEASON_MAP.get(m.group(1).lower(), m.group(1).lower())

    # Temperature
    m = _TEMP_RE.search(text)
    if m:
        ctx["temperature"] = float(m.group(1))

    # Humidity
    m = _HUMID_RE.search(text)
    if m:
        ctx["humidity"] = float(m.group(1) or m.group(2))

    # NPK
    m = _N_RE.search(text)
    if m:
        ctx["nitrogen"] = float(m.group(1))
    m = _P_RE.search(text)
    if m:
        ctx["phosphorus"] = float(m.group(1))
    m = _K_RE.search(text)
    if m:
        ctx["potassium"] = float(m.group(1))

    # Location
    m = _LOC_RE.search(text)
    if m:
        ctx["location"] = m.group(1).strip()

    # Crop
    m = _CROP_RE.search(text)
    if m:
        ctx["crop_name"] = m.group(1).lower().title()

    # Growth stage
    m = _STAGE_RE.search(text)
    if m:
        ctx["growth_stage"] = m.group(1).lower()

    # Symptoms (collect all hits)
    syms = _SYMPTOM_RE.findall(text)
    if syms:
        ctx["symptoms"] = " ".join(set(s.lower() for s in syms))

    return ctx


def _extract_from_profile(profile_data: Dict) -> Dict:
    """Extract relevant fields from farmer long-term memory."""
    ctx = {}
    field_map = {
        "crop_type":  "crop_name",
        "location":   "location",
        "soil_type":  "soil_type",
        "farm_size":  "farm_size",
    }
    for src_key, dst_key in field_map.items():
        val = profile_data.get(src_key)
        if val:
            ctx[dst_key] = val
    return ctx


def _season_from_month(month: int) -> str:
    if 6 <= month <= 10:
        return "kharif"
    elif month in (11, 12, 1, 2, 3):
        return "rabi"
    return "zaid"


class ContextBuilderV3:
    """
    Builds the structured feature dict for the advisory pipeline.

    Priority for each field (highest wins):
      1. Explicit form fields (UI advisory mode)
      2. Free-text extraction from query
      3. Farmer memory profile
      4. Weather API data
      5. Calendar-based defaults
    """

    def __init__(self, memory=None, weather_svc=None):
        self.memory      = memory
        self.weather_svc = weather_svc

    def build(self, query: str, explicit_fields: Optional[Dict] = None) -> Dict:
        """
        Build the full context dict.

        explicit_fields: dict from the UI form (highest priority)
        query: free-text user message
        """
        ctx = {}

        # Step 1: Calendar defaults
        now = datetime.now()
        ctx["month"]  = now.month
        ctx["season"] = _season_from_month(now.month)

        # Step 2: Farmer memory profile (lowest priority base)
        if self.memory:
            profile_ctx = _extract_from_profile(self.memory.long.data)
            ctx.update({k: v for k, v in profile_ctx.items() if v})

        # Step 3: Extract from free text
        if query:
            text_ctx = _extract_from_text(query)
            ctx.update({k: v for k, v in text_ctx.items() if v})

        # Step 4: Weather data
        if self.weather_svc:
            try:
                loc = ctx.get("location") or (
                    self.memory.get_location() if self.memory else None
                )
                w = self.weather_svc.get(loc)
                if w:
                    # Only fill if not already extracted from text
                    if "temperature" not in ctx:
                        ctx["temperature"] = float(w.get("temp", 28))
                    if "humidity" not in ctx:
                        ctx["humidity"] = float(w.get("humid", 65))
                    ctx["weather"] = w
            except Exception:
                pass

        # Step 5: Explicit form fields override everything
        if explicit_fields:
            for k, v in explicit_fields.items():
                if v is not None and v != "" and v != 0:
                    ctx[k] = v

        # Step 6: Numeric defaults for missing NPK
        ctx.setdefault("temperature", 28.0)
        ctx.setdefault("humidity",    65.0)
        ctx.setdefault("nitrogen",    50.0)
        ctx.setdefault("phosphorus",  30.0)
        ctx.setdefault("potassium",   30.0)
        ctx.setdefault("soil_type",   "loamy")
        ctx.setdefault("location",    "")
        ctx.setdefault("crop_name",   "")
        ctx.setdefault("symptoms",    "")
        ctx.setdefault("growth_stage", "")
        ctx["query"] = query

        return ctx

    def describe(self, ctx: Dict) -> str:
        """Human-readable summary of extracted context (for debugging)."""
        parts = []
        if ctx.get("location"):    parts.append(f"Location: {ctx['location']}")
        if ctx.get("soil_type"):   parts.append(f"Soil: {ctx['soil_type']}")
        if ctx.get("season"):      parts.append(f"Season: {ctx['season']}")
        if ctx.get("crop_name"):   parts.append(f"Crop: {ctx['crop_name']}")
        if ctx.get("temperature"): parts.append(f"Temp: {ctx['temperature']}°C")
        if ctx.get("nitrogen"):    parts.append(f"N: {ctx['nitrogen']}")
        if ctx.get("phosphorus"):  parts.append(f"P: {ctx['phosphorus']}")
        if ctx.get("potassium"):   parts.append(f"K: {ctx['potassium']}")
        if ctx.get("symptoms"):    parts.append(f"Symptoms: {ctx['symptoms']}")
        return " | ".join(parts) if parts else "No context extracted"


if __name__ == "__main__":
    cb = ContextBuilderV3()

    tests = [
        "I am growing rice near sivakasi. My soil is loamy and temperature is 32°C. "
        "I see stem borer in my field. N=85 P=40 K=35.",
        "My potato crop has late blight. Season is rabi. Black soil.",
        "What fertilizer should I use for my cotton in kharif?",
    ]
    for t in tests:
        ctx = cb.build(t)
        print(f"\nInput : {t[:60]}...")
        print(f"Output: {cb.describe(ctx)}")
