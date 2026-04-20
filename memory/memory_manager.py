"""
memory/memory_manager.py
========================
Enhanced Memory Manager v2.1

Improvements over v2:
  + Richer extraction — 50+ crop names, state names, soil types
  + Number normalisation (acres, bighas, hectares all unified)
  + Context injection into RAG (memory is now a RAG signal)
  + Profile confidence tracking (how certain we are)
  + Session context window for multi-turn reasoning
"""

import re, json, os, sys
from datetime import datetime
from collections import deque
from typing import Optional, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FARMER_PROFILE, MEMORY_DIR

# ── Comprehensive crop list ───────────────────────────────────
KNOWN_CROPS = {
    # Cereals
    "rice","paddy","wheat","maize","corn","sorghum","jowar","bajra","millet",
    "barley","oats","ragi","finger millet",
    # Pulses
    "chickpea","gram","lentil","cowpea","moong","urad","arhar","pigeon pea",
    "blackgram","greengram","rajma","soybean","soya",
    # Oilseeds
    "mustard","rapeseed","sunflower","groundnut","peanut","sesame","linseed",
    "castor","safflower","nigerseed",
    # Vegetables
    "potato","tomato","onion","garlic","chilli","pepper","brinjal","eggplant",
    "cauliflower","cabbage","okra","bhindi","cucumber","pumpkin","gourd",
    "radish","carrot","spinach","pea","bean","french bean","bitter gourd",
    "bottle gourd","ridge gourd","capsicum","chili",
    # Fruits
    "mango","banana","lemon","orange","papaya","guava","coconut","watermelon",
    "muskmelon","grapes","apple","pomegranate","litchi","jackfruit","pineapple",
    # Cash crops
    "cotton","sugarcane","jute","tea","coffee","tobacco","rubber",
    "arecanut","cashew","cardamom","turmeric","ginger","tapioca","cassava",
    # Other
    "moringa","bamboo","teak","eucalyptus","sandalwood","poplar",
}

# ── Indian states / UTs ───────────────────────────────────────
KNOWN_STATES = {
    "assam","kerala","karnataka","tamil nadu","andhra pradesh","telangana",
    "maharashtra","gujarat","rajasthan","madhya pradesh","uttar pradesh",
    "bihar","west bengal","odisha","jharkhand","chhattisgarh","punjab",
    "haryana","himachal pradesh","uttarakhand","delhi","goa","manipur",
    "meghalaya","mizoram","nagaland","tripura","arunachal pradesh","sikkim",
    "jammu and kashmir","ladakh","chandigarh","puducherry",
}

# ── Soil type keywords ────────────────────────────────────────
SOIL_TYPES = {
    "sandy", "clay", "loamy", "loam", "red", "black", "alluvial",
    "laterite", "silty", "peaty", "chalky", "saline", "alkaline",
}


# ── Extractor patterns ────────────────────────────────────────
def _build_crop_pattern() -> re.Pattern:
    escaped = sorted([re.escape(c) for c in KNOWN_CROPS], key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.I)

def _build_state_pattern() -> re.Pattern:
    escaped = sorted([re.escape(s) for s in KNOWN_STATES], key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.I)

_CROP_RE  = _build_crop_pattern()
_STATE_RE = _build_state_pattern()

_EXTRACTORS = {
    "crop_type": [
        _CROP_RE,
        re.compile(r"\b(?:growing|farming|planting|cultivating|sowing)\s+([\w\s]+?)(?:\s+crop|\s+in|\s+near|$)", re.I),
        re.compile(r"\bmy\s+([\w\s]+?)\s+(?:crop|field|farm|plants?)\b", re.I),
    ],
    "location": [
        _STATE_RE,
        re.compile(r"\b(?:in|near|at|from|district\s+of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b"),
        re.compile(r"\b([A-Z][a-z]+)\s+(?:district|village|taluk|block|mandal|tehsil)\b"),
    ],
    "soil_type": [
        re.compile(
            r"\b(" + "|".join(SOIL_TYPES) + r")\s+(?:soil|land|earth)\b", re.I),
        re.compile(r"\bsoil\s+(?:type\s+is|is)\s+(" + "|".join(SOIL_TYPES) + r")\b", re.I),
    ],
    "farm_size": [
        re.compile(r"(\d+(?:\.\d+)?)\s*(?:acre|bigha|hectare|ha|katha|gunta)s?\b", re.I),
    ],
    "name": [
        re.compile(r"\b(?:my name is|i am|i'm|call me|this is)\s+([A-Z][a-z]+)\b"),
        re.compile(r"\bname\s*[:—]\s*([A-Z][a-z]+)\b"),
    ],
}

# Unit conversion to acres (for uniform storage)
_UNIT_TO_ACRES = {
    "acre": 1.0, "acres": 1.0,
    "hectare": 2.471, "hectares": 2.471, "ha": 2.471,
    "bigha": 0.619,  # varies by state — using UP bigha as default
    "katha": 0.033,
    "gunta": 0.025,
}


def _normalise_farm_size(text: str) -> Optional[str]:
    """Extract and normalise farm size to acres."""
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(acre|bigha|hectare|ha|katha|gunta)s?", text, re.I)
    if not m:
        return None
    amount = float(m.group(1))
    unit   = m.group(2).lower().rstrip("s")
    acres  = amount * _UNIT_TO_ACRES.get(unit, 1.0)
    if acres < 0.1 or acres > 10000:   # sanity check
        return None
    return f"{amount} {unit}s ({acres:.1f} acres)"


# ══════════════════════════════════════════════════════════════
#  Farmer Profile
# ══════════════════════════════════════════════════════════════
class FarmerProfile:
    def __init__(self, path: str = FARMER_PROFILE):
        self.path       = path
        self.data: Dict = {
            "name": None, "crop_type": None, "location": None,
            "soil_type": None, "farm_size": None, "updated": None,
            "confidence": {},   # field -> score
        }
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.data.update(json.load(f))
            except Exception:
                pass

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def update_from_text(self, text: str) -> Dict[str, str]:
        found = {}

        # Crop — use comprehensive crop list
        m = _CROP_RE.search(text)
        if m:
            crop = m.group(1).lower()
            self.data["crop_type"] = crop
            self.data["confidence"]["crop_type"] = 0.95
            found["crop_type"] = crop

        # Location — try state first, then city pattern
        for pat in _EXTRACTORS["location"]:
            m = pat.search(text)
            if m:
                val = m.group(1).strip().lower()
                if len(val) > 2:
                    self.data["location"] = val
                    self.data["confidence"]["location"] = 0.85
                    found["location"] = val
                    break

        # Soil type
        for pat in _EXTRACTORS["soil_type"]:
            m = pat.search(text)
            if m:
                soil = m.group(1).lower()
                self.data["soil_type"] = soil + " soil"
                self.data["confidence"]["soil_type"] = 0.9
                found["soil_type"] = soil
                break

        # Farm size with unit normalisation
        size = _normalise_farm_size(text)
        if size:
            self.data["farm_size"] = size
            self.data["confidence"]["farm_size"] = 0.9
            found["farm_size"] = size

        # Name
        for pat in _EXTRACTORS["name"]:
            m = pat.search(text)
            if m:
                name = m.group(1).strip()
                if 2 <= len(name) <= 30:
                    self.data["name"] = name
                    self.data["confidence"]["name"] = 0.9
                    found["name"] = name
                    break

        if found:
            self.data["updated"] = datetime.now().isoformat()
            self.save()

        return found

    def as_text(self) -> str:
        parts = []
        if self.data.get("name"):
            parts.append(f"Farmer: {self.data['name']}")
        if self.data.get("crop_type"):
            parts.append(f"Crop: {self.data['crop_type']}")
        if self.data.get("location"):
            parts.append(f"Location: {self.data['location']}")
        if self.data.get("soil_type"):
            parts.append(f"Soil: {self.data['soil_type']}")
        if self.data.get("farm_size"):
            parts.append(f"Farm: {self.data['farm_size']}")
        return "; ".join(parts) if parts else "No profile yet."

    def as_rag_context(self) -> str:
        """Compact string for injecting into RAG prompt."""
        parts = []
        if self.data.get("crop_type"):
            parts.append(f"crop={self.data['crop_type']}")
        if self.data.get("location"):
            parts.append(f"location={self.data['location']}")
        if self.data.get("soil_type"):
            parts.append(f"soil={self.data['soil_type']}")
        return ", ".join(parts) if parts else ""

    def get(self, k: str, d=None):
        return self.data.get(k, d)


# ══════════════════════════════════════════════════════════════
#  Conversation Memory
# ══════════════════════════════════════════════════════════════
class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.turns = deque(maxlen=max_turns * 2)

    def add(self, role: str, text: str):
        self.turns.append({
            "role": role, "text": text,
            "time": datetime.now().strftime("%H:%M"),
        })

    def as_text(self) -> str:
        if not self.turns:
            return ""
        return "\n".join(
            f"{'User' if t['role'] == 'user' else 'Bot'}: {t['text']}"
            for t in self.turns
        )

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["text"]
        return ""

    def recent_topics(self) -> str:
        """Extract a short topic summary from recent turns."""
        texts = [t["text"] for t in list(self.turns)[-6:] if t["role"] == "user"]
        return " | ".join(texts[-3:]) if texts else ""

    def clear(self):
        self.turns.clear()


# ══════════════════════════════════════════════════════════════
#  Memory Manager
# ══════════════════════════════════════════════════════════════
class MemoryManager:
    def __init__(self, max_turns: int = 5):
        self.short = ConversationMemory(max_turns)
        self.long  = FarmerProfile()

    def process_input(self, text: str):
        self.short.add("user", text)
        updated = self.long.update_from_text(text)
        if updated:
            print(f"[Memory] Profile updated: {updated}")

    def add_response(self, text: str):
        self.short.add("bot", text)

    def context_text(self) -> str:
        parts = []
        profile = self.long.as_text()
        if profile and profile != "No profile yet.":
            parts.append(f"[Farmer Profile]\n{profile}")
        history = self.short.as_text()
        if history:
            parts.append(f"[Recent turns]\n{history}")
        return "\n\n".join(parts)

    def get_crop(self) -> Optional[str]:
        return self.long.get("crop_type")

    def get_location(self) -> Optional[str]:
        return self.long.get("location")

    def reset_session(self):
        self.short.clear()
        print("[Memory] Session cleared. Profile preserved.")