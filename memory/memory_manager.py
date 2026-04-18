"""
memory/memory_manager.py
========================
Short-term conversation memory (last 5 turns) +
Long-term farmer profile (crop, location, soil, farm size).

FIXES:
  FIX 1: crop_type extraction uses word-boundary check (\b) instead of substring
          matching. Original bug: "rice" in "licorice" would incorrectly match.
  FIX 2: FarmerProfile.save() wrapped in try/except to handle disk-full gracefully
  FIX 3: location extractor patterns tightened to avoid matching "I", "My", "The"
          as location names (require at least 3 chars and not a stopword)
"""

import re
import json
import os
import sys
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FARMER_PROFILE, MEMORY_DIR

KNOWN_CROPS = {
    "rice", "wheat", "corn", "maize", "potato", "tomato", "onion", "garlic",
    "pepper", "chili", "eggplant", "cucumber", "mustard", "cauliflower",
    "cabbage", "broccoli", "bitter gourd", "loofah", "watermelon", "banana",
    "mango", "lemon", "coconut", "betel nut", "tea", "coffee", "sugarcane",
    "cotton", "soybean", "cowpea", "black bean", "ginger", "turmeric", "okra",
    "pumpkin", "gourd", "radish", "brinjal", "spinach", "pea", "bean",
    "groundnut", "sunflower", "jute",
}

# Common English stopwords that should not be captured as location names
_STOPWORDS = {
    "i", "my", "the", "a", "an", "is", "are", "was", "we", "our",
    "this", "that", "it", "he", "she", "they", "you", "in", "at",
    "on", "by", "for", "of", "to", "from",
}

_EXTRACTORS = {
    "crop_type": [
        r"\b(?:growing|farming|planting|cultivating)\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bmy\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:crop|field|farm)\b",
        r"\b(rice|wheat|maize|corn|potato|tomato|onion|garlic|pepper|chili|"
        r"eggplant|mustard|cauliflower|cabbage|banana|mango|lemon|coconut|"
        r"sugarcane|cotton|soybean|groundnut|sunflower|ginger|turmeric|okra|"
        r"pumpkin|radish|brinjal|spinach|pea|bean|cucumber|jute)\b",
    ],
    "location": [
        # FIX 3: Require at least 3 chars and not a stopword
        r"\b(?:in|near|at|from)\s+([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})?)\b",
        r"\b([A-Z][a-zA-Z]{2,})\s+(?:district|region|village|area|block|taluk)\b",
    ],
    "soil_type": [
        r"\b(sandy|clay|loamy|loam|red|black|alluvial|laterite)\s+soil\b",
        r"\bsoil\s+(?:type\s+is|is)\s+(sandy|clay|loamy|red|black|alluvial)\b",
    ],
    "farm_size": [
        r"(\d+(?:\.\d+)?)\s*(?:acre|bigha|hectare|ha|katha)s?\b",
    ],
    "name": [
        r"\b(?:my name is|i am|call me)\s+([A-Z][a-z]+)\b",
    ],
}


class FarmerProfile:
    def __init__(self, path: str = FARMER_PROFILE):
        self.path = path
        self.data = {
            k: None for k in
            ["name", "crop_type", "location", "soil_type", "farm_size", "updated"]
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
        """FIX 2: Wrapped in try/except to handle disk-full or permission errors."""
        try:
            with open(self.path, "w") as f:
                json.dump(self.data, f, indent=2)
        except OSError as e:
            print(f"[Memory] WARNING: Could not save profile: {e}")

    def update_from_text(self, text: str) -> dict:
        found = {}
        for field, patterns in _EXTRACTORS.items():
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().lower()

                    # FIX 1: For crop_type, use WORD BOUNDARY check in KNOWN_CROPS
                    # Original bug: "rice" in "licorice" == True (substring match)
                    # Fixed: check if the exact word val appears in KNOWN_CROPS
                    if field == "crop_type":
                        # Check the extracted value is a known crop (exact match)
                        matched_crop = None
                        for crop in KNOWN_CROPS:
                            if re.search(r'\b' + re.escape(crop) + r'\b', val):
                                matched_crop = crop
                                break
                        if not matched_crop:
                            continue
                        val = matched_crop

                    # FIX 3: For location, reject stopwords and very short strings
                    if field == "location":
                        if val.lower() in _STOPWORDS or len(val) < 3:
                            continue

                    if val and len(val) > 1:
                        self.data[field] = val
                        found[field]     = val
                        break

        if found:
            self.data["updated"] = datetime.now().isoformat()
            self.save()
        return found

    def as_text(self) -> str:
        lines = []
        if self.data.get("name"):       lines.append(f"Farmer: {self.data['name']}")
        if self.data.get("crop_type"):  lines.append(f"Crop: {self.data['crop_type']}")
        if self.data.get("location"):   lines.append(f"Location: {self.data['location']}")
        if self.data.get("soil_type"):  lines.append(f"Soil: {self.data['soil_type']}")
        if self.data.get("farm_size"):  lines.append(f"Farm: {self.data['farm_size']}")
        return "; ".join(lines) if lines else "No profile yet."

    def get(self, k, d=None):
        return self.data.get(k, d)


class ConversationMemory:
    def __init__(self, max_turns: int = 5):
        self.turns = deque(maxlen=max_turns * 2)

    def add(self, role: str, text: str):
        self.turns.append({
            "role": role,
            "text": text,
            "time": datetime.now().strftime("%H:%M"),
        })

    def as_text(self) -> str:
        if not self.turns:
            return "No history."
        return "\n".join(
            f"[{t['time']}] {'User' if t['role'] == 'user' else 'Bot'}: {t['text']}"
            for t in self.turns
        )

    def last_user(self) -> str:
        for t in reversed(self.turns):
            if t["role"] == "user":
                return t["text"]
        return ""

    def clear(self):
        self.turns.clear()


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
        return (f"[Farmer Profile]\n{self.long.as_text()}\n\n"
                f"[Recent Conversation]\n{self.short.as_text()}")

    def get_crop(self):
        return self.long.get("crop_type")

    def get_location(self):
        return self.long.get("location")

    def reset_session(self):
        self.short.clear()
        print("[Memory] Session cleared. Profile preserved.")