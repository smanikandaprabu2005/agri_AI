"""
memory/memory_manager.py
========================
Short-term conversation memory (last 5 turns) +
Long-term farmer profile (crop, location, soil, farm size).
"""

import re, json, os, sys
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FARMER_PROFILE, MEMORY_DIR

KNOWN_CROPS = {
    "rice","wheat","corn","maize","potato","tomato","onion","garlic","pepper",
    "chili","eggplant","cucumber","mustard","cauliflower","cabbage","broccoli",
    "bitter gourd","loofah","watermelon","banana","mango","lemon","coconut",
    "betel nut","tea","coffee","sugarcane","cotton","soybean","cowpea",
    "black bean","ginger","turmeric","okra","pumpkin","gourd","radish",
    "brinjal","spinach","pea","bean","groundnut","sunflower","jute",
}

_EXTRACTORS = {
    "crop_type": [
        r"\b(?:growing|farming|planting|cultivating)\s+([a-z]+(?:\s+[a-z]+)?)\b",
        r"\bmy\s+([a-z]+(?:\s+[a-z]+)?)\s+(?:crop|field|farm)\b",
    ],
    "location": [
        r"\b(?:in|near|at|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
        r"\b([A-Z][a-z]+)\s+(?:district|region|village|area)\b",
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
    def __init__(self, path=FARMER_PROFILE):
        self.path = path
        self.data = {k: None for k in ["name","crop_type","location","soil_type","farm_size","updated"]}
        os.makedirs(MEMORY_DIR, exist_ok=True)
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f: self.data.update(json.load(f))
            except: pass

    def save(self):
        with open(self.path, "w") as f: json.dump(self.data, f, indent=2)

    def update_from_text(self, text):
        found = {}
        for field, patterns in _EXTRACTORS.items():
            for pat in patterns:
                m = re.search(pat, text, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().lower()
                    if field == "crop_type" and not any(c in val for c in KNOWN_CROPS): continue
                    if val and len(val) > 1:
                        self.data[field] = val
                        found[field] = val
                        break
        if found:
            self.data["updated"] = datetime.now().isoformat()
            self.save()
        return found

    def as_text(self):
        lines = []
        if self.data.get("name"):      lines.append(f"Farmer: {self.data['name']}")
        if self.data.get("crop_type"): lines.append(f"Crop: {self.data['crop_type']}")
        if self.data.get("location"):  lines.append(f"Location: {self.data['location']}")
        if self.data.get("soil_type"): lines.append(f"Soil: {self.data['soil_type']}")
        if self.data.get("farm_size"): lines.append(f"Farm: {self.data['farm_size']}")
        return "; ".join(lines) if lines else "No profile yet."

    def get(self, k, d=None): return self.data.get(k, d)


class ConversationMemory:
    def __init__(self, max_turns=5):
        self.turns = deque(maxlen=max_turns * 2)

    def add(self, role, text):
        self.turns.append({"role": role, "text": text,
                           "time": datetime.now().strftime("%H:%M")})

    def as_text(self):
        if not self.turns: return "No history."
        return "\n".join(f"[{t['time']}] {'User' if t['role']=='user' else 'Bot'}: {t['text']}"
                         for t in self.turns)

    def last_user(self):
        for t in reversed(self.turns):
            if t["role"] == "user": return t["text"]
        return ""

    def clear(self): self.turns.clear()


class MemoryManager:
    def __init__(self, max_turns=5):
        self.short = ConversationMemory(max_turns)
        self.long  = FarmerProfile()

    def process_input(self, text):
        self.short.add("user", text)
        updated = self.long.update_from_text(text)
        if updated: print(f"[Memory] Profile updated: {updated}")

    def add_response(self, text):
        self.short.add("bot", text)

    def context_text(self):
        return (f"[Farmer Profile]\n{self.long.as_text()}\n\n"
                f"[Recent Conversation]\n{self.short.as_text()}")

    def get_crop(self): return self.long.get("crop_type")
    def get_location(self): return self.long.get("location")

    def reset_session(self):
        self.short.clear()
        print("[Memory] Session cleared. Profile preserved.")
