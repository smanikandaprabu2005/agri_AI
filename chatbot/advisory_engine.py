"""
chatbot/advisory_engine.py
===========================
SageStorm Advisory Engine
==========================
This is the CORE integration layer that connects:

  1. AgriPredictor (ML)        — predicts crop / fertilizer / pesticide
  2. ContextBuilder (RAG)      — retrieves relevant agricultural knowledge
  3. SageStorm SLM             — generates natural language explanation
  4. MemoryManager             — stores farmer profile across turns

Flow:
  User input (soil params)
       │
       ▼
  AgriPredictor.predict()      → structured predictions dict
       │
       ▼
  ContextBuilder.build_context_str()  → RAG passages
       │
       ▼
  build_slm_explanation_prompt()      → structured SLM prompt
       │
       ▼
  SageStorm.generate()         → natural language advisory
       │
       ▼
  AdvisoryResponse             → returned to user / API
"""

import os
import sys
import re
import json
import time
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.agri_predictor import AgriPredictor, build_slm_explanation_prompt
from config import DEVICE, GEN_MAX_TOKENS, GEN_TEMPERATURE


# ── Section markers for structured SLM output ───────────────
SECTION_MARKERS = [
    "crop recommendation",
    "land preparation",
    "sowing",
    "fertilizer",
    "irrigation",
    "weed",
    "pest",
    "harvest",
    "post-harvest",
    "storage",
]


def extract_sections(text: str) -> Dict[str, str]:
    """
    Parse SLM output into named sections.
    SageStorm is prompted to produce numbered sections;
    this parser extracts them for structured display.
    """
    sections = {}
    current_key = "introduction"
    current_lines = []

    for line in text.split("\n"):
        line_lower = line.lower().strip()
        matched = False
        for marker in SECTION_MARKERS:
            if marker in line_lower and (line.strip().startswith(("1", "2", "3", "4",
                                         "5", "6", "7", "8", "9", "#", "**", "–", "-"))):
                if current_lines:
                    sections[current_key] = "\n".join(current_lines).strip()
                current_key = marker
                current_lines = [line]
                matched = True
                break
        if not matched:
            current_lines.append(line)

    if current_lines:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def clean_advisory_output(text: str) -> str:
    """Strip prompt artifacts from SLM output."""
    # Remove prompt markers if they leaked through
    for marker in ["### Instruction:", "### Input:", "### Response:",
                   "### Context:", "<unk>", "<pad>", "<bos>", "<eos>"]:
        text = text.replace(marker, "")

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Fix common OCR/tokenizer artifacts
    text = text.replace("\\n", "\n").strip()

    # Capitalise first letter
    return (text[0].upper() + text[1:]) if text else text


def is_advisory_query(text: str) -> bool:
    """
    Detect if user is asking for an agricultural advisory
    (as opposed to a simple Q&A question).
    """
    advisory_kw = re.compile(
        r"\b(soil|ph|nitrogen|phosphorus|potassium|n:|p:|k:|npk|"
        r"season|kharif|rabi|zaid|loamy|clay|sandy|silt|"
        r"advisory|recommend|fertilizer|fertiliser|crop suggest|"
        r"what crop|which crop|should i grow|farm size|acres|hectare|"
        r"temperature.*humidity|humidity.*rainfall)\b",
        re.I
    )
    return bool(advisory_kw.search(text))


def parse_advisory_inputs(text: str) -> Optional[Dict]:
    """
    Try to extract structured inputs from free-text user message.
    
    e.g. "My soil is clay, pH 6.5, N=90, P=40, K=50, season Kharif"
    """
    inp = {}

    # Soil type
    soil_m = re.search(r"\b(clay|loamy|sandy|silt|red|black|alluvial)\b", text, re.I)
    if soil_m:
        inp["soil_type"] = soil_m.group(1).capitalize()

    # pH
    ph_m = re.search(r"\bph\s*[=:]\s*([\d.]+)", text, re.I)
    if ph_m:
        inp["ph"] = float(ph_m.group(1))

    # N, P, K
    for nutrient in ["N", "P", "K"]:
        m = re.search(rf"\b{nutrient}\s*[=:]\s*(\d+)", text, re.I)
        if m:
            inp[nutrient] = int(m.group(1))

    # NPK pattern "90-40-50" or "90:40:50"
    npk_m = re.search(r"(\d+)[:\-](\d+)[:\-](\d+)", text)
    if npk_m and "N" not in inp:
        inp["N"], inp["P"], inp["K"] = (int(npk_m.group(1)),
                                         int(npk_m.group(2)),
                                         int(npk_m.group(3)))

    # Temperature
    temp_m = re.search(r"(\d+)\s*°?\s*c\b", text, re.I)
    if temp_m:
        inp["temperature"] = float(temp_m.group(1))

    # Humidity
    hum_m = re.search(r"humidity\s*[=:]?\s*(\d+)", text, re.I)
    if hum_m:
        inp["humidity"] = float(hum_m.group(1))

    # Rainfall
    rain_m = re.search(r"(\d+)\s*mm", text, re.I)
    if rain_m:
        inp["rainfall"] = float(rain_m.group(1))

    # Season
    season_m = re.search(r"\b(kharif|rabi|zaid)\b", text, re.I)
    if season_m:
        inp["season"] = season_m.group(1).capitalize()

    # Farm size
    size_m = re.search(r"(\d+(?:\.\d+)?)\s*(?:acre|bigha|hectare)", text, re.I)
    if size_m:
        inp["farm_size"] = float(size_m.group(1))

    return inp if len(inp) >= 3 else None


# ════════════════════════════════════════════════════════════
#  Advisory Engine
# ════════════════════════════════════════════════════════════
class AdvisoryEngine:
    """
    Integrates ML predictions + RAG + SageStorm SLM.

    The SLM is given:
      - Structured ML prediction facts (crop, fertilizer, pesticide)
      - Farmer's actual field measurements (soil, NPK, climate)
      - RAG passages from ICAR books / knowledge base
      - A structured prompt instructing it to EXPLAIN and NARRATE

    The SLM's job: produce a complete, human-readable advisory
    as if it were an expert agronomist.
    """

    # Structured advisory prompt — forces SageStorm to cover all 10 sections
    ADVISORY_PROMPT_TEMPLATE = """\
### Instruction:
You are SageStorm, an experienced agricultural advisory AI trained on Indian \
farming practices. The ML prediction system has analysed the farmer's field \
data and made predictions. Your task is to EXPLAIN these predictions and \
provide a COMPLETE farming advisory from land preparation to post-harvest.

Be practical, specific, and farmer-friendly. Avoid jargon. Structure your \
response with these sections:
1. Why this crop was recommended
2. Land preparation steps
3. Sowing and planting guide
4. Fertilizer application plan
5. Irrigation schedule
6. Weed management
7. Pest and disease management
8. Harvesting guide
9. Post-harvest and storage

### Input:
[ML System Predictions]
Crop Recommended  : {crop} (confidence: {crop_conf}%)
Fertilizer        : {fert} (confidence: {fert_conf}%)
Pesticide needed  : {pest} for {pest_target}
Pesticide dose    : {pest_dose}
Alternative crops : {alternatives}

[Farmer Field Data]
Location          : {region} India, {season} season
Soil type         : {soil}
Soil pH           : {ph} {ph_note}
Nitrogen (N)      : {N} kg/ha {n_note}
Phosphorus (P)    : {P} kg/ha {p_note}
Potassium (K)     : {K} kg/ha {k_note}
Temperature       : {temp}°C
Humidity          : {humidity}% {humid_note}
Rainfall          : {rainfall} mm
Irrigation        : {irrigation}
Farm size         : {farm_size} acres
Previous crop     : {prev_crop}

[Agricultural Knowledge Base]
{rag_context}

### Response:
"""

    def __init__(self, slm_model, tokenizer, retriever, ctx_builder,
                 predictor: AgriPredictor = None,
                 max_tokens: int = 400,
                 temperature: float = 0.55):
        self.model       = slm_model
        self.tokenizer   = tokenizer
        self.retriever   = retriever
        self.ctx_builder = ctx_builder
        self.predictor   = predictor or AgriPredictor()
        self.max_tokens  = max_tokens
        self.temperature = temperature

    def _build_advisory_prompt(self, predictions: Dict, rag_context: str) -> str:
        """Build the full structured prompt for SageStorm."""
        inp = predictions.get("inputs", {})

        # Top-3 alternatives
        alts = []
        for item in predictions.get("crop_top3", [])[1:3]:
            c = item.get("crop", "")
            conf = item.get("confidence", 0)
            if isinstance(conf, float) and conf < 1:
                conf = round(conf * 100, 1)
            alts.append(f"{c} ({conf}%)")
        alternatives = ", ".join(alts) if alts else "N/A"

        N  = inp.get("N", 80)
        P  = inp.get("P", 40)
        K  = inp.get("K", 40)
        ph = inp.get("ph", 6.5)
        humidity = inp.get("humidity", 65)

        return self.ADVISORY_PROMPT_TEMPLATE.format(
            crop         = predictions["crop"],
            crop_conf    = predictions["crop_confidence"],
            fert         = predictions["fertilizer"],
            fert_conf    = predictions["fertilizer_confidence"],
            pest         = predictions.get("pesticide", "N/A"),
            pest_target  = predictions.get("pesticide_target", "pests"),
            pest_dose    = predictions.get("pesticide_dose", "as directed"),
            alternatives = alternatives,
            region       = inp.get("region", "East"),
            season       = inp.get("season", "Kharif"),
            soil         = inp.get("soil_type", "Loamy"),
            ph           = ph,
            ph_note      = ("(acidic)" if ph < 5.8 else "(alkaline)" if ph > 7.8 else "(optimal)"),
            N            = N,
            n_note       = ("⚠ low" if N < 60 else "✓ ok"),
            P            = P,
            p_note       = ("⚠ deficient" if P < 30 else "✓ ok"),
            K            = K,
            k_note       = ("⚠ low" if K < 40 else "✓ ok"),
            temp         = inp.get("temperature", 25),
            humidity     = humidity,
            humid_note   = ("⚠ fungal risk" if humidity > 75 else ""),
            rainfall     = inp.get("rainfall", 100),
            irrigation   = inp.get("irrigation", "Sprinkler"),
            farm_size    = inp.get("farm_size", 2),
            prev_crop    = inp.get("previous_crop", "Not specified"),
            rag_context  = rag_context[:700] if rag_context else "General Indian agriculture best practices apply.",
        )

    def _slm_generate(self, prompt: str) -> str:
        """Feed prompt to SageStorm and decode the output."""
        if self.model is None:
            return ""
        try:
            ids     = self.tokenizer.encode_prompt(prompt)
            gen_ids = self.model.generate(
                ids,
                max_tokens  = self.max_tokens,
                temperature = self.temperature,
                device      = DEVICE,
            )
            raw = self.tokenizer.decode(gen_ids, skip_special=True)
            return clean_advisory_output(raw)
        except Exception as e:
            print(f"[AdvisoryEngine] SLM generation error: {e}")
            return ""

    def _fallback_advisory(self, predictions: Dict) -> str:
        """
        Rule-based advisory text when SLM is unavailable.
        Mirrors ICAR standard recommendations.
        """
        inp  = predictions.get("inputs", {})
        crop = predictions["crop"]
        fert = predictions["fertilizer"]
        pest = predictions.get("pesticide", "N/A")
        N, P, K = inp.get("N", 80), inp.get("P", 40), inp.get("K", 40)
        ph   = inp.get("ph", 6.5)
        season = inp.get("season", "Kharif")
        soil = inp.get("soil_type", "Loamy")
        humidity = inp.get("humidity", 65)
        rainfall = inp.get("rainfall", 100)
        region   = inp.get("region", "East")
        irrigation = inp.get("irrigation", "Sprinkler")
        farm_size  = inp.get("farm_size", 2)

        n_flag = " (low — priority input)" if N < 60 else ""
        p_flag = " (deficient — apply basal dose)" if P < 30 else ""
        k_flag = " (low — affects grain quality)" if K < 40 else ""
        ph_note = ("Soil is acidic. Apply agricultural lime @ 2 t/ha six weeks before sowing."
                   if ph < 5.8 else
                   "Soil is slightly alkaline. Apply gypsum @ 200 kg/acre and avoid excess urea."
                   if ph > 7.8 else
                   "Soil pH is in the optimal range for most crops.")
        humidity_note = (f" High humidity ({humidity}%) increases risk of fungal diseases. "
                         "Scout fields every 5 days and apply fungicide at first sign of disease."
                         if humidity > 75 else "")

        crop_profiles = {
            "Rice": {
                "sowing": "Transplant 21–25 day old seedlings at 20×15 cm spacing. Seed rate: 25–30 kg/ha.",
                "water": "Maintain 5 cm standing water during tillering. Drain 10 days before harvest.",
                "harvest": "Harvest when 80–85% grains turn golden yellow (100–120 days).",
            },
            "Wheat": {
                "sowing": "Drill seeds at 5–6 cm depth, row spacing 22.5 cm. Seed rate: 100–125 kg/ha.",
                "water": "6 irrigations: CRI (21 DAS), tillering, jointing, flowering, grain fill, dough stage.",
                "harvest": "Harvest at physiological maturity when straw turns golden (110–125 days).",
            },
            "Maize": {
                "sowing": "Sow at 60×25 cm spacing. Seed rate: 18–20 kg/ha (hybrids). Treat seed with Thiram.",
                "water": "Critical irrigations at knee-high, tasseling, and grain fill stages.",
                "harvest": "Harvest at 120–135 days when husks turn brown and dry.",
            },
            "Cotton": {
                "sowing": "Sow Bt cotton at 90×60 cm spacing. Seed rate: 1 kg/acre (hybrid).",
                "water": "Irrigate at squaring, flowering, and boll development. Stop 3 weeks before picking.",
                "harvest": "Pick bolls in 3–4 pickings from 160–180 DAS.",
            },
        }
        cp = crop_profiles.get(crop, {
            "sowing": f"Follow standard sowing guidelines for {crop} in {season} season.",
            "water": "Irrigate based on crop growth stage and soil moisture.",
            "harvest": f"Harvest {crop} at physiological maturity.",
        })

        fert_schedule = {
            "Urea + DAP": f"Basal: DAP 100 kg/ha + Urea 50 kg/ha. Top dress: Urea 50 kg/ha at 30 DAS, Urea 25 kg/ha at 55 DAS.",
            "DAP (18-46-0)": f"Basal: DAP 125 kg/ha (full P dose). Top-dress with Urea 50 kg/ha at 25 DAS and 40 kg/ha at 50 DAS.",
            "NPK 17-17-17": f"Basal: NPK 200 kg/ha. Top-dress with Urea 40 kg/ha at 30 DAS + MOP 30 kg/ha at 55 DAS.",
            "MOP (0-0-60)": f"Basal: DAP 100 kg/ha + MOP 50 kg/ha. Top-dress Urea 60 kg/ha at 25 DAS + 40 kg/ha at 50 DAS.",
            "NPK 12-32-16": f"Basal: NPK 250 kg/ha. Top-dress Urea 40 kg/ha at 30 and 60 DAS.",
        }
        fert_app = fert_schedule.get(fert,
            f"Apply {fert} as per package directions. Split application reduces nitrogen loss.")

        advisory = f"""Based on your field data, I recommend growing {crop} this {season} season.

1. Why {crop} was recommended
Your {soil} soil with pH {ph} and current N={N}{n_flag}, P={P}{p_flag}, K={K}{k_flag} kg/ha, \
combined with {rainfall} mm rainfall and {region} India conditions, \
strongly favour {crop}. {ph_note} \
The ML model analysed {predictions['crop_confidence']}% probability for {crop} over {len(predictions.get('crop_top3',[])) or 2} candidate crops.

2. Land preparation
Plough the field to 20–25 cm depth 3–4 weeks before sowing. \
Apply 10 t/ha farmyard manure and incorporate. Level the field for uniform water distribution. \
For {soil} soil, ensure proper drainage channels are in place.

3. Sowing guide
{cp['sowing']} Best time: {"June–July" if season == "Kharif" else "October–November" if season == "Rabi" else "April–May"}.

4. Fertilizer plan ({fert})
{fert_app} \
{"Phosphorus is deficient — ensure full basal P dose is applied before sowing." if P < 30 else ""} \
{"Potassium is low — do not skip MOP application during grain fill." if K < 40 else ""}

5. Irrigation schedule
{cp['water']} \
{"With " + irrigation + " irrigation, schedule based on crop stage and soil moisture readings." if irrigation != "Rainfed" else "Supplement rainfall with irrigation during dry spells."}

6. Weed management
Maintain weed-free conditions for the first 35 days after sowing. \
Apply recommended pre-emergence herbicide within 3 DAS. \
Carry out one manual or mechanical weeding at 25–30 DAS.

7. Pest and disease management
Primary risk: {predictions.get('pesticide_target', 'common crop pests')}. \
Apply {pest} at {predictions.get('pesticide_dose', 'recommended dose')} when pest pressure exceeds economic threshold. \
{predictions.get('pesticide', {}).get('fungicide_alert', '') if isinstance(predictions.get('pesticide'), dict) else ''} \
{humidity_note} \
Scout fields every 7 days and keep spray records.

8. Harvesting
{cp['harvest']} \
Expected yield: {"4–6 t/ha" if crop == "Rice" else "4–6 t/ha" if crop == "Wheat" else "5–8 t/ha" if crop == "Maize" else "3–5 t/ha"} \
under recommended management on your {farm_size}-acre farm.

9. Post-harvest and storage
Dry grain to safe moisture (12–14%) before storage. \
Store in clean, fumigated gunny bags or metal bins. \
Apply phosphine fumigation every 3 months for long-term storage. \
Keep records of yield, input cost, and sale price for next season planning."""

        return advisory

    def generate_advisory(self, field_inputs: Dict) -> Dict:
        """
        Main entry point for structured advisory generation.

        Returns:
            {
                "predictions": {...},
                "advisory_text": "...",   ← SLM narration
                "sections": {...},        ← parsed sections
                "source": "sagestorm" | "fallback",
                "latency_ms": 142.5,
            }
        """
        t0 = time.perf_counter()

        # 1. ML Predictions
        predictions = self.predictor.predict(field_inputs)

        # 2. Build RAG query from predicted crop + user context
        rag_query = (f"{predictions['crop']} farming {field_inputs.get('season', '')} "
                     f"{field_inputs.get('soil_type', '')} soil fertilizer pest management India")

        rag_context = ""
        try:
            context_str, confidence = self.ctx_builder.build_context_str(rag_query)
            rag_context = context_str
        except Exception as e:
            print(f"[AdvisoryEngine] RAG error: {e}")

        # 3. Build structured prompt
        prompt = self._build_advisory_prompt(predictions, rag_context)

        # 4. SageStorm generation
        slm_output = self._slm_generate(prompt)

        # 5. Validate output quality
        min_words = 80
        has_content = (len(slm_output.split()) >= min_words and
                       any(kw in slm_output.lower() for kw in
                           ["crop", "soil", "fertilizer", "seed", "water",
                            "harvest", "apply", "spray", "grow"]))

        if has_content:
            advisory_text = slm_output
            source = "sagestorm"
        else:
            # Fallback to rule-based advisory
            advisory_text = self._fallback_advisory(predictions)
            source = "fallback"

        sections = extract_sections(advisory_text)
        latency  = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "predictions":   predictions,
            "advisory_text": advisory_text,
            "sections":      sections,
            "prompt_used":   prompt,
            "rag_context":   rag_context,
            "source":        source,
            "latency_ms":    latency,
        }

    def chat_advisory(self, user_message: str) -> Tuple[str, str]:
        """
        Drop-in replacement for ResponseGenerator.generate().
        Used by the chat engine when an advisory query is detected.

        Returns (advisory_text, source_label).
        """
        parsed_inputs = parse_advisory_inputs(user_message)

        if parsed_inputs is None:
            return (
                "Please share your field details for a complete advisory. "
                "For example: 'My soil is Clay, pH 6.5, N=90, P=40, K=50, "
                "temperature 28°C, humidity 70%, rainfall 120mm, Kharif season.'",
                "fallback"
            )

        result = self.generate_advisory(parsed_inputs)
        return result["advisory_text"], result["source"]