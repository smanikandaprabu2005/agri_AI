"""
prompts/advisory_prompt.py
===========================
Strict advisory prompt templates for the LLM explanation layer.

Rules enforced:
  - No placeholder {fields} left unfilled
  - No invented information beyond what structured data provides
  - Output is clean, farmer-friendly, structured
  - System prompt is strict about hallucination
"""

# ── System prompt for the LLM ─────────────────────────────────
SYSTEM_PROMPT = """You are an expert agriculture advisory assistant for Indian farmers.

STRICT RULES — FOLLOW WITHOUT EXCEPTION:
1. Use ONLY the structured data provided in the prompt. Do NOT invent, guess, or add extra information.
2. If any field says "Not provided" or "Not available", output "Not available" for that section — do NOT guess.
3. Do NOT recommend crops, pesticides, or chemicals not listed in the MODEL OUTPUTS section.
4. Do NOT introduce new chemicals, brand names, or dosages not in the input.
5. Keep language simple and farmer-friendly (Class 8 reading level).
6. Do NOT output any placeholder text like {crop_name} or {{ }}.
7. The report must be complete — no trailing sentences, no mid-sentence stops.
8. Numbers and dosages must be copied exactly from the input — do NOT round or modify.
9. If WARNINGS are listed, include them clearly at the end.
10. Tone: helpful, respectful, direct. Do NOT be alarmist."""


# ── User prompt template ──────────────────────────────────────
ADVISORY_PROMPT_TEMPLATE = """
-------------------------------------
FARMER PROFILE:
Location: {location}
Soil Type: {soil_type}
Season: {season}
Temperature: {temperature}
Soil Nutrients:
  Nitrogen (N): {nitrogen}
  Phosphorus (P): {phosphorus}
  Potassium (K): {potassium}
Weather Today: {weather_advisory}
Spraying Safe Today: {spray_safe}
-------------------------------------
MODEL OUTPUTS (DO NOT MODIFY THESE VALUES):

Crop Recommendation:
- Crop: {crop_name}
- Confidence: {crop_confidence}
- Reason Tags: {crop_reason_tags}

Fertilizer Recommendation:
- Fertilizer: {fertilizer_name}
- Dosage: {fertilizer_dosage}
- Application Stage: {fertilizer_stage}

Pest & Disease Risk:
- Pest: {pest_name}
- Pesticide: {pesticide_name}
- Dosage: {pesticide_dosage}
- Organic Option: {organic_alternative}

Warnings from Analysis: {warnings}
Overall System Confidence: {overall_confidence}
-------------------------------------
TASK:
Generate a complete agriculture advisory report using ONLY the data above.
-------------------------------------
OUTPUT FORMAT (FOLLOW EXACTLY):

## Recommended Crop: {crop_name}

**Why this crop:**
[Explain in 2–3 sentences using the soil type, season, temperature, and reason tags. Keep simple.]

---

## Fertilizer Plan

**Fertilizer:** {fertilizer_name}

**Dosage:** {fertilizer_dosage}

**When to apply:** {fertilizer_stage}

**Nutrient Status:**
- Nitrogen: Based on {nitrogen}
- Phosphorus: Based on {phosphorus}
- Potassium: Based on {potassium}

---

## Pest & Disease Control

**Risk identified:** {pest_name}

**Solution:** {pesticide_name}

**Dosage:** {pesticide_dosage}

**Organic alternative:** {organic_alternative}

---

## Weather Advisory

{weather_advisory}

**Spraying today:** {spray_safe}

---

## Practical Tips

[Give exactly 3 practical tips based ONLY on the crop, soil, season, and weather data provided above. Do NOT mention any new chemicals. Tips should cover irrigation, field monitoring, and one soil/crop care action.]

---

## Warnings

{warnings}

---
*Advisory generated with {overall_confidence} confidence. For critical decisions, consult your local Krishi Vigyan Kendra (KVK).*
-------------------------------------
FINAL CHECK: Ensure no empty placeholders appear in your output.
"""


def build_advisory_prompt(prompt_dict: dict) -> str:
    """
    Fill the advisory template with the aggregated model outputs.
    Raises ValueError if any placeholder is not filled.
    """
    try:
        filled = ADVISORY_PROMPT_TEMPLATE.format(**prompt_dict)
    except KeyError as e:
        raise ValueError(f"Missing key in prompt dict: {e}")

    # Sanity check — no unfilled placeholders
    remaining = re.findall(r"\{[a-zA-Z_]+\}", filled)
    if remaining:
        raise ValueError(f"Unfilled placeholders in prompt: {remaining}")

    return filled


import re  # needed by build_advisory_prompt


# ── Shorter conversational prompt (for chat mode) ─────────────
CHAT_PROMPT_TEMPLATE = """
You are a helpful agriculture advisor. Answer the farmer's question using ONLY the context below.
If the context doesn't contain the answer, say "I don't have specific information on that — please consult your local KVK."
Do NOT invent information. Keep your answer practical and under 100 words.

Farmer Context:
{farmer_context}

Question: {question}

Answer:"""


def build_chat_prompt(farmer_context: str, question: str) -> str:
    return CHAT_PROMPT_TEMPLATE.format(
        farmer_context=farmer_context or "No profile available.",
        question=question,
    )
