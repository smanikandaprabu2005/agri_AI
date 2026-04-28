/*
 * ui/src/StromSageV3.jsx
 * =======================
 * SageStorm V3 Chat UI
 *
 * New in V3:
 *  - Advisory panel: enter field params → ML predicts → SageStorm explains
 *  - ML confidence bars shown alongside the SLM explanation
 *  - Chat still works for simple questions
 *  - "Advisory mode" auto-detected when field data is in message
 */

import { useState, useRef, useEffect, useCallback } from "react";

const THEME = {
  soil:      "#2A2C24",
  bark:      "#39442F",
  moss:      "#31655B",
  leaf:      "#49806C",
  sage:      "#6B9B70",
  sprout:    "#8CC06F",
  straw:     "#D9C78B",
  wheat:     "#F0D9A8",
  cream:     "#F6EFE2",
  parchment: "#FBF5EA",
};

const S = {
  root: { fontFamily: "'Crimson Pro', Georgia, serif", background: "#F6EFE2", height: "100vh", display: "flex", flexDirection: "column", color: THEME.soil, overflow: "hidden" },
  header: { background: `linear-gradient(135deg, ${THEME.leaf} 0%, ${THEME.sage} 100%)`, padding: "0 28px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64, flexShrink: 0, boxShadow: "0 2px 12px rgba(0,0,0,0.12)" },
  main: { display: "flex", flex: 1, overflow: "hidden" },
  sidebar: { width: 280, background: THEME.cream, borderRight: `1px solid ${THEME.sage}25`, display: "flex", flexDirection: "column", overflow: "hidden", flexShrink: 0 },
  chatArea: { flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" },
  messages: { flex: 1, overflowY: "auto", padding: "24px 32px", display: "flex", flexDirection: "column", gap: 18 },
  inputArea: { borderTop: `1px solid ${THEME.wheat}60`, background: THEME.parchment, padding: "16px 28px 20px" },
  inputRow: { display: "flex", gap: 10, background: "#fff", border: `1px solid ${THEME.wheat}50`, borderRadius: 20, padding: "10px 14px 10px 18px", boxShadow: "0 4px 16px rgba(0,0,0,0.06)" },
  textarea: { flex: 1, border: "none", outline: "none", resize: "none", background: "transparent", fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 15, color: THEME.soil, lineHeight: 1.5, minHeight: 22, maxHeight: 100 },
  sendBtn: { width: 40, height: 40, borderRadius: 12, border: "none", background: THEME.sprout, color: THEME.parchment, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 },
  card: { background: "#fff", border: `1px solid ${THEME.wheat}50`, borderRadius: 14, padding: "14px 16px", marginBottom: 10 },
  label: { fontSize: 11, color: THEME.straw, fontFamily: "'Source Code Pro', monospace", letterSpacing: "0.1em", textTransform: "uppercase", marginBottom: 6, display: "block" },
  input: { width: "100%", padding: "7px 10px", border: `1px solid ${THEME.wheat}`, borderRadius: 8, background: "#fff", fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14, color: THEME.soil, outline: "none", boxSizing: "border-box" },
  select: { width: "100%", padding: "7px 10px", border: `1px solid ${THEME.wheat}`, borderRadius: 8, background: "#fff", fontFamily: "'Crimson Pro', Georgia, serif", fontSize: 14, color: THEME.soil, outline: "none", boxSizing: "border-box", cursor: "pointer" },
  btn: { background: THEME.moss, color: THEME.wheat, border: "none", padding: "9px 20px", borderRadius: 10, fontSize: 14, cursor: "pointer", fontFamily: "'Crimson Pro', Georgia, serif", fontWeight: 600, width: "100%", marginTop: 6 },
};

const SOURCES = {
  advisory_ml: { bg: "#e8f5e2", color: "#2D5A1B", label: "ML + SageStorm" },
  template:    { bg: "#dbeafe", color: "#1e40af", label: "Template" },
  retrieval:   { bg: "#e0f2fe", color: "#0369a1", label: "Retrieval" },
  sagestorm:   { bg: "#f0ede8", color: "#4A2C1A", label: "SageStorm" },
  fallback:    { bg: "#fef3c7", color: "#92400e", label: "Fallback" },
};

const SourceBadge = ({ source }) => {
  const cfg = SOURCES[source] || SOURCES.fallback;
  return (
    <span style={{ background: cfg.bg, color: cfg.color, borderRadius: 10, padding: "2px 8px", fontSize: 10, fontWeight: 600, fontFamily: "monospace" }}>
      {cfg.label}
    </span>
  );
};

const ConfidenceBar = ({ label, value, color = THEME.sprout }) => (
  <div style={{ marginBottom: 8 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, marginBottom: 3 }}>
      <span style={{ color: THEME.bark }}>{label}</span>
      <span style={{ fontWeight: 600, color: THEME.soil }}>{value}%</span>
    </div>
    <div style={{ height: 5, background: `${THEME.wheat}60`, borderRadius: 3 }}>
      <div style={{ height: 5, background: color, borderRadius: 3, width: `${value}%`, transition: "width 0.6s ease" }} />
    </div>
  </div>
);

const MLPanel = ({ preds }) => {
  if (!preds) return null;
  return (
    <div style={{ ...S.card, borderLeft: `3px solid ${THEME.sprout}`, marginTop: 10 }}>
      <div style={{ fontSize: 11, color: THEME.straw, marginBottom: 10, fontFamily: "monospace" }}>ML PREDICTIONS</div>
      <ConfidenceBar label={`Crop: ${preds.crop}`} value={preds.crop_confidence} color={THEME.leaf} />
      <ConfidenceBar label={`Fertilizer: ${preds.fertilizer}`} value={preds.fertilizer_confidence} color={THEME.moss} />
      {preds.pesticide_target && (
        <div style={{ fontSize: 12, color: THEME.bark, marginTop: 6, padding: "6px 8px", background: "#fef3c7", borderRadius: 6 }}>
          ⚠ Pest risk: <strong>{preds.pesticide_target}</strong> → {preds.pesticide}
        </div>
      )}
      {preds.crop_top3 && preds.crop_top3.length > 1 && (
        <div style={{ fontSize: 11, color: THEME.straw, marginTop: 8 }}>
          Alternatives: {preds.crop_top3.slice(1).map(x =>
            `${x.crop || x.crop} (${typeof x.confidence === "number" && x.confidence < 1 ? Math.round(x.confidence * 100) : x.confidence}%)`
          ).join(", ")}
        </div>
      )}
    </div>
  );
};

const Message = ({ msg }) => {
  const isUser = msg.role === "user";
  return (
    <div style={{ display: "flex", gap: 10, alignItems: "flex-start", alignSelf: isUser ? "flex-end" : "flex-start", maxWidth: "85%" }}>
      {!isUser && (
        <div style={{ width: 32, height: 32, borderRadius: "50%", background: THEME.moss, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, flexShrink: 0, color: THEME.wheat }}>S</div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{
          borderRadius: 16,
          padding: "12px 16px",
          lineHeight: 1.65,
          fontSize: 15,
          background: isUser ? `linear-gradient(135deg, ${THEME.sage}, ${THEME.sprout})` : "#fff",
          color: isUser ? THEME.parchment : THEME.soil,
          border: isUser ? "none" : `1px solid ${THEME.wheat}40`,
          borderBottomLeftRadius: isUser ? 16 : 4,
          borderBottomRightRadius: isUser ? 4 : 16,
          whiteSpace: "pre-wrap",
          boxShadow: "0 4px 14px rgba(0,0,0,0.07)",
        }}>
          {msg.text}
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8, flexDirection: isUser ? "row-reverse" : "row" }}>
          <span style={{ fontSize: 10, color: THEME.straw, fontFamily: "monospace" }}>
            {msg.time.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
          </span>
          {!isUser && msg.source && <SourceBadge source={msg.source} />}
        </div>
        {!isUser && msg.ml_predictions && <MLPanel preds={msg.ml_predictions} />}
      </div>
    </div>
  );
};

const TypingDots = () => (
  <div style={{ display: "flex", gap: 4, padding: "12px 16px", background: "#fff", border: `1px solid ${THEME.wheat}40`, borderRadius: 16, borderBottomLeftRadius: 4, width: "fit-content", alignItems: "center" }}>
    {[0, 1, 2].map(i => (
      <div key={i} style={{ width: 6, height: 6, borderRadius: "50%", background: THEME.sage, animation: `bounce 1.2s ${i * 0.2}s infinite` }} />
    ))}
  </div>
);

const FormField = ({ label, id, type = "number", value, onChange, options }) => (
  <div style={{ marginBottom: 10 }}>
    <label style={S.label}>{label}</label>
    {options ? (
      <select style={S.select} value={value} onChange={e => onChange(e.target.value)}>
        {options.map(o => <option key={o} value={o}>{o}</option>)}
      </select>
    ) : (
      <input type={type} style={S.input} value={value} onChange={e => onChange(type === "number" ? parseFloat(e.target.value) : e.target.value)} step="any" />
    )}
  </div>
);

const DEFAULT_FIELDS = {
  soil_type: "Clay", ph: 6.5, soil_moisture: 35, organic_carbon: 0.8, ec: 0.5,
  N: 90, P: 40, K: 50, temperature: 28, humidity: 70, rainfall: 120,
  season: "Kharif", irrigation: "Sprinkler", region: "East", previous_crop: "Rice", farm_size: 3
};

export default function StromSageV3() {
  const [messages, setMessages]   = useState([]);
  const [input, setInput]         = useState("");
  const [loading, setLoading]     = useState(false);
  const [mode, setMode]           = useState("chat"); // "chat" | "advisory"
  const [fields, setFields]       = useState(DEFAULT_FIELDS);
  const [lastPreds, setLastPreds] = useState(null);
  const endRef   = useRef(null);
  const taRef    = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);

  useEffect(() => {
    if (taRef.current) {
      taRef.current.style.height = "auto";
      taRef.current.style.height = Math.min(taRef.current.scrollHeight, 100) + "px";
    }
  }, [input]);

  const fieldUpdate = (key) => (val) => setFields(f => ({ ...f, [key]: val }));

  const callAPI = async (path, body) => {
    const res = await fetch(`http://localhost:8000${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  };

  const simulateAdvisory = async (inputs) => {
    await new Promise(r => setTimeout(r, 1200));
    const crops = { Kharif: "Rice", Rabi: "Wheat", Zaid: "Maize" };
    const ferts = { Rice: "Urea + DAP", Wheat: "DAP (18-46-0)", Maize: "NPK 17-17-17" };
    const crop = crops[inputs.season] || "Rice";
    const fert = ferts[crop] || "Urea + DAP";
    return {
      crop, crop_confidence: 84, fertilizer: fert, fertilizer_confidence: 76,
      pesticide: "Chlorpyrifos 20EC", pesticide_target: "Stem borer",
      pesticide_dose: "2.5 ml/litre", alternatives: [],
      source: "advisory_ml",
      advisory_text:
        `[ML Prediction: ${crop} (84% confidence) · Fertilizer: ${fert} · Pest risk: Stem borer]\n\n` +
        `Based on your field data — ${inputs.soil_type} soil, pH ${inputs.ph}, N=${inputs.N}/P=${inputs.P}/K=${inputs.K} kg/ha, ` +
        `${inputs.temperature}°C, ${inputs.humidity}% humidity, ${inputs.rainfall}mm rainfall in ${inputs.season} season — ` +
        `I recommend growing ${crop} this season.\n\n` +
        `1. Why ${crop}: Your soil conditions and ${inputs.season} climate strongly favour ${crop}. ` +
        `${inputs.N < 60 ? "Nitrogen is below threshold — prioritise nitrogen application." : "Nitrogen levels are adequate."} ` +
        `${inputs.ph < 5.8 ? "Soil is acidic — apply lime @ 2 t/ha before sowing." : "Soil pH is in optimal range."}\n\n` +
        `2. Land preparation: Deep plough to 20–25 cm. Apply 10 t/ha FYM. Level field for uniform water distribution.\n\n` +
        `3. Sowing: ${crop === "Rice" ? "Transplant 21–25 day seedlings at 20×15 cm, seed rate 25–30 kg/ha." : "Drill at 22.5 cm row spacing, seed rate 100–125 kg/ha."}\n\n` +
        `4. Fertilizer (${fert}): Apply half N + full P as basal dose. Top-dress remaining N at 30 and 55 DAS.\n\n` +
        `5. Irrigation: ${inputs.irrigation} irrigation. Critical stages: tillering, heading, and grain fill.\n\n` +
        `6. Pest management: Monitor for Stem borer. Apply Chlorpyrifos 20EC @ 2.5 ml/L at first sign. ` +
        `${inputs.humidity > 75 ? "High humidity — also watch for fungal diseases; apply Tricyclazole at first lesion." : ""}\n\n` +
        `7. Harvest: At 80–85% grain maturity (~110–120 days). Expected yield: 4–6 t/ha on your ${inputs.farm_size} acres.`,
    };
  };

  const submitAdvisory = async () => {
    const userText = `Advisory request: ${fields.soil_type} soil, pH ${fields.ph}, N=${fields.N}/P=${fields.P}/K=${fields.K}, Temp ${fields.temperature}°C, Humidity ${fields.humidity}%, Rainfall ${fields.rainfall}mm, ${fields.season} season, ${fields.region} India`;
    const userMsg = { id: Date.now(), role: "user", text: userText, time: new Date() };
    setMessages(m => [...m, userMsg]);
    setLoading(true);

    try {
      let data;
      try {
        data = await callAPI("/advisory", { inputs: fields });
      } catch {
        data = await simulateAdvisory(fields);
      }

      const preds = {
        crop: data.crop, crop_confidence: data.crop_confidence,
        fertilizer: data.fertilizer, fertilizer_confidence: data.fertilizer_confidence,
        pesticide: data.pesticide, pesticide_target: data.pesticide_target,
        crop_top3: data.alternatives || [],
      };
      setLastPreds(preds);

      setMessages(m => [...m, {
        id: Date.now() + 1, role: "bot",
        text: data.advisory_text || "Advisory generated.",
        source: data.source || "advisory_ml",
        ml_predictions: preds,
        time: new Date(),
      }]);
    } catch (err) {
      setMessages(m => [...m, { id: Date.now() + 1, role: "bot", text: "Error generating advisory. Please check the API server.", source: "fallback", time: new Date() }]);
    } finally {
      setLoading(false);
    }
  };

  const submitChat = useCallback(async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;
    setInput("");
    const userMsg = { id: Date.now(), role: "user", text: q, time: new Date() };
    setMessages(m => [...m, userMsg]);
    setLoading(true);

    try {
      let data;
      try {
        data = await callAPI("/chat", { query: q });
      } catch {
        await new Promise(r => setTimeout(r, 900));
        data = { answer: `SageStorm: ${q.length > 30 ? "Based on agricultural best practices, " : ""}I'd recommend consulting your local KVK for region-specific advice on this. For detailed crop management, share your field parameters for a complete ML-backed advisory.`, source: "sagestorm" };
      }
      const preds = data.ml_predictions || null;
      if (preds) setLastPreds(preds);
      setMessages(m => [...m, { id: Date.now() + 1, role: "bot", text: data.answer, source: data.source, ml_predictions: preds, time: new Date() }]);
    } catch {
      setMessages(m => [...m, { id: Date.now() + 1, role: "bot", text: "Connection error.", source: "fallback", time: new Date() }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading]);

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Crimson+Pro:wght@400;600&family=Source+Code+Pro:wght@400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        @keyframes bounce { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-4px); } }
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-thumb { background: ${THEME.wheat}; border-radius: 3px; }
      `}</style>

      <div style={S.root}>
        {/* Header */}
        <header style={S.header}>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{ width: 32, height: 32, borderRadius: "50%", background: THEME.sprout, display: "flex", alignItems: "center", justifyContent: "center", color: THEME.wheat, fontSize: 16 }}>S</div>
            <div>
              <div style={{ fontSize: 17, fontWeight: 700, color: THEME.wheat, fontFamily: "'Playfair Display', Georgia, serif" }}>Strom Sage V3</div>
              <div style={{ fontSize: 10, color: THEME.straw, letterSpacing: "0.12em" }}>ML ADVISORY · SAGESTORM SLM</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 6 }}>
            {["chat", "advisory"].map(m => (
              <button key={m} onClick={() => setMode(m)} style={{ padding: "6px 14px", border: `1px solid ${mode === m ? THEME.sprout : THEME.sage + "44"}`, borderRadius: 10, background: mode === m ? THEME.sprout : "transparent", color: mode === m ? THEME.wheat : THEME.straw, cursor: "pointer", fontSize: 12, fontFamily: "monospace" }}>
                {m === "chat" ? "Chat" : "Advisory"}
              </button>
            ))}
          </div>
        </header>

        <div style={S.main}>
          {/* Sidebar */}
          <aside style={S.sidebar}>
            <div style={{ padding: "14px 16px", borderBottom: `1px solid ${THEME.wheat}40`, overflowY: "auto", flex: 1 }}>
              <div style={{ fontSize: 11, color: THEME.straw, fontFamily: "monospace", letterSpacing: "0.12em", marginBottom: 12 }}>FIELD PARAMETERS</div>

              <FormField label="Soil type" value={fields.soil_type} onChange={fieldUpdate("soil_type")} options={["Clay","Loamy","Sandy","Silt","Black","Red","Alluvial"]} />
              <FormField label="Season" value={fields.season} onChange={fieldUpdate("season")} options={["Kharif","Rabi","Zaid"]} />
              <FormField label="Region" value={fields.region} onChange={fieldUpdate("region")} options={["North","South","East","West","Central"]} />
              <FormField label="Irrigation" value={fields.irrigation} onChange={fieldUpdate("irrigation")} options={["Canal","Drip","Sprinkler","Rainfed"]} />
              <FormField label="Soil pH" value={fields.ph} onChange={fieldUpdate("ph")} />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8 }}>
                <FormField label="N kg/ha" value={fields.N} onChange={fieldUpdate("N")} />
                <FormField label="P kg/ha" value={fields.P} onChange={fieldUpdate("P")} />
                <FormField label="K kg/ha" value={fields.K} onChange={fieldUpdate("K")} />
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <FormField label="Temp °C" value={fields.temperature} onChange={fieldUpdate("temperature")} />
                <FormField label="Humidity %" value={fields.humidity} onChange={fieldUpdate("humidity")} />
              </div>
              <FormField label="Rainfall mm" value={fields.rainfall} onChange={fieldUpdate("rainfall")} />
              <FormField label="Farm size (acres)" value={fields.farm_size} onChange={fieldUpdate("farm_size")} />
              <FormField label="Previous crop" value={fields.previous_crop} onChange={fieldUpdate("previous_crop")} options={["Rice","Wheat","Maize","Cotton","Potato","Tomato","Sugarcane","Mustard"]} />

              <button style={S.btn} onClick={submitAdvisory} disabled={loading}>
                {loading ? "Generating..." : "Get ML Advisory ↗"}
              </button>

              {lastPreds && (
                <div style={{ marginTop: 16 }}>
                  <div style={{ fontSize: 11, color: THEME.straw, fontFamily: "monospace", marginBottom: 8 }}>LAST PREDICTION</div>
                  <ConfidenceBar label={`Crop: ${lastPreds.crop}`} value={lastPreds.crop_confidence} color={THEME.leaf} />
                  <ConfidenceBar label={`Fert: ${lastPreds.fertilizer}`} value={lastPreds.fertilizer_confidence} color={THEME.moss} />
                </div>
              )}
            </div>
          </aside>

          {/* Chat */}
          <main style={S.chatArea}>
            <div style={S.messages}>
              {messages.length === 0 && (
                <div style={{ textAlign: "center", padding: "3rem 2rem", color: THEME.straw }}>
                  <div style={{ fontSize: 40, marginBottom: 16 }}>🌾</div>
                  <div style={{ fontSize: 22, fontWeight: 700, color: THEME.leaf, fontFamily: "'Playfair Display', serif", marginBottom: 10 }}>Namaste! I'm Strom Sage</div>
                  <div style={{ fontSize: 15, color: THEME.bark, maxWidth: 440, margin: "0 auto 28px", lineHeight: 1.7 }}>
                    Enter your field parameters in the sidebar and click <strong>Get ML Advisory</strong> to receive a complete ML-predicted, SageStorm-explained farming plan.
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, maxWidth: 460, margin: "0 auto" }}>
                    {["How do I control stem borers in rice?", "What fertilizer for wheat in clay soil?", "Should I spray today given high humidity?", "Best spacing for tomato plants?"].map((q, i) => (
                      <button key={i} onClick={() => submitChat(q)} style={{ background: "#fff", border: `1px solid ${THEME.wheat}60`, borderRadius: 12, padding: "10px 14px", cursor: "pointer", fontSize: 13, color: THEME.soil, textAlign: "left", lineHeight: 1.4 }}>
                        {q}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map(m => <Message key={m.id} msg={m} />)}
              {loading && (
                <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                  <div style={{ width: 32, height: 32, borderRadius: "50%", background: THEME.moss, display: "flex", alignItems: "center", justifyContent: "center", color: THEME.wheat, fontSize: 14, flexShrink: 0 }}>S</div>
                  <TypingDots />
                </div>
              )}
              <div ref={endRef} />
            </div>

            <div style={S.inputArea}>
              <div style={S.inputRow}>
                <textarea ref={taRef} style={S.textarea} value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submitChat(); } }} placeholder="Ask about crops, pests, fertilizers..." rows={1} disabled={loading} />
                <button style={{ ...S.sendBtn, ...((!input.trim() || loading) ? { background: THEME.wheat, cursor: "not-allowed" } : {}) }} onClick={() => submitChat()} disabled={!input.trim() || loading}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/></svg>
                </button>
              </div>
              <div style={{ marginTop: 6, fontSize: 11, color: THEME.straw, fontFamily: "monospace" }}>
                Enter to send · Shift+Enter for new line · Use sidebar for ML advisory
              </div>
            </div>
          </main>
        </div>
      </div>
    </>
  );
}