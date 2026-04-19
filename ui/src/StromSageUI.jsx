import { useState, useRef, useEffect, useCallback } from "react";

// ── Design Tokens ─────────────────────────────────────────────
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
  sky:       "#4D7592",
  rain:      "#68A8C2",
  cloud:     "#E2EDF4",
  emerald:   "#0F9D58",
  amber:     "#F0B429",
  indigo:    "#5B67F5",
  rose:      "#E35A7E",
  error:     "#BC2B2B",
  warn:      "#C57C27",
  info:      "#3B6F91",
  success:   "#0F9D58",
};

// ── Inline Styles (no external CSS) ───────────────────────────
const styles = {
  root: {
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    background: "#F6EFE2",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    color: THEME.soil,
    overflow: "hidden",
  },
  // Header
  header: {
    background: `linear-gradient(135deg, ${THEME.leaf} 0%, ${THEME.sage} 100%)`,
    borderBottom: `3px solid ${THEME.sprout}40`,
    padding: "0 32px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    height: 72,
    flexShrink: 0,
    position: "sticky",
    top: 0,
    zIndex: 100,
    boxShadow: "0 4px 18px rgba(0,0,0,0.12)",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    cursor: "pointer",
    userSelect: "none",
  },
  logoIcon: {
    width: 34,
    height: 34,
  },
  logoText: {
    fontFamily: "'Playfair Display', 'Georgia', serif",
    fontSize: 20,
    fontWeight: 700,
    color: THEME.wheat,
    letterSpacing: "0.02em",
  },
  logoSub: {
    fontSize: 10,
    color: THEME.straw,
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    marginTop: -2,
    display: "block",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: 12,
  },
  statusBadge: {
    display: "flex",
    alignItems: "center",
    gap: 6,
    background: "rgba(122,173,74,0.15)",
    border: `1px solid ${THEME.sprout}`,
    borderRadius: 20,
    padding: "4px 12px",
    fontSize: 12,
    color: THEME.sprout,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.05em",
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: THEME.sprout,
    animation: "pulse 2s infinite",
  },
  // Main layout
  main: {
    display: "flex",
    flexDirection: "row",
    flex: 1,
    overflow: "hidden",
    height: "calc(100vh - 72px)",
    background: `linear-gradient(135deg, ${THEME.parchment} 0%, ${THEME.cream} 100%)`,
    alignItems: "stretch",
  },
  // Sidebar
  sidebar: {
    width: 300,
    background: `linear-gradient(180deg, ${THEME.cream} 0%, ${THEME.wheat}10 100%)`,
    borderRight: `1px solid ${THEME.sage}20`,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    flexShrink: 0,
    height: "100%",
    boxShadow: "2px 0 14px rgba(0,0,0,0.08)",
    position: "sticky",
    top: 72,
  },
  sidebarSection: {
    padding: "20px 20px 12px",
    borderBottom: `1px solid ${THEME.wheat}40`,
  },
  sidebarTitle: {
    fontSize: 11,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    color: THEME.straw,
    marginBottom: 12,
    fontWeight: 600,
  },
  profileCard: {
    background: `linear-gradient(135deg, #ffffff 0%, ${THEME.cream} 100%)`,
    border: `1px solid ${THEME.wheat}50`,
    borderRadius: 16,
    padding: "18px 18px",
    boxShadow: "0 10px 25px rgba(0,0,0,0.08)",
    transition: "transform 0.25s ease, box-shadow 0.25s ease",
  },
  profileCardHover: {
    transform: "translateY(-2px)",
    boxShadow: "0 6px 20px rgba(0,0,0,0.12)",
  },
  profileRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 6,
    fontSize: 13,
    color: THEME.bark,
  },
  profileLabel: {
    fontSize: 11,
    color: THEME.straw,
    minWidth: 50,
  },
  profileValue: {
    color: THEME.soil,
    fontWeight: 600,
    fontSize: 13,
  },
  // Weather card
  weatherCard: {
    background: `linear-gradient(135deg, ${THEME.cloud} 0%, ${THEME.cream} 100%)`,
    border: `1px solid ${THEME.sky}25`,
    borderRadius: 16,
    padding: "18px 18px",
    boxShadow: "0 12px 26px rgba(0,0,0,0.08)",
    transition: "transform 0.25s ease, box-shadow 0.25s ease",
  },
  weatherCity: {
    fontSize: 13,
    fontWeight: 600,
    color: THEME.sky,
    marginBottom: 4,
  },
  weatherGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 6,
  },
  weatherItem: {
    fontSize: 12,
    color: THEME.bark,
  },
  weatherVal: {
    fontWeight: 700,
    color: THEME.soil,
  },
  weatherWarn: {
    marginTop: 8,
    background: `${THEME.warn}20`,
    border: `1px solid ${THEME.warn}44`,
    borderRadius: 6,
    padding: "6px 8px",
    fontSize: 11,
    color: THEME.warn,
    display: "flex",
    gap: 5,
    alignItems: "flex-start",
  },
  // Source indicators
  sourceTag: {
    display: "inline-flex",
    alignItems: "center",
    gap: 4,
    padding: "2px 8px",
    borderRadius: 12,
    fontSize: 10,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    fontWeight: 600,
  },
  // Conversation area
  chatArea: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    background: "transparent",
    minHeight: 0,
  },
  messages: {
    flex: 1,
    overflowY: "auto",
    padding: "32px 40px",
    display: "flex",
    flexDirection: "column",
    gap: 24,
    scrollBehavior: "smooth",
    scrollbarWidth: "thin",
    scrollbarColor: `${THEME.sage}40 transparent`,
    minHeight: 0,
  },
  // Welcome screen
  welcome: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    padding: "60px 32px",
    textAlign: "center",
    flex: 1,
  },
  welcomeIcon: {
    width: 100,
    height: 100,
    marginBottom: 32,
    filter: "drop-shadow(0 4px 8px rgba(0,0,0,0.2))",
  },
  welcomeTitle: {
    fontFamily: "'Playfair Display', 'Georgia', serif",
    fontSize: 38,
    fontWeight: 700,
    color: THEME.leaf,
    marginBottom: 14,
    lineHeight: 1.15,
    textShadow: "0 3px 10px rgba(0,0,0,0.08)",
  },
  welcomeSub: {
    fontSize: 18,
    color: THEME.bark,
    maxWidth: 600,
    lineHeight: 1.7,
    marginBottom: 48,
  },
  suggestionsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 16,
    width: "100%",
    maxWidth: 720,
  },
  suggestionCard: {
    background: `linear-gradient(135deg, rgba(255,255,255,0.9) 0%, ${THEME.cream} 100%)`,
    border: `1px solid ${THEME.wheat}60`,
    borderRadius: 18,
    padding: "18px 22px",
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.25s ease",
    display: "flex",
    alignItems: "flex-start",
    gap: 12,
    boxShadow: "0 14px 30px rgba(0,0,0,0.06)",
  },
  suggestionCardHover: {
    transform: "translateY(-4px)",
    boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
    borderColor: THEME.sage,
  },
  suggestionEmoji: {
    fontSize: 22,
    lineHeight: 1,
    flexShrink: 0,
    marginTop: 1,
  },
  suggestionText: {
    fontSize: 13,
    color: THEME.soil,
    lineHeight: 1.4,
    fontFamily: "'Crimson Pro', 'Georgia', serif",
  },
  // Message bubbles
  msgRow: {
    display: "flex",
    gap: 12,
    alignItems: "flex-start",
    maxWidth: 780,
  },
  msgRowUser: {
    flexDirection: "row-reverse",
    alignSelf: "flex-end",
  },
  msgRowBot: {
    alignSelf: "flex-start",
  },
  avatar: {
    width: 36,
    height: 36,
    borderRadius: "50%",
    flexShrink: 0,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: 16,
  },
  avatarBot: {
    background: THEME.moss,
    border: `2px solid ${THEME.sage}`,
  },
  avatarUser: {
    background: THEME.bark,
    border: `2px solid ${THEME.straw}`,
  },
  bubble: {
    borderRadius: 20,
    padding: "16px 20px",
    maxWidth: 680,
    lineHeight: 1.65,
    fontSize: 16,
    position: "relative",
    boxShadow: "0 10px 24px rgba(0,0,0,0.08)",
    transition: "transform 0.25s ease",
  },
  bubbleUser: {
    background: `linear-gradient(135deg, ${THEME.sage} 0%, ${THEME.sprout} 100%)`,
    color: THEME.parchment,
    borderBottomRightRadius: 8,
  },
  bubbleBot: {
    background: `linear-gradient(135deg, #ffffff 0%, ${THEME.cloud} 100%)`,
    color: THEME.soil,
    border: `1px solid ${THEME.wheat}40`,
    borderBottomLeftRadius: 8,
  },
  bubbleFooter: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginTop: 6,
    justifyContent: "space-between",
  },
  msgTime: {
    fontSize: 11,
    color: THEME.straw,
    fontFamily: "'Source Code Pro', monospace",
  },
  // Typing indicator
  typing: {
    display: "flex",
    gap: 4,
    padding: "14px 18px",
    alignItems: "center",
  },
  typingDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: THEME.sage,
    animation: "bounce 1.2s infinite",
  },
  // Input area
  inputArea: {
    borderTop: `2px solid ${THEME.wheat}60`,
    background: `linear-gradient(180deg, ${THEME.cream} 0%, ${THEME.parchment} 100%)`,
    padding: "20px 32px 24px",
    boxShadow: "0 -2px 8px rgba(0,0,0,0.05)",
  },
  inputRow: {
    display: "flex",
    gap: 12,
    alignItems: "flex-end",
    background: `linear-gradient(135deg, #ffffff 0%, ${THEME.cloud} 100%)`,
    border: `1px solid ${THEME.wheat}50`,
    borderRadius: 24,
    padding: "14px 16px 14px 22px",
    transition: "all 0.3s ease",
    boxShadow: "0 12px 28px rgba(0,0,0,0.06)",
  },
  inputRowFocus: {
    borderColor: THEME.sage,
    boxShadow: "0 4px 16px rgba(90,122,58,0.2)",
  },
  textarea: {
    flex: 1,
    border: "none",
    outline: "none",
    resize: "none",
    background: "transparent",
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    fontSize: 15,
    color: THEME.soil,
    lineHeight: 1.5,
    minHeight: 24,
    maxHeight: 120,
    paddingTop: 2,
  },
  sendBtn: {
    width: 44,
    height: 44,
    borderRadius: 14,
    border: "none",
    background: THEME.sprout,
    color: THEME.parchment,
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    transition: "all 0.2s ease",
    boxShadow: "0 10px 22px rgba(69,115,85,0.18)",
  },
  sendBtnDisabled: {
    background: THEME.wheat,
    color: THEME.straw,
    cursor: "not-allowed",
  },
  inputHints: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 8,
    padding: "0 4px",
  },
  hintText: {
    fontSize: 11,
    color: THEME.straw,
    fontFamily: "'Source Code Pro', monospace",
  },
  quickBtns: {
    display: "flex",
    gap: 6,
  },
  quickBtn: {
    background: "transparent",
    border: `1px solid ${THEME.wheat}60`,
    borderRadius: 12,
    padding: "6px 12px",
    fontSize: 12,
    color: THEME.bark,
    cursor: "pointer",
    fontFamily: "'Source Code Pro', monospace",
    transition: "all 0.2s ease",
  },
  // Right panel
  rightPanel: {
    width: 280,
    background: `linear-gradient(180deg, ${THEME.cream} 0%, ${THEME.cloud} 100%)`,
    borderLeft: `1px solid ${THEME.sage}20`,
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    flexShrink: 0,
    height: "100%",
    boxShadow: "-2px 0 14px rgba(0,0,0,0.08)",
    position: "sticky",
    top: 72,
  },
  rightSection: {
    padding: "20px",
    borderBottom: `1px solid ${THEME.wheat}40`,
  },
  historyItem: {
    padding: "8px 10px",
    borderRadius: 8,
    cursor: "pointer",
    marginBottom: 4,
    transition: "background 0.1s",
  },
  historyText: {
    fontSize: 12,
    color: THEME.bark,
    lineHeight: 1.3,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  historyTime: {
    fontSize: 10,
    color: THEME.straw,
    fontFamily: "'Source Code Pro', monospace",
    marginTop: 2,
  },
  // Modal / Settings
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(44,26,14,0.5)",
    zIndex: 200,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  modal: {
    background: THEME.parchment,
    border: `2px solid ${THEME.wheat}`,
    borderRadius: 16,
    padding: 28,
    width: 460,
    maxWidth: "90vw",
  },
  modalTitle: {
    fontFamily: "'Playfair Display', 'Georgia', serif",
    fontSize: 20,
    fontWeight: 700,
    color: THEME.moss,
    marginBottom: 20,
  },
  formGroup: {
    marginBottom: 16,
  },
  label: {
    display: "block",
    fontSize: 12,
    color: THEME.bark,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    marginBottom: 6,
  },
  input: {
    width: "100%",
    padding: "9px 12px",
    border: `1.5px solid ${THEME.wheat}`,
    borderRadius: 8,
    background: "#fff",
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    fontSize: 14,
    color: THEME.soil,
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s",
  },
  select: {
    width: "100%",
    padding: "9px 12px",
    border: `1.5px solid ${THEME.wheat}`,
    borderRadius: 8,
    background: "#fff",
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    fontSize: 14,
    color: THEME.soil,
    outline: "none",
    boxSizing: "border-box",
  },
  btnRow: {
    display: "flex",
    gap: 10,
    justifyContent: "flex-end",
    marginTop: 24,
  },
  btnPrimary: {
    background: THEME.moss,
    color: THEME.wheat,
    border: "none",
    borderRadius: 10,
    padding: "10px 24px",
    fontSize: 14,
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    cursor: "pointer",
    fontWeight: 600,
    transition: "background 0.15s",
  },
  btnSecondary: {
    background: "transparent",
    color: THEME.bark,
    border: `1.5px solid ${THEME.wheat}`,
    borderRadius: 10,
    padding: "10px 24px",
    fontSize: 14,
    fontFamily: "'Crimson Pro', 'Georgia', serif",
    cursor: "pointer",
    transition: "all 0.15s",
  },
  iconBtn: {
    background: "transparent",
    border: `1px solid ${THEME.wheat}`,
    borderRadius: 8,
    width: 32,
    height: 32,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    cursor: "pointer",
    color: THEME.bark,
    transition: "all 0.15s",
    flexShrink: 0,
  },
};

// ── Source Badge Config ────────────────────────────────────────
const SOURCE_CONFIG = {
  template:   { bg: "#e8f5e2", color: "#2D5A1B", border: "#7AAD4A44", label: "✦ template" },
  retrieval:  { bg: "#e3f0f8", color: "#1A4A6B", border: "#3A6B8A44", label: "⟳ retrieval" },
  sagestorm:  { bg: "#f0ede8", color: "#4A2C1A", border: "#C8A96E44", label: "✧ sagestorm v2" },
  fallback:   { bg: "#fdf3e3", color: "#6B3B0A", border: "#D4781C44", label: "⚠ fallback" },
};

// ── Weather Data (mock) ────────────────────────────────────────
const MOCK_WEATHER = {
  city: "Guwahati",
  temp: 32,
  humid: 78,
  wind: 12,
  desc: "Partly cloudy",
  rain: false,
  rain_pct: 25,
  src: "mock",
};

// ── Suggestions ───────────────────────────────────────────────
const SUGGESTIONS = [
  { emoji: "🌾", text: "How do I control stem borers in my rice crop?" },
  { emoji: "🍅", text: "Fertilizer dose for tomatoes per acre" },
  { emoji: "🐛", text: "Aphids on my lemon tree — what should I do?" },
  { emoji: "☔", text: "Should I spray pesticide today given the weather?" },
  { emoji: "🌱", text: "Best planting spacing for banana trees" },
  { emoji: "🍃", text: "How to prevent late blight in tomatoes?" },
];

// ── Quick Phrases ─────────────────────────────────────────────
const QUICK_PHRASES = [
  "pest control", "fertilizer dose", "disease symptoms", "spacing guide",
];

// ── SVG Icons ─────────────────────────────────────────────────
const LeafIcon = ({ size = 24, color = THEME.wheat }) => (
  <svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <path d="M12 2C7 2 3 7 3 12c0 3 1.5 5.5 4 7l5-5 5 5c2.5-1.5 4-4 4-7 0-5-4-10-9-10z" fill={color} opacity="0.9"/>
    <path d="M12 22V12M12 12C12 12 8 9 6 6M12 12C12 12 16 9 18 6" stroke={color} strokeWidth="1.5" strokeLinecap="round" opacity="0.6"/>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
    <path d="M22 2L11 13M22 2L15 22L11 13M22 2L2 9L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const SettingsIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
    <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="2"/>
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" stroke="currentColor" strokeWidth="2"/>
  </svg>
);

const TrashIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <polyline points="3 6 5 6 21 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const RefreshIcon = () => (
  <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
    <polyline points="23 4 23 10 17 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

const CloseIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
    <line x1="18" y1="6" x2="6" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
    <line x1="6" y1="6" x2="18" y2="18" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
  </svg>
);

// ── Format timestamp ──────────────────────────────────────────
const fmtTime = (d) => d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

// ── Simulate API Call (replace with real fetch to your FastAPI) 
const simulateAPI = async (query, profile) => {
  await new Promise(r => setTimeout(r, 800 + Math.random() * 600));

  const q = query.toLowerCase();

  // Template-level responses
  if (q.includes("stem borer") && q.includes("rice")) {
    return {
      answer: "To control stem borers in rice: spray Classic-20 at 2 ml/litre of water. Clear field stubbles to remove breeding sites. Apply during tillering and panicle initiation stages for best results. Monitor fields weekly for dead hearts and white ears.",
      source: "template",
    };
  }
  if (q.includes("aphid") && q.includes("lemon")) {
    return {
      answer: "For aphids on lemon trees: spray Malathion 50EC at 2 ml/litre water, every 5 days × 3 times. Avoid spraying on rainy days or when temperatures exceed 35°C. You can also use Neem-based spray (5 ml/litre) as an organic alternative.",
      source: "template",
    };
  }
  if (q.includes("fertilizer") && (q.includes("rice") || q.includes("paddy"))) {
    return {
      answer: "For rice: apply Urea 18 kg/acre + Di-ammonium Phosphate 27 kg/acre + Potassium Chloride 6 kg/acre as base fertilizer. Top-dress with Urea at 9 kg/acre at tillering and panicle initiation stages.",
      source: "template",
    };
  }
  if (q.includes("late blight") || (q.includes("blight") && q.includes("tomato"))) {
    return {
      answer: "For late blight on tomatoes: spray Indofil M-45 at 2.5 g/litre every 7 days. Remove and destroy infected leaves immediately. Improve air circulation by pruning, avoid overhead irrigation, and apply copper-based fungicide as a preventive measure.",
      source: "template",
    };
  }
  if (q.includes("spray") && (q.includes("today") || q.includes("weather") || q.includes("rain"))) {
    return {
      answer: `Based on current weather in ${profile.location || "your area"}: conditions show ${MOCK_WEATHER.rain_pct}% rain probability. ${MOCK_WEATHER.rain_pct > 40 ? "Do NOT spray today — rain will wash off chemicals. Wait for dry weather (2-3 days)." : "Conditions are suitable for spraying. Spray early morning (6–9 AM) or evening to avoid rapid evaporation."}`,
      source: "retrieval",
    };
  }
  if (q.includes("spacing") && q.includes("banana")) {
    return {
      answer: "Banana planting spacing: 1.8 m × 1.8 m between plants and rows. For high-density planting, use 1.5 m × 1.5 m. Ensure proper drainage and choose suckers (sword suckers preferred) of 1–1.5 kg weight.",
      source: "template",
    };
  }

  // Generic SageStorm response
  const answers = [
    `Based on agricultural best practices for ${profile.crop_type || "your crop"}: Consult your local Krishi Vigyan Kendra (KVK) for region-specific advice. Soil health testing every 2 years is recommended to calibrate your fertilizer applications accurately.`,
    `For ${profile.location || "your region"}, the recommended approach is integrated pest management (IPM) — combining biological controls, cultural practices, and targeted chemical application only when pest pressure exceeds economic thresholds.`,
    `Regular monitoring is key. Scout your fields every 5–7 days. Document observations for pattern recognition. Early intervention prevents yield losses of 20–40% commonly seen with delayed treatment.`,
  ];
  return {
    answer: answers[Math.floor(Math.random() * answers.length)],
    source: "sagestorm",
  };
};

// ── Source Badge Component ────────────────────────────────────
const SourceBadge = ({ source }) => {
  const cfg = SOURCE_CONFIG[source] || SOURCE_CONFIG.fallback;
  return (
    <span style={{
      ...styles.sourceTag,
      background: cfg.bg,
      color: cfg.color,
      border: `1px solid ${cfg.border}`,
    }}>
      {cfg.label}
    </span>
  );
};

// ── Typing Indicator ──────────────────────────────────────────
const TypingIndicator = () => (
  <div style={{ ...styles.msgRow, ...styles.msgRowBot }}>
    <div style={{ ...styles.avatar, ...styles.avatarBot }}>
      <LeafIcon size={18} color={THEME.wheat} />
    </div>
    <div style={{ ...styles.bubble, ...styles.bubbleBot }}>
      <div style={styles.typing}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            ...styles.typingDot,
            animationDelay: `${i * 0.2}s`,
          }} />
        ))}
      </div>
    </div>
  </div>
);

// ── Message Component ─────────────────────────────────────────
const Message = ({ msg }) => {
  const isUser = msg.role === "user";
  return (
    <div style={{
      ...styles.msgRow,
      ...(isUser ? styles.msgRowUser : styles.msgRowBot),
    }}>
      <div style={{
        ...styles.avatar,
        ...(isUser ? styles.avatarUser : styles.avatarBot),
      }}>
        {isUser ? "👤" : <LeafIcon size={18} color={THEME.wheat} />}
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, maxWidth: 600 }}>
        <div style={{
          ...styles.bubble,
          ...(isUser ? styles.bubbleUser : styles.bubbleBot),
        }} className="message-bubble">
          {msg.text}
        </div>
        <div style={{
          ...styles.bubbleFooter,
          flexDirection: isUser ? "row-reverse" : "row",
        }}>
          <span style={styles.msgTime}>{fmtTime(msg.time)}</span>
          {!isUser && msg.source && <SourceBadge source={msg.source} />}
        </div>
      </div>
    </div>
  );
};

// ── Settings Modal ────────────────────────────────────────────
const SettingsModal = ({ profile, onSave, onClose }) => {
  const [form, setForm] = useState({ ...profile });
  return (
    <div style={styles.overlay} onClick={e => e.target === e.currentTarget && onClose()}>
      <div style={styles.modal}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h2 style={styles.modalTitle}>Farmer Profile & Settings</h2>
          <button style={styles.iconBtn} onClick={onClose}><CloseIcon /></button>
        </div>

        <div style={styles.formGroup}>
          <label style={styles.label}>Your Name</label>
          <input
            style={styles.input}
            value={form.name || ""}
            onChange={e => setForm(p => ({ ...p, name: e.target.value }))}
            placeholder="e.g. Ramesh Kumar"
          />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Primary Crop</label>
            <select
              style={styles.select}
              value={form.crop_type || ""}
              onChange={e => setForm(p => ({ ...p, crop_type: e.target.value }))}
            >
              <option value="">Select crop</option>
              {["Rice", "Wheat", "Maize", "Tomato", "Potato", "Banana", "Cotton", "Sugarcane", "Mustard", "Onion", "Chilli", "Groundnut"].map(c => (
                <option key={c} value={c.toLowerCase()}>{c}</option>
              ))}
            </select>
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Soil Type</label>
            <select
              style={styles.select}
              value={form.soil_type || ""}
              onChange={e => setForm(p => ({ ...p, soil_type: e.target.value }))}
            >
              <option value="">Select soil</option>
              {["Loamy", "Sandy", "Clay", "Red", "Black", "Alluvial", "Laterite"].map(s => (
                <option key={s} value={s.toLowerCase()}>{s}</option>
              ))}
            </select>
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Location</label>
            <input
              style={styles.input}
              value={form.location || ""}
              onChange={e => setForm(p => ({ ...p, location: e.target.value }))}
              placeholder="City or district"
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Farm Size</label>
            <input
              style={styles.input}
              value={form.farm_size || ""}
              onChange={e => setForm(p => ({ ...p, farm_size: e.target.value }))}
              placeholder="e.g. 3 acres"
            />
          </div>
        </div>

        <div style={{ borderTop: `1px solid ${THEME.wheat}`, paddingTop: 16, marginTop: 8 }}>
          <div style={styles.formGroup}>
            <label style={styles.label}>API City (Weather)</label>
            <input
              style={styles.input}
              value={form.weather_city || "Guwahati"}
              onChange={e => setForm(p => ({ ...p, weather_city: e.target.value }))}
              placeholder="e.g. Mumbai"
            />
          </div>
          <div style={styles.formGroup}>
            <label style={styles.label}>Backend URL</label>
            <input
              style={styles.input}
              value={form.api_url || "http://localhost:8000"}
              onChange={e => setForm(p => ({ ...p, api_url: e.target.value }))}
              placeholder="http://localhost:8000"
            />
          </div>
        </div>

        <div style={styles.btnRow}>
          <button style={styles.btnSecondary} onClick={onClose}>Cancel</button>
          <button style={styles.btnPrimary} onClick={() => { onSave(form); onClose(); }}>
            Save Profile
          </button>
        </div>
      </div>
    </div>
  );
};

// ── Main App ──────────────────────────────────────────────────
export default function StromSageUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [profile, setProfile]   = useState({
    name: "",
    crop_type: "",
    location: "Guwahati",
    soil_type: "",
    farm_size: "",
    weather_city: "Guwahati",
    api_url: "http://localhost:8000",
  });
  const [weather, setWeather]   = useState(MOCK_WEATHER);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [inputFocused, setInputFocused] = useState(false);

  const messagesEndRef = useRef(null);
  const textareaRef    = useRef(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + "px";
    }
  }, [input]);

  // Load profile from server on mount
  useEffect(() => {
    fetch(`${profile.api_url}/profile`)
      .then(res => res.ok ? res.json() : {})
      .then(data => setProfile(p => ({ ...p, ...data })))
      .catch(() => {}); // ignore errors
  }, [profile.api_url]);

  // Load weather
  useEffect(() => {
    fetch(`${profile.api_url}/weather?city=${encodeURIComponent(profile.weather_city || "Guwahati")}`)
      .then(res => res.ok ? res.json() : MOCK_WEATHER)
      .then(data => setWeather(data))
      .catch(() => setWeather(MOCK_WEATHER));
  }, [profile.api_url, profile.weather_city]);

  // Extract profile updates from messages
  const updateProfileFromText = useCallback((text) => {
    const updates = {};
    const cropMatch = text.match(/\b(?:growing|farming|planting)\s+(\w+)/i);
    if (cropMatch) updates.crop_type = cropMatch[1].toLowerCase();
    const locMatch = text.match(/\bnear\s+([A-Z][a-z]+)/);
    if (locMatch) updates.location = locMatch[1];
    const soilMatch = text.match(/\b(loamy|sandy|clay|red|black|alluvial)\s+soil/i);
    if (soilMatch) updates.soil_type = soilMatch[1].toLowerCase();
    const sizeMatch = text.match(/(\d+(?:\.\d+)?)\s*(?:acre|bigha|hectare)/i);
    if (sizeMatch) updates.farm_size = sizeMatch[0];
    if (Object.keys(updates).length > 0) {
      setProfile(p => ({ ...p, ...updates }));
    }
  }, []);

  const sendMessage = useCallback(async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;

    setInput("");
    const userMsg = { id: Date.now(), role: "user", text: q, time: new Date() };
    setMessages(prev => [...prev, userMsg]);
    updateProfileFromText(q);

    // Save to session history
    setSessionHistory(prev => [
      { text: q, time: new Date() },
      ...prev.slice(0, 9),
    ]);

    setLoading(true);
    try {
      // Try real API first, fall back to simulation
      let data;
      try {
        const res = await fetch(`${profile.api_url}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, city: profile.weather_city || "Guwahati" }),
          signal: AbortSignal.timeout(8000),
        });
        if (res.ok) data = await res.json();
        else throw new Error("API error");
      } catch {
        // Fallback to simulation
        data = await simulateAPI(q, profile);
      }

      const botMsg = {
        id: Date.now() + 1,
        role: "bot",
        text: data.answer || data.response || "I couldn't generate a response.",
        source: data.source || "sagestorm",
        time: new Date(),
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: "bot",
        text: "Something went wrong. Please check your connection and try again.",
        source: "fallback",
        time: new Date(),
      }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, profile, updateProfileFromText]);

  const clearChat = () => {
    setMessages([]);
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Inject Google Fonts + keyframes */}
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Crimson+Pro:wght@400;600&family=Source+Code+Pro:wght@400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: ${THEME.parchment}; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-5px); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: ${THEME.cream}; }
        ::-webkit-scrollbar-thumb { background: ${THEME.wheat}; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: ${THEME.straw}; }
        textarea::placeholder { color: ${THEME.straw}; opacity: 0.75; }
        .suggestion-card { transition: all 0.25s ease; }
        .suggestion-card:hover { 
          background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, ${THEME.wheat}25 100%) !important; 
          border-color: ${THEME.sprout} !important; 
          transform: translateY(-4px); 
          box-shadow: 0 16px 36px rgba(0,0,0,0.12) !important;
        }
        .profile-card { transition: all 0.3s ease; }
        .profile-card:hover { transform: translateY(-3px); box-shadow: 0 8px 22px rgba(0,0,0,0.12) !important; }
        .weather-card { transition: all 0.3s ease; }
        .weather-card:hover { transform: translateY(-3px); box-shadow: 0 10px 26px rgba(0,0,0,0.12) !important; }
        .input-row { transition: all 0.3s ease; }
        .input-row:focus-within { border-color: ${THEME.sprout} !important; box-shadow: 0 0 0 4px rgba(140,192,111,0.18) !important; }
        .message-bubble { animation: fadeIn 0.4s ease-out; }
        .quick-btn:hover { background: ${THEME.wheat}25 !important; border-color: ${THEME.sage} !important; }
        .history-item:hover { background: ${THEME.wheat}35 !important; }
        .icon-btn:hover { background: ${THEME.wheat}30 !important; }
        .send-btn:hover:not(:disabled) { background: ${THEME.sprout} !important; transform: scale(1.04); }
      `}</style>

      <div style={styles.root}>
        {/* ── Header ─────────────────────────────────────── */}
        <header style={styles.header}>
          <div style={styles.logo}>
            <svg style={styles.logoIcon} viewBox="0 0 34 34" fill="none">
              <circle cx="17" cy="17" r="16" fill={THEME.moss} />
              <path d="M17 6C12 6 8 11 8 16c0 2.5 1 4.5 3 6l6-6 6 6c2-1.5 3-3.5 3-6 0-5-4-10-9-10z" fill={THEME.sprout}/>
              <path d="M17 28V18M17 18C17 18 14 16 12 13M17 18C17 18 20 16 22 13" stroke={THEME.wheat} strokeWidth="1.5" strokeLinecap="round"/>
            </svg>
            <div>
              <span style={styles.logoText}>Strom Sage</span>
              <span style={styles.logoSub}>Agriculture AI · v2.0</span>
            </div>
          </div>

          <div style={styles.headerRight}>
            <div style={styles.statusBadge}>
              <div style={styles.statusDot} />
              SageStorm V2 · RAG Active
            </div>
            <button
              style={{ ...styles.iconBtn, border: `1px solid ${THEME.sage}44`, color: THEME.wheat }}
              className="icon-btn"
              onClick={() => setShowSettings(true)}
            >
              <SettingsIcon />
            </button>
            {messages.length > 0 && (
              <button
                style={{ ...styles.iconBtn, border: `1px solid ${THEME.sage}44`, color: THEME.wheat }}
                className="icon-btn"
                onClick={clearChat}
                title="Clear chat"
              >
                <TrashIcon />
              </button>
            )}
          </div>
        </header>

        {/* ── Main Layout ─────────────────────────────────── */}
        <div style={styles.main}>
          {/* ── Left Sidebar ────────────────────────────── */}
          <aside style={styles.sidebar}>
            {/* Farmer Profile */}
            <div style={styles.sidebarSection}>
              <div style={styles.sidebarTitle}>Farmer Profile</div>
              <div style={styles.profileCard} className="profile-card">
                {profile.name && (
                  <div style={styles.profileRow}>
                    <span>👤</span>
                    <span style={styles.profileValue}>{profile.name}</span>
                  </div>
                )}
                {profile.crop_type && (
                  <div style={styles.profileRow}>
                    <span>🌾</span>
                    <span style={styles.profileLabel}>Crop</span>
                    <span style={styles.profileValue}>{profile.crop_type}</span>
                  </div>
                )}
                {profile.location && (
                  <div style={styles.profileRow}>
                    <span>📍</span>
                    <span style={styles.profileLabel}>Location</span>
                    <span style={styles.profileValue}>{profile.location}</span>
                  </div>
                )}
                {profile.soil_type && (
                  <div style={styles.profileRow}>
                    <span>🪨</span>
                    <span style={styles.profileLabel}>Soil</span>
                    <span style={styles.profileValue}>{profile.soil_type}</span>
                  </div>
                )}
                {profile.farm_size && (
                  <div style={styles.profileRow}>
                    <span>📐</span>
                    <span style={styles.profileLabel}>Farm</span>
                    <span style={styles.profileValue}>{profile.farm_size}</span>
                  </div>
                )}
                {!profile.name && !profile.crop_type && (
                  <div style={{ fontSize: 12, color: THEME.straw, lineHeight: 1.5 }}>
                    Tell me about your farm — I'll remember your crops, location, and soil type automatically.
                  </div>
                )}
                <button
                  style={{
                    marginTop: 10,
                    width: "100%",
                    padding: "7px",
                    background: "transparent",
                    border: `1px dashed ${THEME.wheat}`,
                    borderRadius: 7,
                    fontSize: 12,
                    color: THEME.bark,
                    cursor: "pointer",
                    fontFamily: "'Source Code Pro', monospace",
                  }}
                  onClick={() => setShowSettings(true)}
                >
                  + Edit Profile
                </button>
              </div>
            </div>

            {/* Weather */}
            <div style={styles.sidebarSection}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={styles.sidebarTitle}>Weather</div>
                <button style={{ ...styles.iconBtn, width: 24, height: 24, border: "none" }} className="icon-btn">
                  <RefreshIcon />
                </button>
              </div>
              <div style={styles.weatherCard} className="weather-card">
                <div style={styles.weatherCity}>
                  📍 {profile.weather_city || "Guwahati"}
                  <span style={{ fontSize: 10, color: THEME.sky, marginLeft: 6 }}>{weather.src}</span>
                </div>
                <div style={{ fontSize: 28, fontWeight: 700, color: THEME.sky, margin: "6px 0" }}>
                  {weather.temp}°C
                </div>
                <div style={styles.weatherGrid}>
                  <div style={styles.weatherItem}>
                    💧 <span style={styles.weatherVal}>{weather.humid}%</span> humid
                  </div>
                  <div style={styles.weatherItem}>
                    🌬 <span style={styles.weatherVal}>{weather.wind} km/h</span>
                  </div>
                  <div style={styles.weatherItem} >
                    ☁️ {weather.desc}
                  </div>
                  <div style={styles.weatherItem}>
                    ☔ <span style={styles.weatherVal}>{weather.rain_pct}%</span> rain
                  </div>
                </div>
                {weather.rain_pct > 50 && (
                  <div style={styles.weatherWarn}>
                    ⚠️ Rain likely — avoid spraying today
                  </div>
                )}
                {!weather.rain && (
                  <div style={{
                    ...styles.weatherWarn,
                    background: "#e8f5e222",
                    border: `1px solid ${THEME.sage}33`,
                    color: THEME.sage,
                    marginTop: 8,
                  }}>
                    ✓ Good conditions for field operations
                  </div>
                )}
              </div>
            </div>

            {/* Model Info */}
            <div style={styles.sidebarSection}>
              <div style={styles.sidebarTitle}>Model Info</div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {[
                  ["Model", "SageStorm V2"],
                  ["Parameters", "48M"],
                  ["Context", "512 tokens"],
                  ["Architecture", "GPT + GQA + RoPE"],
                  ["RAG", "Word2Vec + BM25"],
                  ["Dataset", "337K samples"],
                ].map(([k, v]) => (
                  <div key={k} style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 12,
                    borderBottom: `1px solid ${THEME.wheat}55`,
                    paddingBottom: 4,
                  }}>
                    <span style={{ color: THEME.straw, fontFamily: "'Source Code Pro', monospace" }}>{k}</span>
                    <span style={{ color: THEME.bark, fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Response Legend */}
            <div style={{ padding: "12px 16px" }}>
              <div style={styles.sidebarTitle}>Response Sources</div>
              {Object.entries(SOURCE_CONFIG).map(([key, cfg]) => (
                <div key={key} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                  <span style={{
                    ...styles.sourceTag,
                    background: cfg.bg,
                    color: cfg.color,
                    border: `1px solid ${cfg.border}`,
                    fontSize: 9,
                    padding: "2px 6px",
                  }}>
                    {cfg.label}
                  </span>
                  <span style={{ fontSize: 11, color: THEME.straw }}>
                    {key === "template" && "Rule-based"}
                    {key === "retrieval" && "RAG document"}
                    {key === "sagestorm" && "SLM generated"}
                    {key === "fallback" && "Safe default"}
                  </span>
                </div>
              ))}
            </div>
          </aside>

          {/* ── Chat Area ──────────────────────────────────── */}
          <main style={styles.chatArea}>
            <div style={styles.messages}>
              {messages.length === 0 ? (
                /* Welcome Screen */
                <div style={styles.welcome}>
                  <svg style={styles.welcomeIcon} viewBox="0 0 80 80" fill="none">
                    <circle cx="40" cy="40" r="38" fill={THEME.moss} opacity="0.1" stroke={THEME.sage} strokeWidth="1.5"/>
                    <path d="M40 15C28 15 18 26 18 38c0 6 2.5 11 7 14.5L40 38l15 14.5C59.5 49 62 44 62 38c0-12-10-23-22-23z" fill={THEME.sage}/>
                    <path d="M40 68V44M40 44C40 44 32 39 27 31M40 44C40 44 48 39 53 31" stroke={THEME.moss} strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="40" cy="40" r="4" fill={THEME.wheat}/>
                  </svg>
                  <h1 style={styles.welcomeTitle}>
                    Namaste! I'm Strom Sage 🌿
                  </h1>
                  <p style={styles.welcomeSub}>
                    Your intelligent agriculture advisor — powered by SageStorm V2, a 48M-parameter domain-specific AI with RAG retrieval and live weather integration. Ask me anything about crops, pests, fertilizers, or best practices.
                  </p>
                  <div style={styles.suggestionsGrid}>
                    {SUGGESTIONS.map((s, i) => (
                      <button
                        key={i}
                        style={styles.suggestionCard}
                        className="suggestion-card"
                        onClick={() => sendMessage(s.text)}
                      >
                        <span style={styles.suggestionEmoji}>{s.emoji}</span>
                        <span style={styles.suggestionText}>{s.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                /* Messages */
                <>
                  {messages.map(msg => (
                    <Message key={msg.id} msg={msg} />
                  ))}
                  {loading && <TypingIndicator />}
                </>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* ── Input Area ──────────────────────────── */}
            <div style={styles.inputArea}>
              <div
                style={{
                  ...styles.inputRow,
                  ...(inputFocused ? { borderColor: THEME.sage, boxShadow: `0 0 0 3px ${THEME.sage}22` } : {}),
                }}
                className="input-row"
              >
                <textarea
                  ref={textareaRef}
                  style={styles.textarea}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={handleKeyDown}
                  onFocus={() => setInputFocused(true)}
                  onBlur={() => setInputFocused(false)}
                  placeholder="Ask about crops, pests, fertilizers, weather..."
                  rows={1}
                  disabled={loading}
                />
                <button
                  style={{
                    ...styles.sendBtn,
                    ...((!input.trim() || loading) ? styles.sendBtnDisabled : {}),
                  }}
                  className="send-btn"
                  onClick={() => sendMessage()}
                  disabled={!input.trim() || loading}
                >
                  <SendIcon />
                </button>
              </div>

              <div style={styles.inputHints}>
                <span style={styles.hintText}>
                  {loading ? "⟳ Thinking..." : "Press Enter to send · Shift+Enter for new line"}
                </span>
                <div style={styles.quickBtns}>
                  {QUICK_PHRASES.map(p => (
                    <button
                      key={p}
                      style={styles.quickBtn}
                      className="quick-btn"
                      onClick={() => setInput(prev => prev ? `${prev} ${p}` : p)}
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </main>

          {/* ── Right Panel ────────────────────────────────── */}
          <aside style={styles.rightPanel}>
            <div style={styles.rightSection}>
              <div style={styles.sidebarTitle}>Session History</div>
              {sessionHistory.length === 0 ? (
                <div style={{ fontSize: 12, color: THEME.straw, lineHeight: 1.6 }}>
                  Your recent questions will appear here.
                </div>
              ) : (
                sessionHistory.map((item, i) => (
                  <div
                    key={i}
                    style={styles.historyItem}
                    className="history-item"
                    onClick={() => setInput(item.text)}
                  >
                    <div style={styles.historyText}>{item.text}</div>
                    <div style={styles.historyTime}>{fmtTime(item.time)}</div>
                  </div>
                ))
              )}
            </div>

            {/* Stats */}
            <div style={styles.rightSection}>
              <div style={styles.sidebarTitle}>Session Stats</div>
              {[
                ["Messages", messages.length],
                ["Bot responses", messages.filter(m => m.role === "bot").length],
                ["Template hits", messages.filter(m => m.source === "template").length],
                ["RAG hits", messages.filter(m => m.source === "retrieval").length],
              ].map(([k, v]) => (
                <div key={k} style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  marginBottom: 8,
                  fontSize: 12,
                }}>
                  <span style={{ color: THEME.straw }}>{k}</span>
                  <span style={{
                    background: THEME.wheat,
                    color: THEME.bark,
                    borderRadius: 12,
                    padding: "2px 8px",
                    fontFamily: "'Source Code Pro', monospace",
                    fontSize: 11,
                    fontWeight: 600,
                  }}>{v}</span>
                </div>
              ))}
            </div>

            {/* Tips */}
            <div style={{ padding: "12px 16px", flex: 1 }}>
              <div style={styles.sidebarTitle}>Quick Tips</div>
              <div style={{
                background: `${THEME.sage}15`,
                border: `1px solid ${THEME.sage}33`,
                borderRadius: 8,
                padding: "10px 12px",
                fontSize: 12,
                color: THEME.bark,
                lineHeight: 1.6,
              }}>
                💡 Mention your crop, location, and soil type in your first message for more personalized advice.
              </div>
              <div style={{
                background: `${THEME.sky}15`,
                border: `1px solid ${THEME.sky}33`,
                borderRadius: 8,
                padding: "10px 12px",
                fontSize: 12,
                color: THEME.bark,
                lineHeight: 1.6,
                marginTop: 8,
              }}>
                🌧️ Ask "should I spray today?" and I'll check the weather before answering.
              </div>
              <div style={{
                background: `${THEME.straw}25`,
                border: `1px solid ${THEME.straw}44`,
                borderRadius: 8,
                padding: "10px 12px",
                fontSize: 12,
                color: THEME.bark,
                lineHeight: 1.6,
                marginTop: 8,
              }}>
                🔗 Connect to your backend by setting the API URL in Settings.
              </div>
            </div>
          </aside>
        </div>
      </div>

      {/* ── Settings Modal ──────────────────────────────────── */}
      {showSettings && (
        <SettingsModal
          profile={profile}
          onSave={(form) => {
            setProfile(form);
            const profileData = {
              name: form.name,
              crop_type: form.crop_type,
              location: form.location,
              soil_type: form.soil_type,
              farm_size: form.farm_size,
            };
            fetch(`${profile.api_url}/profile`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify(profileData),
            }).catch(() => {}); // ignore errors
          }}
          onClose={() => setShowSettings(false)}
        />
      )}
    </>
  );
}
