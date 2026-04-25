import { useState, useRef, useEffect, useCallback } from "react";

// ── Design Tokens (Modern Enhanced) ───────────────────────────
const THEME = {
  // Modern vibrant palette
  primary:    "#10B981",    // Emerald green
  secondary:  "#8B5CF6",    // Purple
  accent:     "#F59E0B",    // Amber
  danger:     "#EF4444",    // Red
  
  // Refined farming colors
  soil:       "#1F2937",    // Dark gray
  bark:       "#374151",    // Medium gray  
  moss:       "#059669",    // Deep emerald
  leaf:       "#10B981",    // Bright emerald
  sage:       "#34D399",    // Light green
  sprout:     "#6EE7B7",    // Mint green
  straw:      "#94A3B8",    // Slate
  wheat:      "#F3F4F6",    // Light gray
  cream:      "#F9FAFB",    // Off-white
  parchment:  "#FFFFFF",    // Pure white
  
  // Extended palette
  sky:        "#0284C7",    // Deep blue
  rain:       "#06B6D4",    // Cyan
  cloud:      "#E0F2FE",    // Light blue
  emerald:    "#10B981",    // Emerald
  amber:      "#F59E0B",    // Amber
  indigo:     "#6366F1",    // Indigo
  rose:       "#F43F5E",    // Rose
  error:      "#DC2626",    // Red
  warn:       "#D97706",    // Orange
  info:       "#0284C7",    // Blue
  success:    "#10B981",    // Green
};

// ── Inline Styles (no external CSS) ───────────────────────────
const styles = {
  root: {
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    background: "#FFFFFF",
    height: "100vh",
    display: "flex",
    flexDirection: "column",
    color: THEME.soil,
    overflow: "hidden",
  },
  // Header
  header: {
    background: `linear-gradient(135deg, ${THEME.leaf} 0%, ${THEME.primary} 50%, ${THEME.sage} 100%)`,
    borderBottom: "none",
    padding: "0 32px",
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    height: 72,
    flexShrink: 0,
    position: "sticky",
    top: 0,
    zIndex: 100,
    boxShadow: "0 12px 32px rgba(16, 185, 129, 0.15)",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: 12,
    cursor: "pointer",
    userSelect: "none",
  },
  logoIcon: {
    width: 36,
    height: 36,
  },
  logoText: {
    fontFamily: "'Poppins', sans-serif",
    fontSize: 20,
    fontWeight: 700,
    color: "#FFFFFF",
    letterSpacing: "-0.01em",
  },
  logoSub: {
    fontSize: 11,
    color: "rgba(255,255,255,0.9)",
    letterSpacing: "0.05em",
    textTransform: "uppercase",
    marginTop: 0,
    display: "block",
    fontWeight: 500,
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
    background: "rgba(255,255,255,0.2)",
    border: "1px solid rgba(255,255,255,0.4)",
    borderRadius: 20,
    padding: "6px 14px",
    fontSize: 12,
    color: "#FFFFFF",
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.05em",
    fontWeight: 500,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: "50%",
    background: "#FFFFFF",
    animation: "pulse 2s infinite",
  },
  // Main layout
  main: {
    display: "flex",
    flexDirection: "row",
    flex: 1,
    overflow: "hidden",
    height: "calc(100vh - 72px)",
    background: "#F8FAFC",
    alignItems: "stretch",
  },
  // Sidebar
  sidebar: {
    width: 300,
    background: "#FFFFFF",
    borderRight: "1px solid #E2E8F0",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    flexShrink: 0,
    height: "100%",
    boxShadow: "2px 0 12px rgba(0,0,0,0.05)",
    position: "sticky",
    top: 72,
  },
  sidebarSection: {
    padding: "18px 20px 12px",
    borderBottom: "1px solid #E2E8F0",
  },
  sidebarTitle: {
    fontSize: 11,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.15em",
    textTransform: "uppercase",
    color: THEME.primary,
    marginBottom: 12,
    fontWeight: 700,
  },
  profileCard: {
    background: `linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%)`,
    border: `1.5px solid ${THEME.sage}40`,
    borderRadius: 16,
    padding: "16px 16px",
    boxShadow: "0 4px 12px rgba(16, 185, 129, 0.08)",
    transition: "transform 0.25s ease, box-shadow 0.25s ease",
  },
  profileCardHover: {
    transform: "translateY(-2px)",
    boxShadow: "0 8px 20px rgba(16, 185, 129, 0.15)",
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
    background: `linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%)`,
    border: `1.5px solid ${THEME.sky}30`,
    borderRadius: 16,
    padding: "16px 16px",
    boxShadow: "0 4px 12px rgba(2, 132, 199, 0.08)",
    transition: "transform 0.25s ease, box-shadow 0.25s ease",
  },
  weatherCity: {
    fontSize: 13,
    fontWeight: 700,
    color: THEME.sky,
    marginBottom: 8,
  },
  weatherGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 8,
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
    marginTop: 10,
    background: `${THEME.warn}15`,
    border: `1px solid ${THEME.warn}40`,
    borderRadius: 8,
    padding: "8px 10px",
    fontSize: 11,
    color: THEME.warn,
    display: "flex",
    gap: 6,
    alignItems: "flex-start",
    fontWeight: 500,
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
    padding: "32px 48px",
    display: "flex",
    flexDirection: "column",
    gap: 20,
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
    padding: "60px 48px",
    textAlign: "center",
    flex: 1,
  },
  welcomeIcon: {
    width: 120,
    height: 120,
    marginBottom: 32,
    filter: "drop-shadow(0 8px 16px rgba(0,0,0,0.1))",
  },
  welcomeTitle: {
    fontFamily: "'Poppins', sans-serif",
    fontSize: 40,
    fontWeight: 700,
    color: THEME.primary,
    marginBottom: 16,
    lineHeight: 1.15,
  },
  welcomeSub: {
    fontSize: 16,
    color: THEME.bark,
    maxWidth: 700,
    lineHeight: 1.7,
    marginBottom: 48,
  },
  suggestionsGrid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 12,
    width: "100%",
    maxWidth: 720,
  },
  suggestionCard: {
    background: `linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%)`,
    border: `1.5px solid #E2E8F0`,
    borderRadius: 16,
    padding: "16px 18px",
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.25s ease",
    display: "flex",
    alignItems: "flex-start",
    gap: 12,
    boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
  },
  suggestionCardHover: {
    transform: "translateY(-4px)",
    boxShadow: "0 12px 28px rgba(16, 185, 129, 0.15)",
    borderColor: THEME.primary,
  },
  suggestionEmoji: {
    fontSize: 22,
    lineHeight: 1,
    flexShrink: 0,
    marginTop: 1,
  },
  suggestionText: {
    fontSize: 14,
    color: THEME.soil,
    lineHeight: 1.5,
    fontWeight: 500,
  },
  // Message bubbles
  msgRow: {
    display: "flex",
    gap: 12,
    alignItems: "flex-start",
    maxWidth: 900,
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
    fontWeight: 600,
  },
  avatarBot: {
    background: `linear-gradient(135deg, ${THEME.leaf} 0%, ${THEME.sage} 100%)`,
    border: "none",
    color: "#FFFFFF",
  },
  avatarUser: {
    background: `linear-gradient(135deg, ${THEME.primary} 0%, ${THEME.leaf} 100%)`,
    border: "none",
    color: "#FFFFFF",
  },
  bubble: {
    borderRadius: 18,
    padding: "14px 18px",
    maxWidth: 700,
    lineHeight: 1.6,
    fontSize: 15,
    position: "relative",
    boxShadow: "0 4px 12px rgba(0,0,0,0.06)",
    transition: "transform 0.25s ease",
  },
  bubbleUser: {
    background: `linear-gradient(135deg, ${THEME.primary} 0%, ${THEME.leaf} 100%)`,
    color: "#FFFFFF",
    borderBottomRightRadius: 6,
  },
  bubbleBot: {
    background: "#FFFFFF",
    color: THEME.soil,
    border: "1px solid #E2E8F0",
    borderBottomLeftRadius: 6,
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
    padding: "12px 16px",
    alignItems: "center",
  },
  typingDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    background: THEME.primary,
    animation: "bounce 1.2s infinite",
  },
  // Input area
  inputArea: {
    borderTop: "1px solid #E2E8F0",
    background: "#FFFFFF",
    padding: "20px 32px 28px",
    boxShadow: "0 -4px 16px rgba(0,0,0,0.04)",
  },
  inputRow: {
    display: "flex",
    gap: 12,
    alignItems: "flex-end",
    background: "#F8FAFC",
    border: "1.5px solid #E2E8F0",
    borderRadius: 28,
    padding: "14px 20px 14px 20px",
    transition: "all 0.3s ease",
    boxShadow: "0 4px 12px rgba(0,0,0,0.04)",
  },
  inputRowFocus: {
    borderColor: THEME.primary,
    boxShadow: "0 0 0 3px rgba(16, 185, 129, 0.1)",
  },
  textarea: {
    flex: 1,
    border: "none",
    outline: "none",
    resize: "none",
    background: "transparent",
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    fontSize: 15,
    color: THEME.soil,
    lineHeight: 1.5,
    minHeight: 24,
    maxHeight: 120,
    paddingTop: 0,
  },
  sendBtn: {
    width: 40,
    height: 40,
    borderRadius: 12,
    border: "none",
    background: THEME.primary,
    color: "#FFFFFF",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    flexShrink: 0,
    transition: "all 0.2s ease",
    boxShadow: "0 4px 12px rgba(16, 185, 129, 0.2)",
  },
  sendBtnDisabled: {
    background: "#E2E8F0",
    color: THEME.straw,
    cursor: "not-allowed",
    boxShadow: "none",
  },
  inputHints: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginTop: 10,
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
    border: `1px solid #E2E8F0`,
    borderRadius: 12,
    padding: "6px 12px",
    fontSize: 11,
    color: THEME.bark,
    cursor: "pointer",
    fontFamily: "'Source Code Pro', monospace",
    transition: "all 0.2s ease",
    fontWeight: 500,
  },
  // Right panel
  rightPanel: {
    width: 280,
    background: "#FFFFFF",
    borderLeft: "1px solid #E2E8F0",
    display: "flex",
    flexDirection: "column",
    overflow: "hidden",
    flexShrink: 0,
    height: "100%",
    boxShadow: "-2px 0 12px rgba(0,0,0,0.05)",
    position: "sticky",
    top: 72,
  },
  rightSection: {
    padding: "18px 20px",
    borderBottom: "1px solid #E2E8F0",
  },
  historyItem: {
    padding: "10px 12px",
    borderRadius: 10,
    cursor: "pointer",
    marginBottom: 6,
    transition: "background 0.15s",
  },
  historyText: {
    fontSize: 12,
    color: THEME.bark,
    lineHeight: 1.4,
    overflow: "hidden",
    textOverflow: "ellipsis",
    whiteSpace: "nowrap",
  },
  historyTime: {
    fontSize: 10,
    color: THEME.straw,
    fontFamily: "'Source Code Pro', monospace",
    marginTop: 4,
  },
  // Modal / Settings
  overlay: {
    position: "fixed",
    inset: 0,
    background: "rgba(0, 0, 0, 0.5)",
    zIndex: 200,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
  modal: {
    background: "#FFFFFF",
    border: "1.5px solid #E2E8F0",
    borderRadius: 20,
    padding: 32,
    width: 480,
    maxWidth: "90vw",
    boxShadow: "0 20px 60px rgba(0, 0, 0, 0.15)",
  },
  modalTitle: {
    fontFamily: "'Poppins', sans-serif",
    fontSize: 22,
    fontWeight: 700,
    color: THEME.primary,
    marginBottom: 24,
  },
  formGroup: {
    marginBottom: 18,
  },
  label: {
    display: "block",
    fontSize: 12,
    color: THEME.bark,
    fontFamily: "'Source Code Pro', monospace",
    letterSpacing: "0.1em",
    textTransform: "uppercase",
    marginBottom: 8,
    fontWeight: 600,
  },
  input: {
    width: "100%",
    padding: "10px 14px",
    border: "1.5px solid #E2E8F0",
    borderRadius: 10,
    background: "#F8FAFC",
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    fontSize: 14,
    color: THEME.soil,
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.15s, box-shadow 0.15s",
  },
  select: {
    width: "100%",
    padding: "10px 14px",
    border: "1.5px solid #E2E8F0",
    borderRadius: 10,
    background: "#F8FAFC",
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    fontSize: 14,
    color: THEME.soil,
    outline: "none",
    boxSizing: "border-box",
  },
  btnRow: {
    display: "flex",
    gap: 12,
    justifyContent: "flex-end",
    marginTop: 28,
  },
  btnPrimary: {
    background: THEME.primary,
    color: "#FFFFFF",
    border: "none",
    borderRadius: 12,
    padding: "11px 26px",
    fontSize: 14,
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    cursor: "pointer",
    fontWeight: 600,
    transition: "background 0.15s, transform 0.1s",
    boxShadow: "0 4px 12px rgba(16, 185, 129, 0.2)",
  },
  btnSecondary: {
    background: "#F8FAFC",
    color: THEME.bark,
    border: "1.5px solid #E2E8F0",
    borderRadius: 12,
    padding: "11px 26px",
    fontSize: 14,
    fontFamily: "'Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', sans-serif",
    cursor: "pointer",
    fontWeight: 600,
    transition: "all 0.15s",
  },
  iconBtn: {
    background: "transparent",
    border: "1px solid #E2E8F0",
    borderRadius: 10,
    width: 36,
    height: 36,
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
  template:   { bg: "#DBEAFE", color: "#0C4A6E", border: "#0284C744", label: "✦ template" },
  retrieval:  { bg: "#D1FAE5", color: "#065F46", border: "#10B98144", label: "⟳ retrieval" },
  sagestorm:  { bg: "#FEE2E2", color: "#7F1D1D", border: "#DC262644", label: "✧ sagestorm v2" },
  fallback:   { bg: "#FEF3C7", color: "#78350F", border: "#D9703044", label: "⚠ fallback" },
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
const LeafIcon = ({ size = 24, color = "#FFFFFF" }) => (
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

const normalizeSource = (source) => {
  if (!source) return "fallback";
  if (source === "template") return "template";
  if (["retrieval", "retrieval_summarized", "retrieval_fallback", "retrieval_raw"].includes(source)) return "retrieval";
  if (["rag_generated", "sagestorm"].includes(source)) return "sagestorm";
  if (["fallback", "error"].includes(source)) return "fallback";
  return "fallback";
};

// ── Source Badge Component ────────────────────────────────────
const SourceBadge = ({ source }) => {
  const normalized = normalizeSource(source);
  const cfg = SOURCE_CONFIG[normalized] || SOURCE_CONFIG.fallback;
  return (
    <span style={{
      ...styles.sourceTag,
      background: cfg.bg,
      color: cfg.color,
      border: `1px solid ${cfg.border}`,
      fontSize: 10,
      padding: "3px 8px",
    }}>
      {cfg.label}
    </span>
  );
};

// ── Typing Indicator ──────────────────────────────────────────
const TypingIndicator = () => (
  <div style={{ ...styles.msgRow, ...styles.msgRowBot }}>
    <div style={{ ...styles.avatar, ...styles.avatarBot }}>
      <LeafIcon size={18} color="#FFFFFF" />
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
        {isUser ? "👤" : <LeafIcon size={18} color="#FFFFFF" />}
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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 24 }}>
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

        <div style={{ borderTop: `1px solid #E2E8F0`, paddingTop: 16, marginTop: 8 }}>
          <div style={styles.formGroup}>
            <label style={styles.label}>Weather City (Optional)</label>
            <input
              style={styles.input}
              value={form.weather_city || ""}
              onChange={e => setForm(p => ({ ...p, weather_city: e.target.value }))}
              placeholder="Leave empty to use auto-detected location"
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
    location: "",
    soil_type: "",
    farm_size: "",
    weather_city: "",
    api_url: "http://localhost:8000",
  });
  const [weather, setWeather]   = useState(MOCK_WEATHER);
  const [sessionHistory, setSessionHistory] = useState([]);
  const [inputFocused, setInputFocused] = useState(false);
  const [userLocation, setUserLocation] = useState({ lat: null, lon: null, city: "" });

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

  // Geolocation on mount
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const { latitude, longitude } = position.coords;
          setUserLocation({ lat: latitude, lon: longitude, city: "" });
          // Update profile with user's location
          setProfile(p => ({
            ...p,
            weather_city: p.weather_city || "",
            location: p.location || "",
          }));
        },
        (error) => {
          console.log("Geolocation denied or unavailable:", error);
          setUserLocation({ lat: null, lon: null, city: "Guwahati" });
        }
      );
    }
  }, []);

  // Load profile from server on mount
  useEffect(() => {
    fetch(`${profile.api_url}/profile`)
      .then(res => res.ok ? res.json() : {})
      .then(data => setProfile(p => ({ ...p, ...data })))
      .catch(() => {}); // ignore errors
  }, [profile.api_url]);

  // Load weather - use user's location if available
  useEffect(() => {
    const weatherCity = profile.weather_city || userLocation.city || "Guwahati";
    fetch(`${profile.api_url}/weather?city=${encodeURIComponent(weatherCity)}`)
      .then(res => res.ok ? res.json() : MOCK_WEATHER)
      .then(data => setWeather(data))
      .catch(() => setWeather(MOCK_WEATHER));
  }, [profile.api_url, profile.weather_city, userLocation.city]);

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
      const res = await fetch(`${profile.api_url}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, city: profile.weather_city || "Guwahati" }),
        signal: AbortSignal.timeout(30000),
      });

      if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Backend error ${res.status}: ${errorText}`);
      }

      const data = await res.json();
      const botMsg = {
        id: Date.now() + 1,
        role: "bot",
        text: data.answer || data.response || "I couldn't generate a response.",
        source: data.source || "sagestorm",
        time: new Date(),
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (err) {
      const messageText = err?.message
        ? `Backend request failed: ${err.message}`
        : "Something went wrong. Please check your connection and try again.";
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: "bot",
        text: messageText,
        source: "error",
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
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600&family=Source+Code+Pro:wght@400;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        html { scroll-behavior: smooth; }
        body { background: #FFFFFF; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        @keyframes bounce {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-6px); }
        }
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes slideIn {
          from { opacity: 0; transform: translateX(-10px); }
          to { opacity: 1; transform: translateX(0); }
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #F8FAFC; }
        ::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
        textarea::placeholder { color: #94A3B8; opacity: 0.8; }
        input::placeholder { color: #94A3B8; opacity: 0.8; }
        .suggestion-card { animation: slideIn 0.3s ease-out; }
        .suggestion-card:hover { 
          background: linear-gradient(135deg, #FFFFFF 0%, #F0FDF4 100%) !important; 
          border-color: ${THEME.primary} !important; 
          transform: translateY(-6px) !important; 
          box-shadow: 0 20px 36px rgba(16, 185, 129, 0.15) !important;
        }
        .profile-card { transition: all 0.3s ease; }
        .profile-card:hover { transform: translateY(-2px); box-shadow: 0 12px 24px rgba(16, 185, 129, 0.12) !important; }
        .weather-card { transition: all 0.3s ease; }
        .weather-card:hover { transform: translateY(-2px); box-shadow: 0 12px 24px rgba(2, 132, 199, 0.12) !important; }
        .input-row { transition: all 0.3s ease; }
        .input-row:focus-within { border-color: ${THEME.primary} !important; box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1) !important; }
        .message-bubble { animation: fadeIn 0.4s ease-out; }
        .quick-btn:hover { background: #F0FDF4 !important; border-color: ${THEME.primary} !important; color: ${THEME.primary} !important; }
        .history-item:hover { background: #F8FAFC !important; }
        .icon-btn:hover { background: #F8FAFC !important; border-color: ${THEME.primary} !important; color: ${THEME.primary} !important; }
        .send-btn:hover:not(:disabled) { background: #059669 !important; transform: scale(1.06); }
        .send-btn:active:not(:disabled) { transform: scale(0.98); }
        button:focus { outline: 2px solid ${THEME.primary}; outline-offset: 2px; }
        input:focus, textarea:focus { outline: 2px solid ${THEME.primary}; outline-offset: 2px; }
      `}</style>

      <div style={styles.root}>
        {/* ── Header ─────────────────────────────────────── */}
        <header style={styles.header}>
          <div style={styles.logo}>
            <svg style={styles.logoIcon} viewBox="0 0 34 34" fill="none">
              <circle cx="17" cy="17" r="16" fill="#FFFFFF" opacity="0.2" />
              <path d="M17 6C12 6 8 11 8 16c0 2.5 1 4.5 3 6l6-6 6 6c2-1.5 3-3.5 3-6 0-5-4-10-9-10z" fill="#FFFFFF"/>
              <path d="M17 28V18M17 18C17 18 14 16 12 13M17 18C17 18 20 16 22 13" stroke="#FFFFFF" strokeWidth="1.5" strokeLinecap="round" opacity="0.8"/>
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
              style={{ ...styles.iconBtn, border: `1px solid rgba(255,255,255,0.3)`, color: "#FFFFFF", background: "rgba(255,255,255,0.1)" }}
              className="icon-btn"
              onClick={() => setShowSettings(true)}
            >
              <SettingsIcon />
            </button>
            {messages.length > 0 && (
              <button
                style={{ ...styles.iconBtn, border: `1px solid rgba(255,255,255,0.3)`, color: "#FFFFFF", background: "rgba(255,255,255,0.1)" }}
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
                    marginTop: 12,
                    width: "100%",
                    padding: "8px",
                    background: "transparent",
                    border: `1.5px dashed ${THEME.primary}40`,
                    borderRadius: 10,
                    fontSize: 12,
                    color: THEME.primary,
                    cursor: "pointer",
                    fontFamily: "'Source Code Pro', monospace",
                    fontWeight: 600,
                    transition: "all 0.2s",
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
                <button style={{ ...styles.iconBtn, width: 24, height: 24, border: "none", background: "transparent" }} className="icon-btn">
                  <RefreshIcon />
                </button>
              </div>
              <div style={styles.weatherCard} className="weather-card">
                <div style={styles.weatherCity}>
                  📍 {profile.weather_city || userLocation.city || "Your Location"}
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
                    background: "#D1FAE520",
                    border: `1px solid ${THEME.success}40`,
                    color: THEME.success,
                    marginTop: 8,
                  }}>
                    ✓ Good conditions for field operations
                  </div>
                )}
              </div>
            </div>
          </aside>

          {/* ── Chat Area ──────────────────────────────────── */}
          <main style={styles.chatArea}>
            <div style={styles.messages}>
              {messages.length === 0 ? (
                /* Welcome Screen */
                <div style={styles.welcome}>
                  <svg style={styles.welcomeIcon} viewBox="0 0 80 80" fill="none">
                    <circle cx="40" cy="40" r="38" fill={THEME.primary} opacity="0.1" stroke={THEME.primary} strokeWidth="1.5"/>
                    <path d="M40 15C28 15 18 26 18 38c0 6 2.5 11 7 14.5L40 38l15 14.5C59.5 49 62 44 62 38c0-12-10-23-22-23z" fill={THEME.primary}/>
                    <path d="M40 68V44M40 44C40 44 32 39 27 31M40 44C40 44 48 39 53 31" stroke={THEME.leaf} strokeWidth="2" strokeLinecap="round"/>
                    <circle cx="40" cy="40" r="4" fill={THEME.primary}/>
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
                ["Template hits", messages.filter(m => normalizeSource(m.source) === "template").length],
                ["RAG hits", messages.filter(m => normalizeSource(m.source) === "retrieval").length],
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
