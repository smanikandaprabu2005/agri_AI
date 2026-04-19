"""
ui/app.py
=========
SageStorm V2 — Gradio Web UI

Features:
  • Chat interface with message history
  • Farmer profile sidebar
  • Weather panel (live or mock)
  • Source badge on every response (template / retrieval / sagestorm / fallback)
  • Profile display updates in real time
  • Session reset button

Usage:
    pip install gradio
    python ui/app.py
    python ui/app.py --share          (public Gradio link)
    python ui/app.py --port 8080
"""

import os, sys, argparse, textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from config import (
    FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY, DEVICE,
    GEN_MAX_TOKENS, GEN_TEMPERATURE, UI_HOST, UI_PORT,
)


# ══════════════════════════════════════════════════════════════
#  System loader
# ══════════════════════════════════════════════════════════════
_gen    = None
_memory = None
_loaded = False

def load_system(weights_path, api_key="", city=DEFAULT_CITY):
    global _gen, _memory, _loaded

    from retrieval.vector_search import load_retriever
    from memory.memory_manager   import MemoryManager
    from weather.weather_api     import WeatherService
    from rag.context_builder     import ContextBuilder
    from chatbot.response_generator import ResponseGenerator
    from models.tokenizer        import SageTokenizer

    print("[UI] Loading system...")

    ret, vocab = load_retriever(RETRIEVER_DIR)
    memory     = MemoryManager()
    weather    = WeatherService(city=city, api_key=api_key)
    ctx        = ContextBuilder(ret, memory, weather, top_k=5)
    tok        = SageTokenizer()

    model = None
    try:
        import torch
        if os.path.exists(weights_path):
            from models.sagestorm_v2 import SageStormV2
            model = SageStormV2.load(weights_path, DEVICE)
            print(f"[UI] SageStorm V2 loaded from {weights_path}")
        else:
            print(f"[UI] Weights not found at {weights_path} — retrieval-only mode")
    except ImportError:
        print("[UI] PyTorch not available — retrieval-only mode")

    gen  = ResponseGenerator(model, tok, ret, ctx,
                              max_tokens=GEN_MAX_TOKENS,
                              temperature=GEN_TEMPERATURE)
    _gen    = gen
    _memory = memory
    _loaded = True
    print("[UI] System ready.")
    return gen, memory


# ══════════════════════════════════════════════════════════════
#  Source badge styling
# ══════════════════════════════════════════════════════════════
SOURCE_COLORS = {
    "template"  : ("#d1fae5", "#065f46", "✦ Template"),
    "retrieval" : ("#dbeafe", "#1e40af", "⟳ Retrieval"),
    "sagestorm" : ("#ede9fe", "#4c1d95", "✧ SageStorm V2"),
    "fallback"  : ("#fef3c7", "#92400e", "⚠ Fallback"),
}

def badge_html(source):
    bg, fg, label = SOURCE_COLORS.get(source, ("#f3f4f6","#374151","? Unknown"))
    return (f'<span style="background:{bg};color:{fg};border-radius:12px;'
            f'padding:2px 10px;font-size:11px;font-weight:600;">{label}</span>')


# ══════════════════════════════════════════════════════════════
#  Chat function
# ══════════════════════════════════════════════════════════════
def chat_fn(message, history):
    if not _loaded:
        return history + [[message, "⚠ System not loaded. Please restart."]], "", "No profile yet."

    _memory.process_input(message)
    answer, source = _gen.generate(message)
    _memory.add_response(answer)

    badge    = badge_html(source)
    response = f"{answer}\n\n{badge}"
    history  = history + [[message, response]]
    profile  = _memory.long.as_text()
    return history, "", profile


def reset_fn():
    if _memory: _memory.reset_session()
    return [], "", "No profile yet."


def get_weather_fn(city):
    if not _loaded: return "System not loaded."
    try:
        w = _gen.ctx_builder.weather_svc.get(city or DEFAULT_CITY)
        from weather.weather_api import weather_context
        return weather_context(w)
    except Exception as e:
        return f"Error: {e}"


# ══════════════════════════════════════════════════════════════
#  Gradio Layout
# ══════════════════════════════════════════════════════════════
CSS = """
#chatbot { height: 520px; overflow-y: auto; }
.source-badge { font-size: 11px; }
#title { text-align: center; padding: 8px 0 0; }
#profile-box { font-size: 13px; font-family: monospace; }
footer { display: none !important; }
"""

EXAMPLES = [
    "I am growing rice near Guwahati, my soil is loamy.",
    "How do I control stem borers in my rice crop?",
    "Should I spray pesticide today?",
    "What is the fertilizer dose for rice per acre?",
    "There are aphids on my lemon tree. What should I do?",
    "How to prevent late blight in tomatoes?",
    "What is the planting spacing for tomatoes?",
    "How to control whitefly on pepper?",
]

def build_ui():
    with gr.Blocks(css=CSS, title="SageStorm V2 Agriculture Chatbot") as demo:

        gr.Markdown(
            "# 🌾 SageStorm V2 Agriculture Advisory Chatbot\n"
            "Powered by a 48M-parameter domain-specific GPT + RAG retrieval + live weather",
            elem_id="title"
        )

        with gr.Row():

            # ── Left: Chat ────────────────────────────────────
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Chat",
                    elem_id="chatbot",
                    bubble_full_width=False,
                    show_label=False,
                )
                with gr.Row():
                    msg_box = gr.Textbox(
                        placeholder="Ask about crops, pests, fertilizers, weather...",
                        show_label=False,
                        scale=5,
                        container=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                gr.Examples(
                    examples=EXAMPLES,
                    inputs=msg_box,
                    label="Example questions",
                )
                reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

            # ── Right: Info panels ────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Farmer Profile")
                profile_box = gr.Textbox(
                    value="No profile yet.",
                    label="",
                    interactive=False,
                    lines=6,
                    elem_id="profile-box",
                )

                gr.Markdown("### 🌤 Weather")
                weather_city = gr.Textbox(
                    value=DEFAULT_CITY,
                    label="City",
                    placeholder="Enter city name",
                )
                weather_btn = gr.Button("Get Weather", size="sm")
                weather_out = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=7,
                    elem_id="profile-box",
                )

                gr.Markdown("### ℹ Response Sources")
                gr.HTML(
                    f'<div style="font-size:12px;line-height:2">'
                    f'{badge_html("template")} Rule-based, highest accuracy<br>'
                    f'{badge_html("retrieval")} RAG document passage<br>'
                    f'{badge_html("sagestorm")} SageStorm V2 generation<br>'
                    f'{badge_html("fallback")} Default safe response'
                    f'</div>'
                )

        # ── Event bindings ────────────────────────────────────
        def submit(msg, hist):
            return chat_fn(msg, hist)

        send_btn.click(submit, [msg_box, chatbot], [chatbot, msg_box, profile_box])
        msg_box.submit(submit, [msg_box, chatbot], [chatbot, msg_box, profile_box])
        reset_btn.click(reset_fn, outputs=[chatbot, msg_box, profile_box])
        weather_btn.click(get_weather_fn, [weather_city], [weather_out])

    return demo


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",  default=FINETUNED_CKPT)
    parser.add_argument("--api_key",  default="")
    parser.add_argument("--city",     default=DEFAULT_CITY)
    parser.add_argument("--port",     type=int, default=UI_PORT)
    parser.add_argument("--share",    action="store_true")
    parser.add_argument("--no_load",  action="store_true",
                        help="Skip model loading (UI preview only)")
    args = parser.parse_args()

    if not args.no_load:
        load_system(args.weights, api_key=args.api_key, city=args.city)

    demo = build_ui()
    demo.launch(
        server_name = UI_HOST,
        server_port = args.port,
        share       = args.share,
        inbrowser   = True,
    )


if __name__ == "__main__":
    main()
