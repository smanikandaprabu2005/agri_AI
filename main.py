"""
main.py
=======
SageStorm V2 Agriculture Chatbot — Terminal Entry Point

Auto-selects SageStorm V2 if weights are present,
falls back to retrieval-only mode if torch is unavailable.

Usage:
    python main.py                           (interactive chat)
    python main.py --demo                    (scripted demo)
    python main.py --weights path/to/ckpt.pt
    python main.py --city Mumbai --api_key YOUR_OWM_KEY
    python main.py --verbose                 (show answer source)
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY, DEVICE, GEN_TEMPERATURE, GEN_MAX_TOKENS


DEMO_QUERIES = [
    "I am growing rice near Guwahati. My soil is loamy and I have 3 acres.",
    "How do I control stem borers in my rice crop?",
    "Should I spray pesticide today?",
    "What is the recommended fertilizer dose for rice?",
    "There are aphids on my lemon tree. What should I do?",
    "How to prevent late blight in tomatoes?",
    "What is the planting spacing for tomatoes?",
    "profile",
]


def load_system(weights_path, api_key, city, temperature, max_tokens):
    from retrieval.vector_search    import load_retriever
    from memory.memory_manager      import MemoryManager
    from weather.weather_api        import WeatherService
    from rag.context_builder        import ContextBuilder
    from chatbot.response_generator import ResponseGenerator
    from models.tokenizer           import SageTokenizer

    print("[Main] Loading RAG components...")
    ret, vocab = load_retriever(RETRIEVER_DIR)
    memory     = MemoryManager()
    weather    = WeatherService(city=city, api_key=api_key)
    ctx        = ContextBuilder(ret, memory, weather, top_k=5)
    tok        = SageTokenizer()

    model = None
    backend = "retrieval-only"
    try:
        import torch
        if os.path.exists(weights_path):
            from models.sagestorm_v2 import SageStormV2
            model   = SageStormV2.load(weights_path, DEVICE)
            backend = f"SageStorm V2 ({model.param_count()['total_M']}M params)"
        else:
            print(f"  [!] Weights not found: {weights_path}")
            print(f"      Download sage_slm_v2_final.pt and place it in saved_models/")
    except ImportError:
        print("  [!] PyTorch not installed — retrieval-only mode")

    gen = ResponseGenerator(model, tok, ret, ctx, max_tokens=max_tokens, temperature=temperature)
    print(f"[Main] Ready — backend: {backend}\n")
    return gen, memory


def run_demo(engine, memory, verbose=False):
    from chatbot.chat_engine import ChatEngine
    chat = ChatEngine(engine, memory, verbose=verbose)
    print("\n" + "="*55)
    print("  DEMO — SageStorm V2 + RAG Agriculture Chatbot")
    print("="*55)
    for q in DEMO_QUERIES:
        print(f"\nUser: {q}")
        r = chat.chat(q)
        if r == "__EXIT__": break
        print(f"Bot:  {r}")
    print("\n" + "="*55)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",     default=FINETUNED_CKPT)
    parser.add_argument("--city",        default=DEFAULT_CITY)
    parser.add_argument("--api_key",     default="")
    parser.add_argument("--demo",        action="store_true")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--temperature", type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--max_tokens",  type=int,   default=GEN_MAX_TOKENS)
    args = parser.parse_args()

    gen, memory = load_system(
        args.weights, args.api_key, args.city,
        args.temperature, args.max_tokens,
    )

    if args.demo:
        run_demo(gen, memory, verbose=args.verbose)
    else:
        from chatbot.chat_engine import ChatEngine
        ChatEngine(gen, memory, verbose=args.verbose).run()


if __name__ == "__main__":
    main()
