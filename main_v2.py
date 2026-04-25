"""
main.py  (V2 Upgraded)
======================
SageStorm V2 Agriculture Chatbot — Terminal Entry Point

Changes from original:
  • Loads VectorDB (FAISS + BM25) if available
  • Falls back to Word2Vec if VectorDB not built yet
  • Uses ContextBuilderV2 with larger context window

Usage:
    python main.py                           (interactive chat)
    python main.py --demo                    (scripted demo)
    python main.py --weights path/to/ckpt.pt
    python main.py --city Mumbai --api_key YOUR_OWM_KEY
    python main.py --verbose                 (show answer source)
    python main.py --no_vector_db            (use Word2Vec only)
"""

import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FINETUNED_CKPT, RETRIEVER_DIR, DEFAULT_CITY, DEVICE,
    GEN_TEMPERATURE, GEN_MAX_TOKENS
)
from dotenv import load_dotenv
import os

load_dotenv()
VECTOR_DB_DIR = os.path.join("saved_models", "vector_index")

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


def load_system(weights_path, api_key, city, temperature, max_tokens, use_vector_db=True):
    from retrieval.vector_search    import load_retriever
    from memory.memory_manager      import MemoryManager
    from weather.weather_api        import WeatherService
    from chatbot.response_generator import ResponseGenerator
    from models.tokenizer           import SageTokenizer

    print("[Main] Loading RAG components...")
    ret, vocab = load_retriever(RETRIEVER_DIR)
    memory     = MemoryManager()
    weather    = WeatherService(city=city, api_key=api_key)
    tok        = SageTokenizer()

    # Try to load VectorDB
    vector_db = None
    if use_vector_db:
        vdb_config = os.path.join(VECTOR_DB_DIR, "config.json")
        if os.path.exists(vdb_config):
            try:
                from retrieval.vector_db import VectorDB
                vector_db = VectorDB.load(VECTOR_DB_DIR)
                stats = vector_db.stats()
                print(f"[Main] VectorDB loaded: {stats['total_docs']:,} chunks")
                print(f"       Sources: {list(stats['sources'].keys())[:5]}")
            except Exception as e:
                print(f"[Main] VectorDB load failed: {e} — using Word2Vec only")
        else:
            print(f"[Main] No VectorDB found at {VECTOR_DB_DIR}")
            print(f"       Run: python data_pipeline/step6_ingest_books.py")
            print(f"       (continuing with Word2Vec retriever)")

    # Build context builder (V2 if VectorDB available, V1 otherwise)
    if vector_db is not None:
        from rag.context_builder_v2 import ContextBuilderV2
        ctx = ContextBuilderV2(ret, memory, weather, vector_db=vector_db, top_k=3)
        print("[Main] Using ContextBuilderV2 (FAISS + BM25 + Word2Vec)")
    else:
        from rag.context_builder import ContextBuilder
        ctx = ContextBuilder(ret, memory, weather, top_k=3)
        print("[Main] Using ContextBuilder (Word2Vec only)")

    # Load SageStorm model
    model   = None
    backend = "retrieval-only"
    try:
        import torch
        if os.path.exists(weights_path):
            from models.sagestorm_v2 import SageStormV2
            model   = SageStormV2.load(weights_path, DEVICE)
            backend = f"SageStorm V2 ({model.param_count()['total_M']}M params)"
        else:
            print(f"  [!] Weights not found: {weights_path}")
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
    parser.add_argument("--weights",       default=FINETUNED_CKPT)
    parser.add_argument("--city",          default=DEFAULT_CITY)
    parser.add_argument("--api_key",       default="")
    parser.add_argument("--demo",          action="store_true")
    parser.add_argument("--verbose",       action="store_true")
    parser.add_argument("--no_vector_db",  action="store_true",
                        help="Skip VectorDB, use Word2Vec only")
    parser.add_argument("--temperature",   type=float, default=GEN_TEMPERATURE)
    parser.add_argument("--max_tokens",    type=int,   default=GEN_MAX_TOKENS)
    args = parser.parse_args()

    gen, memory = load_system(
        args.weights, args.api_key, args.city,
        args.temperature, args.max_tokens,
        use_vector_db=not args.no_vector_db,
    )

    if args.demo:
        run_demo(gen, memory, verbose=args.verbose)
    else:
        from chatbot.chat_engine import ChatEngine
        ChatEngine(gen, memory, verbose=args.verbose).run()


if __name__ == "__main__":
    main()

"""
1. Crop Recommendation (should NOT give IPM 😄)

Use these to test your routing:

Which crop is best for loamy soil in Tamil Nadu?
What crops give high yield in Sivakasi region?
Suggest suitable crops for dryland farming
Which crop is profitable in summer season?
Best crop for sandy soil with low rainfall
🌱 2. Fertilizer Questions (should hit TEMPLATE)
What is the fertilizer dose for rice?
How much urea should I apply for paddy?
Fertilizer schedule for tomato crop
NPK requirement for groundnut
When should I apply fertilizers in rice?
🐛 3. Pest & Disease (important — your weak area)
How to control stem borers in rice?
What pesticide should I use for aphids in cotton?
How to control leaf blight in rice?
Symptoms of pest attack in tomato
Organic methods to control pests in vegetables
🌧️ 4. Weather-based Questions
Should I spray pesticide today?
Is it safe to irrigate my field today?
Will rain affect fertilizer application?
Best time to spray pesticides?
Can I sow seeds today based on weather?
🌾 5. Spacing & Cultivation (your system failed here)
What is spacing for rice?
Plant spacing for tomato crop
How far apart should I plant cotton?
Seed rate for paddy per acre
Row spacing for maize
🧠 6. Context / Memory-Based Questions (VERY IMPORTANT)

Ask in sequence:

I am growing rice in Sivakasi on loamy soil
What fertilizer should I use?
How to control pests in my crop?
What spacing should I follow?

👉 Your system should remember rice, not answer generically.
"""