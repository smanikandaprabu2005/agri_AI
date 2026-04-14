"""
chatbot/chat_engine.py
======================
Terminal chat loop. Commands: profile | reset | help | quit
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BANNER = """
╔══════════════════════════════════════════════════════════╗
║  🌾  SageStorm V2 Agriculture Advisory Chatbot  🌾      ║
║  Type 'help' for commands                                ║
╚══════════════════════════════════════════════════════════╝
"""

class ChatEngine:
    def __init__(self, generator, memory, verbose=False):
        self.gen     = generator
        self.memory  = memory
        self.verbose = verbose

    def _cmd(self, text):
        t = text.strip().lower()
        if t in ("quit","exit"):     return "__EXIT__"
        if t == "reset":
            self.memory.reset_session()
            return "Session cleared. Your farmer profile is preserved."
        if t == "profile":
            return f"Your Farmer Profile:\n{self.memory.long.as_text()}"
        if t in ("help","?"):
            return "Commands: quit | exit | reset | profile | help\nAsk any agricultural question."
        return None

    def chat(self, query):
        cmd = self._cmd(query)
        if cmd: return cmd
        self.memory.process_input(query)
        answer, source = self.gen.generate(query)
        if self.verbose: print(f"  [source: {source}]")
        self.memory.add_response(answer)
        return answer

    def run(self):
        print(BANNER)
        while True:
            try:
                q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBot: Goodbye! Happy farming! 🌱")
                break
            if not q: continue
            r = self.chat(q)
            if r == "__EXIT__":
                print("Bot: Goodbye! Happy farming! 🌱")
                break
            print(f"Bot: {r}\n")
