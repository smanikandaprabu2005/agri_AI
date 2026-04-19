# Strom Sage AI — React UI

Complete React + FastAPI UI for the SageStorm V2 agriculture chatbot.

## File Structure

```
project-root/
├── api/
│   └── server.py          ← FastAPI backend (copy from outputs)
├── ui/
│   ├── src/
│   │   ├── main.jsx       ← React entry point
│   │   └── StromSageUI.jsx ← Main UI component
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
```

## Quick Start

### 1. Start the FastAPI backend

```bash
# Install FastAPI
pip install fastapi uvicorn

# From project root
python api/server.py
# → Running at http://localhost:8000
```

### 2. Start the React frontend

```bash
cd ui/
npm install
npm run dev
# → Running at http://localhost:5173
```

### 3. Open your browser

Navigate to `http://localhost:5173`

---

## API Endpoints

| Method | Path       | Description                     |
|--------|------------|---------------------------------|
| GET    | /health    | Health check + backend info     |
| POST   | /chat      | Main chat (query → answer)      |
| GET    | /weather   | Weather data for a city         |
| GET    | /profile   | Get current farmer profile      |
| POST   | /profile   | Update farmer profile           |
| POST   | /reset     | Reset conversation session      |
| GET    | /stats     | Session statistics              |

### Chat Request

```json
POST /chat
{
  "query": "How do I control stem borers in rice?",
  "city":  "Guwahati",
  "verbose": false
}
```

### Chat Response

```json
{
  "answer":      "To control stem borers...",
  "source":      "template",
  "latency_ms":  142.5,
  "timestamp":   "2024-01-15T10:30:00"
}
```

`source` values:
- `template`   — Rule-based (fastest, highest confidence)
- `retrieval`  — RAG document passage
- `sagestorm`  — SageStorm V2 generation
- `fallback`   — Safe default response

---

## Features

- **Earthy botanical design** — distinctive, non-generic aesthetic
- **Farmer profile panel** — auto-extracts crop/location/soil from conversation
- **Live weather panel** — shows spray advisory based on conditions
- **Session history** — quick re-ask from right sidebar
- **Source badges** — shows exactly how each answer was generated
- **Stats panel** — template vs RAG vs SLM hit counts
- **Settings modal** — configure profile, API URL, weather city
- **Offline fallback** — simulated responses when backend is unavailable
- **Auto-profile extraction** — mentions like "growing rice near Guwahati" update the profile automatically

---

## Production Build

```bash
cd ui/
npm run build
# Outputs to ui/dist/

# Serve static files from FastAPI (optional)
# Add to api/server.py:
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="ui/dist", html=True), name="static")
```

## Environment Variables

```bash
# Optional: real OpenWeatherMap API key
export OWM_API_KEY="your_owm_key_here"

# Then start the server
python api/server.py
```
