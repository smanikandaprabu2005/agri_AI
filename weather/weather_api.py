"""
weather/weather_api.py
======================
OpenWeatherMap integration with intelligent mock fallback.
Cached for 30 minutes. Generates agricultural advisories.

FIXES:
  FIX 1: _get() bare `except:` → `except Exception:` 
          (bare except catches KeyboardInterrupt, SystemExit — should never do that)
  FIX 2: fetch_weather() mock returns include "at" key for consistency with live data
  FIX 3: WeatherService.get() cache check: self._cache and self._time both checked
         before arithmetic (was already OK, kept for clarity)
"""

import json
import os
import re
import sys
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_CITY, WEATHER_CACHE_MIN

OWM_URL      = "https://api.openweathermap.org/data/2.5/weather"
OWM_FORE_URL = "https://api.openweathermap.org/data/2.5/forecast"

_WEATHER_KW = re.compile(
    r"\b(weather|rain|spray|water|irrigat|temperature|wind|humid|forecast"
    r"|today|tomorrow|season|monsoon|climate)\b", re.I)


def needs_weather(q: str) -> bool:
    return bool(_WEATHER_KW.search(q))


def _get(url: str, params: dict):
    """
    FIX 1: Changed bare `except:` to `except Exception:`.
    Bare except clauses catch KeyboardInterrupt and SystemExit which should
    never be silently swallowed.
    """
    try:
        full = url + "?" + urllib.parse.urlencode(params)
        req  = urllib.request.Request(full, headers={"User-Agent": "AgriBot/2.0"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception:   # FIX 1: was bare `except:`
        return None


def _mock(city: str) -> dict:
    """
    FIX 2: Added "at" key to mock response for consistency with live data.
    weather_context() doesn't use it, but other consumers might.
    """
    m  = datetime.now().month
    at = datetime.now().strftime("%Y-%m-%d %H:%M")

    if 6 <= m <= 9:
        return {"city": city, "temp": 32.0, "humid": 85,
                "desc": "heavy rain", "rain": True, "rain_pct": 75,
                "wind": 15.0, "src": "mock", "at": at}
    elif m in (10, 11):
        return {"city": city, "temp": 28.0, "humid": 65,
                "desc": "partly cloudy", "rain": False, "rain_pct": 10,
                "wind": 8.0, "src": "mock", "at": at}
    elif m in (12, 1, 2):
        return {"city": city, "temp": 18.0, "humid": 55,
                "desc": "clear sky", "rain": False, "rain_pct": 0,
                "wind": 5.0, "src": "mock", "at": at}
    else:
        return {"city": city, "temp": 35.0, "humid": 70,
                "desc": "thunderstorm possible", "rain": True, "rain_pct": 40,
                "wind": 18.0, "src": "mock", "at": at}


def fetch_weather(city: str = DEFAULT_CITY, api_key: str = "") -> dict:
    if not api_key:
        api_key = os.environ.get("OWM_API_KEY", "")
    if not api_key:
        print("[Weather] No API key — using mock data")
        return _mock(city)

    cur = _get(OWM_URL, {"q": city, "appid": api_key, "units": "metric"})
    if not cur:
        return _mock(city)

    temp   = cur.get("main",    {}).get("temp",     25)
    humid  = cur.get("main",    {}).get("humidity", 60)
    wind   = cur.get("wind",    {}).get("speed",     0) * 3.6
    desc   = cur.get("weather", [{}])[0].get("description", "")
    rain_n = "rain" in desc or "drizzle" in desc

    fore    = _get(OWM_FORE_URL,
                   {"q": city, "appid": api_key, "units": "metric", "cnt": 8})
    rain_c  = 0
    if fore:
        rain_c = sum(
            1 for item in fore.get("list", [])
            if ("rain" in item.get("weather", [{}])[0].get("description", "").lower()
                or item.get("rain", {}))
        )

    return {
        "city":     city,
        "temp":     round(temp, 1),
        "humid":    humid,
        "desc":     desc,
        "rain":     rain_n or rain_c >= 2,
        "rain_pct": min(100, rain_c * 12),
        "wind":     round(wind, 1),
        "src":      "live",
        "at":       datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


def weather_context(w: dict) -> str:
    lines = [
        f"Location: {w['city']}  ({w.get('src', 'mock')} data)",
        f"Temp: {w['temp']}°C  Humidity: {w['humid']}%  Wind: {w['wind']} km/h",
        f"Conditions: {w['desc'].title()}",
    ]
    if w["rain"]:
        lines += [
            f"⚠ Rain expected (prob ~{w['rain_pct']}%)",
            "Recommendation: Avoid pesticide/fertiliser spraying.",
        ]
    else:
        lines += [
            f"No significant rain expected (prob ~{w['rain_pct']}%)",
            "Conditions suitable for field operations.",
        ]
    return "\n".join(lines)


def weather_advisory(w: dict) -> str:
    adv = []
    if w["rain"]:
        adv += ["Do not spray — rain will wash chemicals off.", "Check field drainage."]
    if w["temp"] > 38:
        adv.append("High temp: water crops early morning or evening.")
    if w["temp"] < 12:
        adv.append("Cold alert: protect seedlings if needed.")
    if w["humid"] > 80 and not w["rain"]:
        adv.append("High humidity: monitor for fungal disease.")
    if w["wind"] > 30:
        adv.append("Strong winds: avoid spraying today.")
    return (" | ".join(adv) if adv
            else "Conditions are suitable for normal field operations.")


class WeatherService:
    def __init__(self, city: str = DEFAULT_CITY, api_key: str = ""):
        self.city    = city
        self.api_key = api_key
        self._cache  = None
        self._time   = None

    def get(self, city: str = None) -> dict:
        city = city or self.city
        now  = datetime.now()
        if (self._cache is not None
                and self._cache.get("city") == city
                and self._time is not None
                and (now - self._time).total_seconds() < WEATHER_CACHE_MIN * 60):
            return self._cache
        w = fetch_weather(city, self.api_key)
        self._cache = w
        self._time  = now
        return w

    def context_str(self, city: str = None) -> str:
        return weather_context(self.get(city))

    def advisory(self, city: str = None) -> str:
        return weather_advisory(self.get(city))