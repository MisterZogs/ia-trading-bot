"""
Fear & Greed Index — données historiques depuis alternative.me.
Cache disque 7 jours (même convention que data_cache.py).
Fournit un dict {date: valeur} pour les backtests.

Utilisation :
    fg = load()          # dict {datetime.date -> int (0-100)}
    val = fg.get(date)   # None si pas disponible (avant nov 2020)

Règles appliquées dans les backtests :
    val > FG_GREED_VETO  → bloque le BUY (marché en euphorie)
    val < FG_FEAR_BONUS  → confirme le BUY (opportunité de peur extrême)
"""

import json
import pickle
import time
import urllib.request
from datetime import datetime
from pathlib import Path

CACHE_PATH  = Path(__file__).parent / "data_cache" / "fear_greed.pkl"
MAX_AGE_HOURS = 24 * 7          # recharge tous les 7 jours
API_URL     = "https://api.alternative.me/fng/?limit=2000&format=json"

# Seuils — modifiables
FG_GREED_VETO = 85   # au-dessus : bloque BUY
FG_FEAR_BONUS = 20   # en-dessous : score BUY +1 (bonus)


def _is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    return (time.time() - CACHE_PATH.stat().st_mtime) / 3600 < MAX_AGE_HOURS


def _download() -> dict:
    """Télécharge le F&G historique et retourne {date -> int}."""
    with urllib.request.urlopen(API_URL, timeout=15) as r:
        data = json.loads(r.read())
    result = {}
    for entry in data["data"]:
        dt = datetime.fromtimestamp(int(entry["timestamp"])).date()
        result[dt] = int(entry["value"])
    return result


def load(verbose: bool = True) -> dict:
    """
    Retourne le F&G historique en dict {datetime.date -> int}.
    Cache disque 7 jours.
    """
    if _is_fresh():
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)

    if verbose:
        print("  [F&G]   Téléchargement Fear & Greed historique...", end=" ", flush=True)
    fg = _download()
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(fg, f)
    if verbose:
        print(f"{len(fg)} jours")
    return fg
