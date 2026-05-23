"""
Cache disque pour les données OHLCV Binance.
- Sauvegarde en parquet dans data_cache/
- Recharge depuis le disque si le fichier a moins de 24h
- Évite de re-télécharger 20 symboles × N timeframes à chaque backtest
"""

import time
import pickle
from pathlib import Path

import ccxt
import pandas as pd

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)

MAX_AGE_HOURS = 24 * 7  # Recharge si le fichier a plus de 7 jours

# Nombre de bougies à télécharger pour 4 ans
CANDLES_4Y = {
    "30m": 70080, "1h": 35040, "2h": 17520,
    "4h":   8760, "6h":  5840, "12h": 2920, "1d": 1460,
}

# Nombre de bougies pour 8 ans (période ancienne = années -5 à -8)
CANDLES_8Y = {
    "30m": 140160, "1h": 70080, "2h": 35040,
    "4h":   17520, "6h": 11680, "12h": 5840, "1d": 2920,
}

# Nombre de bougies pour 10 ans (années calendaires 2016-2017)
CANDLES_10Y = {
    "2h": 43800, "4h": 21900, "6h": 14600, "12h": 7300, "1d": 3650,
}

# TFs pour lesquels on télécharge 8 ans (30m/1h trop lents, on skip)
TF_WITH_8Y  = ["2h", "4h", "6h", "12h", "1d"]
TF_WITH_10Y = ["2h", "4h", "6h", "12h", "1d"]


def _cache_path(symbol: str, timeframe: str, suffix: str = "") -> Path:
    safe = symbol.replace("/", "_")
    return CACHE_DIR / f"{safe}_{timeframe}{suffix}.pkl"


def _is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    return age_hours < MAX_AGE_HOURS


def fetch_ohlcv(symbol: str, timeframe: str, verbose: bool = True) -> pd.DataFrame:
    """
    Retourne un DataFrame OHLCV pour (symbol, timeframe) sur 4 ans.
    - Recharge depuis le disque si le cache est frais (< 24h)
    - Sinon télécharge depuis Binance et sauvegarde sur disque
    """
    path = _cache_path(symbol, timeframe)

    if _is_fresh(path):
        if verbose:
            print(f"  [cache] {symbol} {timeframe} — chargé depuis disque")
        with open(path, "rb") as f:
            df = pickle.load(f)
        return df

    # Téléchargement
    exchange = ccxt.binance({"enableRateLimit": True})
    target   = CANDLES_4Y[timeframe]
    tf_ms    = exchange.parse_timeframe(timeframe) * 1000
    since    = exchange.milliseconds() - target * tf_ms
    candles  = []

    if verbose:
        print(f"  [dl]    {symbol} {timeframe} ({target} bougies)...", end=" ", flush=True)

    while len(candles) < target:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception:
            time.sleep(2)
            continue
        if not batch:
            break
        candles.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    if verbose:
        print(f"{len(candles)} bougies reçues")

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

    # Sauvegarde disque
    with open(path, "wb") as f:
        pickle.dump(df, f)

    return df


def fetch_ohlcv_8y(symbol: str, timeframe: str, verbose: bool = True) -> pd.DataFrame:
    """
    Retourne un DataFrame OHLCV pour (symbol, timeframe) sur 8 ans.
    Cache séparé (_8y.pkl) pour ne pas écraser le cache 4 ans.
    """
    if timeframe not in TF_WITH_8Y:
        raise ValueError(f"{timeframe} non supporté pour 8 ans (trop lent)")

    path = _cache_path(symbol, timeframe, suffix="_8y")

    if _is_fresh(path):
        if verbose:
            print(f"  [cache] {symbol} {timeframe} 8y — chargé depuis disque")
        with open(path, "rb") as f:
            return pickle.load(f)

    exchange = ccxt.binance({"enableRateLimit": True})
    target   = CANDLES_8Y[timeframe]
    tf_ms    = exchange.parse_timeframe(timeframe) * 1000
    since    = exchange.milliseconds() - target * tf_ms
    candles  = []

    if verbose:
        print(f"  [dl]    {symbol} {timeframe} 8y ({target} bougies)...", end=" ", flush=True)

    while len(candles) < target:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception:
            time.sleep(2)
            continue
        if not batch:
            break
        candles.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    if verbose:
        print(f"{len(candles)} bougies reçues")

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

    with open(path, "wb") as f:
        pickle.dump(df, f)

    return df


def prefetch_all_8y(symbols: list[str], timeframes: list[str], verbose: bool = True) -> None:
    """Pré-télécharge 8 ans pour les TFs supportés."""
    supported = [tf for tf in timeframes if tf in TF_WITH_8Y]
    total = len(symbols) * len(supported)
    done  = 0
    for symbol in symbols:
        for tf in supported:
            path = _cache_path(symbol, tf, suffix="_8y")
            if _is_fresh(path):
                if verbose:
                    print(f"  [cache] {symbol} {tf} 8y — déjà en cache", end="\r")
            else:
                try:
                    fetch_ohlcv_8y(symbol, tf, verbose=verbose)
                except Exception as e:
                    if verbose:
                        print(f"  [err]   {symbol} {tf} 8y: {e}")
            done += 1
    if verbose:
        print(f"\n  {done}/{total} paires traitées (8 ans)")


def fetch_ohlcv_10y(symbol: str, timeframe: str, verbose: bool = True) -> pd.DataFrame:
    """
    Retourne un DataFrame OHLCV pour (symbol, timeframe) sur 10 ans.
    Cache séparé (_10y.pkl) pour les années calendaires 2016-2017.
    """
    if timeframe not in TF_WITH_10Y:
        raise ValueError(f"{timeframe} non supporté pour 10 ans")

    path = _cache_path(symbol, timeframe, suffix="_10y")

    if _is_fresh(path):
        if verbose:
            print(f"  [cache] {symbol} {timeframe} 10y — chargé depuis disque")
        with open(path, "rb") as f:
            return pickle.load(f)

    exchange = ccxt.binance({"enableRateLimit": True})
    target   = CANDLES_10Y[timeframe]
    tf_ms    = exchange.parse_timeframe(timeframe) * 1000
    since    = exchange.milliseconds() - target * tf_ms
    candles  = []

    if verbose:
        print(f"  [dl]    {symbol} {timeframe} 10y ({target} bougies)...", end=" ", flush=True)

    while len(candles) < target:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception:
            time.sleep(2)
            continue
        if not batch:
            break
        candles.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    if verbose:
        print(f"{len(candles)} bougies reçues")

    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()

    with open(path, "wb") as f:
        pickle.dump(df, f)

    return df


def prefetch_all_10y(symbols: list[str], timeframes: list[str], verbose: bool = True) -> None:
    """Pré-télécharge 10 ans pour les TFs supportés."""
    supported = [tf for tf in timeframes if tf in TF_WITH_10Y]
    total = len(symbols) * len(supported)
    done  = 0
    for symbol in symbols:
        for tf in supported:
            path = _cache_path(symbol, tf, suffix="_10y")
            if _is_fresh(path):
                if verbose:
                    print(f"  [cache] {symbol} {tf} 10y — déjà en cache", end="\r")
            else:
                try:
                    fetch_ohlcv_10y(symbol, tf, verbose=verbose)
                except Exception as e:
                    if verbose:
                        print(f"  [err]   {symbol} {tf} 10y: {e}")
            done += 1
    if verbose:
        print(f"\n  {done}/{total} paires traitées (10 ans)")


def prefetch_all(symbols: list[str], timeframes: list[str], verbose: bool = True) -> None:
    """Pré-télécharge toutes les combinaisons symbol × timeframe."""
    total   = len(symbols) * len(timeframes)
    done    = 0
    skipped = 0
    for symbol in symbols:
        for tf in timeframes:
            path = _cache_path(symbol, tf)
            if _is_fresh(path):
                skipped += 1
                done += 1
                if verbose:
                    print(f"  [cache] {symbol} {tf} — déjà en cache", end="\r")
                continue
            try:
                fetch_ohlcv(symbol, tf, verbose=verbose)
            except Exception as e:
                if verbose:
                    print(f"  [err]   {symbol} {tf}: {e}")
            done += 1
    if verbose and skipped:
        print(f"\n  {skipped}/{total} symboles chargés depuis le cache disque")
