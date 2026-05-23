"""
Backtest TRIX strategy vs stratégie actuelle.
Les deux avec frais de 0.1% à l'achat ET à la vente.

Stratégie TRIX :
  - TRIX(9) + signal SMA(21) → histogram
  - EMA200 trend filter (BUY bloqué si prix < EMA200)
  - StochRSI(14) confirmation (< 0.8 pour BUY, > 0.2 pour SELL)
  - BUY  : histogram > 0 ET StochRSI < 0.8
  - SELL : histogram < 0 ET StochRSI > 0.2
"""

import time
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init

import config

init(autoreset=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEE_RATE   = 0.001          # 0.1% par côté (achat ET vente)
TIMEFRAMES = ["1h", "4h", "6h", "12h"]
PERIODS    = {1: "1 an", 2: "2 ans", 3: "3 ans", 4: "4 ans"}

CANDLES_4Y = {
    "30m": 70080, "1h": 35040, "2h": 17520,
    "4h":   8760, "6h":  5840, "12h": 2920, "1d": 1460,
}
CANDLES_PER_YEAR = {
    "30m": 17520, "1h": 8760, "2h": 4380,
    "4h":   2190, "6h": 1460, "12h":  730, "1d":  365,
}

_raw_cache: dict = {}


# ---------------------------------------------------------------------------
# Téléchargement données
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    key = f"{symbol}_{timeframe}"
    if key in _raw_cache:
        return _raw_cache[key]

    exchange = ccxt.binance({"enableRateLimit": True})
    target   = CANDLES_4Y[timeframe]
    tf_ms    = exchange.parse_timeframe(timeframe) * 1000
    since    = exchange.milliseconds() - target * tf_ms
    candles  = []

    print(f"  Téléchargement {symbol} {timeframe}...", end=" ", flush=True)
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

    print(f"{len(candles)} bougies")
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    _raw_cache[key] = df
    return df


# ---------------------------------------------------------------------------
# Calcul indicateurs TRIX
# ---------------------------------------------------------------------------
def compute_trix_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # TRIX(9) — triple EMA lissée, en % de variation
    trix_df = ta.trix(df["close"], length=9)
    # pandas_ta renvoie TRIX_9_9 et TRIXs_9_9
    trix_col  = [c for c in trix_df.columns if c.startswith("TRIX_")][0]
    df["trix"] = trix_df[trix_col]

    # Signal = SMA(21) du TRIX (comme dans le repo CryptoRobotFr)
    df["trix_signal"] = ta.sma(df["trix"], length=21)
    df["trix_histo"]  = df["trix"] - df["trix_signal"]

    # EMA200 — filtre de tendance (remplace SMA200 pour coller à la stratégie TRIX)
    df["ema200"] = ta.ema(df["close"], length=200)

    # StochRSI(14) — valeurs entre 0 et 1
    stochrsi = ta.stochrsi(df["close"], length=14, rsi_length=14, k=3, d=3)
    k_col = [c for c in stochrsi.columns if "k" in c.lower()][0]
    df["stochrsi_k"] = stochrsi[k_col] / 100  # normalise 0-1 si en 0-100

    df = df.dropna().reset_index()
    return df


# ---------------------------------------------------------------------------
# Calcul indicateurs stratégie ACTUELLE (avec SMA200 + score)
# ---------------------------------------------------------------------------
def compute_current_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Réplique indicators.compute_all() sans dépendre du module."""
    import indicators
    df = indicators.compute_all(df)
    df = df.dropna().reset_index()
    return df


# ---------------------------------------------------------------------------
# Simulation générique avec frais
# ---------------------------------------------------------------------------
def _simulate_core(df: pd.DataFrame, symbol: str, signal_fn,
                   years: int, timeframe: str) -> dict:
    """
    signal_fn(row_index, df) -> "BUY" | "SELL" | "HOLD"
    Frais : FEE_RATE à l'achat (sur la valeur de la position)
            FEE_RATE à la vente (sur la valeur de sortie)
    """
    n = CANDLES_PER_YEAR[timeframe] * years
    df = df.tail(n).reset_index(drop=True)
    if len(df) < 50:
        return {}

    risk = config.SYMBOL_RISK.get(symbol, {
        "sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT
    })
    sl_pct = risk["sl"]
    tp_pct = risk["tp"]

    capital  = 1000.0
    position = None
    trades   = []
    equity   = [capital]

    for i in range(2, len(df)):
        row   = df.iloc[i]
        low   = row["low"]
        high  = row["high"]
        close = row["close"]

        # Vérification SL/TP en cours de bougie (simulation réaliste high/low)
        if position:
            sl_hit = low  <= position["sl"]
            tp_hit = high >= position["tp"]
            if sl_hit or tp_hit:
                exit_price = position["sl"] if sl_hit else position["tp"]
                fee_exit   = exit_price * position["size"] * FEE_RATE
                pnl        = (exit_price - position["entry"]) * position["size"] - fee_exit
                capital   += pnl
                trades.append(pnl)
                position   = None
                equity.append(capital)
                continue

        signal = signal_fn(i, df)

        if signal == "BUY" and position is None:
            size     = (capital * config.POSITION_SIZE_PCT) / close
            fee_buy  = capital * config.POSITION_SIZE_PCT * FEE_RATE
            capital -= fee_buy  # frais d'entrée déduits du capital
            position = {
                "entry": close,
                "size":  size,
                "sl":    close * (1 - sl_pct),
                "tp":    close * (1 + tp_pct),
            }

        elif signal == "SELL" and position is not None:
            fee_exit = close * position["size"] * FEE_RATE
            pnl      = (close - position["entry"]) * position["size"] - fee_exit
            capital += pnl
            trades.append(pnl)
            position = None

        equity.append(capital)

    # Clôture forcée en fin de période
    if position:
        last     = df.iloc[-1]["close"]
        fee_exit = last * position["size"] * FEE_RATE
        pnl      = (last - position["entry"]) * position["size"] - fee_exit
        capital += pnl
        trades.append(pnl)

    if not trades:
        return {"trades": 0, "return_%": 0.0, "win_%": 0.0,
                "profit_factor": 0.0, "drawdown_%": 0.0}

    wins   = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")
    ret    = (capital - 1000.0) / 1000.0 * 100

    peak   = equity[0]
    max_dd = 0.0
    for v in equity:
        peak   = max(peak, v)
        dd     = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "trades":        len(trades),
        "win_%":         round(len(wins) / len(trades) * 100, 1),
        "return_%":      round(ret, 2),
        "profit_factor": round(pf, 2),
        "drawdown_%":    round(max_dd, 2),
    }


# ---------------------------------------------------------------------------
# Signal TRIX
# ---------------------------------------------------------------------------
def make_trix_signal_fn(df: pd.DataFrame):
    def signal_fn(i: int, _df: pd.DataFrame) -> str:
        row      = _df.iloc[i]
        histo    = row["trix_histo"]
        stochrsi = row["stochrsi_k"]
        close    = row["close"]
        ema200   = row["ema200"]

        in_uptrend = pd.isna(ema200) or close > ema200

        if pd.isna(histo) or pd.isna(stochrsi):
            return "HOLD"

        if in_uptrend and histo > 0 and stochrsi < 0.8:
            return "BUY"
        elif histo < 0 and stochrsi > 0.2:
            return "SELL"
        return "HOLD"
    return signal_fn


# ---------------------------------------------------------------------------
# Signal stratégie ACTUELLE
# ---------------------------------------------------------------------------
def make_current_signal_fn():
    import indicators
    def signal_fn(i: int, df: pd.DataFrame) -> str:
        window = df.iloc[:i + 1].set_index("timestamp")
        result = indicators.score_signal(window)
        return result["signal"]
    return signal_fn


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*70}")
    print("  BACKTEST TRIX vs STRATÉGIE ACTUELLE — avec frais 0.1% achat+vente")
    print(f"{'='*70}{Style.RESET_ALL}")

    # Téléchargement des données
    print(f"\n{Fore.CYAN}Téléchargement des données (4 ans)...{Style.RESET_ALL}")
    for symbol in config.SYMBOLS:
        for tf in TIMEFRAMES:
            try:
                fetch_ohlcv(symbol, tf)
            except Exception as e:
                print(f"  {Fore.RED}Erreur {symbol}/{tf}: {e}{Style.RESET_ALL}")

    # Calcul des indicateurs pour chaque (symbol, tf)
    print(f"\n{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    trix_dfs    = {}
    current_dfs = {}
    for symbol in config.SYMBOLS:
        for tf in TIMEFRAMES:
            raw = _raw_cache.get(f"{symbol}_{tf}")
            if raw is None:
                continue
            try:
                trix_dfs[(symbol, tf)]    = compute_trix_indicators(raw.reset_index())
                current_dfs[(symbol, tf)] = compute_current_indicators(raw.reset_index())
            except Exception as e:
                print(f"  {Fore.RED}Erreur indicateurs {symbol}/{tf}: {e}{Style.RESET_ALL}")

    # Simulations
    print(f"\n{Fore.CYAN}Simulations en cours...{Style.RESET_ALL}")
    trix_results    = {}
    current_results = {}
    current_signal_fn = make_current_signal_fn()

    total = len(config.SYMBOLS) * len(TIMEFRAMES) * len(PERIODS)
    done  = 0

    for symbol in config.SYMBOLS:
        for tf in TIMEFRAMES:
            trix_df    = trix_dfs.get((symbol, tf))
            current_df = current_dfs.get((symbol, tf))
            if trix_df is None:
                done += len(PERIODS)
                continue

            trix_signal_fn = make_trix_signal_fn(trix_df)

            for years in PERIODS:
                if trix_df is not None:
                    trix_results[(symbol, tf, years)] = _simulate_core(
                        trix_df, symbol, trix_signal_fn, years, tf
                    )
                if current_df is not None:
                    current_results[(symbol, tf, years)] = _simulate_core(
                        current_df, symbol, current_signal_fn, years, tf
                    )
                done += 1
            print(f"\r  Progression : {done}/{total}", end="", flush=True)

    print()

    # ---------------------------------------------------------------------------
    # Affichage : tableau comparatif par période et timeframe
    # ---------------------------------------------------------------------------
    for years in PERIODS:
        label = PERIODS[years]
        print(f"\n{Fore.YELLOW}{'='*90}")
        print(f"  PÉRIODE : {label.upper()} — Frais inclus (0.1% achat + 0.1% vente)")
        print(f"{'='*90}{Style.RESET_ALL}")

        headers = ["Symbole", "TF",
                   "TRIX return", "TRIX trades", "TRIX win%", "TRIX DD%",
                   "ACTUEL return", "ACTUEL trades", "ACTUEL win%", "ACTUEL DD%",
                   "Delta"]
        rows = []

        for symbol in config.SYMBOLS:
            # Meilleure TF TRIX pour ce symbole/période
            best_trix_tf  = None
            best_trix_ret = None
            for tf in TIMEFRAMES:
                r = trix_results.get((symbol, tf, years), {})
                ret = r.get("return_%")
                if ret is not None and (best_trix_ret is None or ret > best_trix_ret):
                    best_trix_ret = ret
                    best_trix_tf  = tf

            # Meilleure TF actuelle pour ce symbole/période
            best_cur_tf  = None
            best_cur_ret = None
            for tf in TIMEFRAMES:
                r = current_results.get((symbol, tf, years), {})
                ret = r.get("return_%")
                if ret is not None and (best_cur_ret is None or ret > best_cur_ret):
                    best_cur_ret = ret
                    best_cur_tf  = tf

            tr = trix_results.get((symbol, best_trix_tf, years), {}) if best_trix_tf else {}
            cr = current_results.get((symbol, best_cur_tf, years), {}) if best_cur_tf else {}

            trix_ret = tr.get("return_%", 0)
            cur_ret  = cr.get("return_%", 0)
            delta    = trix_ret - cur_ret

            def fmt_ret(v):
                return f"{v:+.2f}%" if v is not None else "—"

            row = [
                symbol,
                best_trix_tf or "—",
                fmt_ret(trix_ret),
                tr.get("trades", "—"),
                f"{tr.get('win_%', 0):.1f}%",
                f"{tr.get('drawdown_%', 0):.1f}%",
                fmt_ret(cur_ret),
                cr.get("trades", "—"),
                f"{cr.get('win_%', 0):.1f}%",
                f"{cr.get('drawdown_%', 0):.1f}%",
                f"{delta:+.2f}%",
            ]
            rows.append(row)

        # Ligne totaux
        trix_total    = sum(trix_results.get((s, TIMEFRAMES[2], years), {}).get("return_%", 0) for s in config.SYMBOLS)
        current_total = sum(current_results.get((s, TIMEFRAMES[2], years), {}).get("return_%", 0) for s in config.SYMBOLS)
        rows.append([
            "TOTAL (6h)",
            "6h",
            f"{trix_total:+.2f}%",
            "", "", "",
            f"{current_total:+.2f}%",
            "", "", "",
            f"{trix_total - current_total:+.2f}%"
        ])

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Tableau récap : TRIX par timeframe (total portefeuille)
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.CYAN}{'='*70}")
    print("  TRIX — TOTAL PORTEFEUILLE PAR TIMEFRAME ET PÉRIODE (avec frais)")
    print(f"{'='*70}{Style.RESET_ALL}")

    headers2 = ["Timeframe"] + [PERIODS[y] for y in PERIODS]
    rows2 = []
    for tf in TIMEFRAMES:
        row = [tf]
        for years in PERIODS:
            total = sum(
                trix_results.get((s, tf, years), {}).get("return_%", 0)
                for s in config.SYMBOLS
            )
            n_pos = sum(
                1 for s in config.SYMBOLS
                if trix_results.get((s, tf, years), {}).get("return_%", 0) > 0
            )
            row.append(f"{total:+.1f}% ({n_pos}/20✓)")
        rows2.append(row)

    # Ligne stratégie actuelle (6h)
    row_cur = ["ACTUEL (6h)"]
    for years in PERIODS:
        total = sum(
            current_results.get((s, "6h", years), {}).get("return_%", 0)
            for s in config.SYMBOLS
        )
        n_pos = sum(
            1 for s in config.SYMBOLS
            if current_results.get((s, "6h", years), {}).get("return_%", 0) > 0
        )
        row_cur.append(f"{total:+.1f}% ({n_pos}/20✓)")
    rows2.append(row_cur)

    print(tabulate(rows2, headers=headers2, tablefmt="rounded_outline"))

    print(f"\n{Fore.GREEN}Note : frais inclus ({FEE_RATE*100:.1f}% achat + {FEE_RATE*100:.1f}% vente){Style.RESET_ALL}")
