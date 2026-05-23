"""
Optimiseur de Stop-Loss et Take-Profit.
Teste toutes les combinaisons SL/TP sur les meilleures configs identifiées
et retourne le classement des couples (SL, TP) les plus performants.
"""

import time
import ccxt
import pandas as pd
from itertools import product
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators

init(autoreset=True)

# 20 cryptos × timeframe optimal (6h par défaut, 12h pour BTC)
TARGET_CONFIGS = [
    ("BTC/USDT",  "12h"),
    ("ETH/USDT",  "6h"),
    ("BNB/USDT",  "6h"),
    ("SOL/USDT",  "6h"),
    ("XRP/USDT",  "6h"),
    ("ADA/USDT",  "6h"),
    ("AVAX/USDT", "6h"),
    ("DOT/USDT",  "6h"),
    ("LINK/USDT", "6h"),
    ("MATIC/USDT","6h"),
    ("UNI/USDT",  "6h"),
    ("ATOM/USDT", "6h"),
    ("NEAR/USDT", "6h"),
    ("LTC/USDT",  "6h"),
    ("DOGE/USDT", "6h"),
    ("TRX/USDT",  "6h"),
    ("ALGO/USDT", "6h"),
    ("AAVE/USDT", "6h"),
    ("ARB/USDT",  "6h"),
    ("OP/USDT",   "6h"),
]

# Grille élargie — SL plus larges nécessaires avec simulation high/low
STOP_LOSS_VALUES   = [0.05, 0.07, 0.10, 0.12, 0.15, 0.20]        # 5% à 20%
TAKE_PROFIT_VALUES = [0.07, 0.10, 0.15, 0.20, 0.25, 0.30]        # 7% à 30%

CANDLES_PER_TF = {
    "30m": 70080, "1h": 35040, "2h": 17520,
    "4h": 8760,   "6h": 5840,  "12h": 2920, "1d": 1460,
}

# Cache des données pour ne pas re-télécharger à chaque run
_data_cache: dict[str, pd.DataFrame] = {}


def fetch_and_cache(symbol: str, timeframe: str) -> pd.DataFrame:
    key = f"{symbol}_{timeframe}"
    if key in _data_cache:
        return _data_cache[key]

    exchange = ccxt.binance({"enableRateLimit": True})
    target = CANDLES_PER_TF.get(timeframe, 500)
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - target * tf_ms
    all_candles = []

    print(f"  Téléchargement {symbol} {timeframe}...", end=" ", flush=True)
    while len(all_candles) < target:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        except Exception:
            time.sleep(1)
            continue
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
        time.sleep(exchange.rateLimit / 1000)

    print(f"{len(all_candles)} bougies")
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    df = indicators.compute_all(df)
    df = df.dropna().reset_index()
    _data_cache[key] = df
    return df


def simulate(df: pd.DataFrame, symbol: str, sl: float, tp: float,
             initial_capital: float = 1000.0) -> dict:
    """
    Simulation rapide avec SL/TP déclenchés sur hauts/bas intracandle.
      - SL : déclenché si candle LOW  <= stop_loss_price  → exit au prix SL
      - TP : déclenché si candle HIGH >= take_profit_price → exit au prix TP
      - Si les deux dans la même bougie : SL en priorité (cas conservateur)
    """
    capital = initial_capital
    position = None   # {"entry": float, "size": float, "sl": float, "tp": float}
    trades = []
    equity = [capital]

    for i in range(2, len(df)):
        candle      = df.iloc[i]
        candle_low  = candle["low"]
        candle_high = candle["high"]
        close_price = candle["close"]

        # Vérification SL/TP sur hauts/bas intracandle
        if position:
            sl_hit = candle_low  <= position["sl"]
            tp_hit = candle_high >= position["tp"]

            if sl_hit or tp_hit:
                exit_price = position["sl"] if sl_hit else position["tp"]
                pnl = (exit_price - position["entry"]) * position["size"]
                capital += pnl
                trades.append(pnl)
                position = None
                equity.append(capital)
                continue

        window = df.iloc[: i + 1].set_index("timestamp")
        score = indicators.score_signal(window)

        if score["signal"] == "BUY" and position is None:
            size = (capital * config.POSITION_SIZE_PCT) / close_price
            position = {
                "entry": close_price,
                "size": size,
                "sl": close_price * (1 - sl),
                "tp": close_price * (1 + tp),
            }

        elif score["signal"] == "SELL" and position is not None:
            pnl = (close_price - position["entry"]) * position["size"]
            capital += pnl
            trades.append(pnl)
            position = None

        equity.append(capital)

    # Ferme la position restante au dernier prix
    if position:
        last_price = df.iloc[-1]["close"]
        pnl = (last_price - position["entry"]) * position["size"]
        capital += pnl
        trades.append(pnl)

    if not trades:
        return {}

    wins = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "sl_%": round(sl * 100, 0),
        "tp_%": round(tp * 100, 0),
        "trades": len(trades),
        "win_%": round(len(wins) / len(trades) * 100, 1),
        "return_%": round((capital - initial_capital) / initial_capital * 100, 2),
        "capital": round(capital, 2),
        "profit_factor": round(pf, 2),
        "avg_win_$": round(sum(wins) / len(wins) if wins else 0, 2),
        "avg_loss_$": round(sum(losses) / len(losses) if losses else 0, 2),
        "drawdown_%": round(max_dd, 2),
    }


def optimize(symbol: str, timeframe: str):
    print(f"\n{Fore.CYAN}Optimisation {symbol} {timeframe} "
          f"({len(STOP_LOSS_VALUES) * len(TAKE_PROFIT_VALUES)} combinaisons){Style.RESET_ALL}")

    df = fetch_and_cache(symbol, timeframe)
    results = []

    total = len(STOP_LOSS_VALUES) * len(TAKE_PROFIT_VALUES)
    done = 0
    for sl, tp in product(STOP_LOSS_VALUES, TAKE_PROFIT_VALUES):
        if tp <= sl:
            done += 1
            continue  # Take-profit doit être supérieur au stop-loss
        stats = simulate(df, symbol, sl, tp)
        if stats:
            stats["symbol"] = symbol
            stats["timeframe"] = timeframe
            results.append(stats)
        done += 1
        print(f"\r  Progression: {done}/{total}", end="", flush=True)

    print()
    return results


def print_top(results: list[dict], symbol: str, timeframe: str, n: int = 10):
    filtered = [r for r in results if r.get("symbol") == symbol and r.get("timeframe") == timeframe]
    if not filtered:
        print("Aucun résultat.")
        return

    # Trie par return puis profit_factor
    filtered.sort(key=lambda x: (x["return_%"], x["profit_factor"]), reverse=True)
    top = filtered[:n]

    cols = ["sl_%", "tp_%", "trades", "win_%", "return_%", "capital",
            "profit_factor", "avg_win_$", "avg_loss_$", "drawdown_%"]
    rows = [[r.get(c, "—") for c in cols] for r in top]

    print(f"\n{Fore.YELLOW}Top {n} — {symbol} {timeframe}{Style.RESET_ALL}")
    print(tabulate(rows, headers=cols, tablefmt="rounded_outline", floatfmt=".2f"))

    best = top[0]
    print(
        f"{Fore.GREEN}Meilleur: SL={best['sl_%']}% / TP={best['tp_%']}% → "
        f"Return {best['return_%']}% | Win {best['win_%']}% | "
        f"PF {best['profit_factor']} | Drawdown {best['drawdown_%']}%{Style.RESET_ALL}"
    )


def print_summary_table(all_results: list[dict]):
    """Tableau récapitulatif : une ligne par (symbol, timeframe) avec la meilleure config SL/TP."""
    # Pour chaque (symbol, timeframe), garde uniquement le meilleur résultat
    best_per_config: dict[tuple, dict] = {}
    for r in all_results:
        key = (r["symbol"], r["timeframe"])
        if key not in best_per_config or r["return_%"] > best_per_config[key]["return_%"]:
            best_per_config[key] = r

    # Trie par return décroissant
    rows_sorted = sorted(best_per_config.values(), key=lambda x: x["return_%"], reverse=True)

    cols = ["symbol", "timeframe", "sl_%", "tp_%", "trades", "win_%",
            "return_%", "capital", "profit_factor", "drawdown_%"]
    rows = [[r.get(c, "—") for c in cols] for r in rows_sorted]

    print(f"\n{Fore.YELLOW}{'='*80}")
    print("  TABLEAU COMPLET — Meilleur SL/TP par timeframe (2 ans de données)")
    print(f"{'='*80}{Style.RESET_ALL}")
    print(tabulate(rows, headers=cols, tablefmt="rounded_outline", floatfmt=".2f"))

    best = rows_sorted[0]
    print(f"\n{Fore.GREEN}Champion: {best['symbol']} {best['timeframe']} | "
          f"SL={best['sl_%']}% / TP={best['tp_%']}% | "
          f"Return {best['return_%']}% | {best['trades']} trades | "
          f"Win {best['win_%']}% | Drawdown {best['drawdown_%']}%{Style.RESET_ALL}")


if __name__ == "__main__":
    all_results = []

    for symbol, timeframe in TARGET_CONFIGS:
        results = optimize(symbol, timeframe)
        all_results.extend(results)

    print_summary_table(all_results)
