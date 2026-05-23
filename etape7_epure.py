"""
Étape 7 — Stratégie épurée : retrait de b3 (SMA20<SMA50) et b4 (MACD<Signal)

Résultats ML + ablation : ces deux conditions ont un effet négatif sur la performance.
On compare 4 variantes :
  A : 8 conditions (actuel, avec TripleST)
  B : 6 conditions (sans b3/b4, avec TripleST)
  C : 8 conditions sans TripleST (baseline)
  D : 6 conditions sans b3/b4 ni TripleST

MIN_SCORE testé à 2 (sur 6 max) et 3 (sur 6 max) pour la stratégie épurée.

Configs : Top20, Top5, BTC+ETH  |  TF : 12h  |  Périodes : 1-4 ans
Mode : multi / sansSL
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import fear_greed
import multi_sim as ms

init(autoreset=True)

FOCUS_PORTS = {
    "top20":  config.SYMBOLS,
    "top5":   ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
    "btceth": ["BTC/USDT", "ETH/USDT"],
}
PORT_LABELS   = {"top20": "Top 20", "top5": "Top 5", "btceth": "BTC+ETH"}
FOCUS_TF      = ["6h", "12h"]
FOCUS_PERIODS = {1: "1an", 2: "2ans", 3: "3ans", 4: "4ans"}

VERSIONS = {
    "A 8cond+ST (actuel)":   {"use_triple_st": True,  "use_sma_macd": True,  "min_score": 3},
    "B 6cond+ST ms=2":       {"use_triple_st": True,  "use_sma_macd": False, "min_score": 2},
    "B 6cond+ST ms=3":       {"use_triple_st": True,  "use_sma_macd": False, "min_score": 3},
    "C 8cond-ST (baseline)": {"use_triple_st": False, "use_sma_macd": True,  "min_score": 3},
    "D 6cond-ST ms=2":       {"use_triple_st": False, "use_sma_macd": False, "min_score": 2},
    "D 6cond-ST ms=3":       {"use_triple_st": False, "use_sma_macd": False, "min_score": 3},
}


def buy_and_hold(symbols, tf, years):
    n = ms.CANDLES_PER_YEAR[tf] * years
    returns = []
    for sym in symbols:
        df = ms.get_df(sym, tf)
        if df is None or len(df) < 10:
            continue
        dp = df.tail(n)
        s, e = dp["close"].iloc[0], dp["close"].iloc[-1]
        if s > 0:
            returns.append((e - s) / s * 100)
    return round(sum(returns) / len(returns), 1) if returns else 0.0


def run(syms, tf, years, use_triple_st, use_sma_macd, min_score, fg):
    # Patch temporaire du MIN_SCORE
    import config as cfg
    original = cfg.MIN_SCORE_TO_TRADE
    cfg.MIN_SCORE_TO_TRADE = min_score
    try:
        r = ms.sim_multi(syms, tf, years, use_sl=False, fg=fg,
                         use_triple_st=use_triple_st, use_sma_macd=use_sma_macd)
    finally:
        cfg.MIN_SCORE_TO_TRADE = original
    return r


if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*110}")
    print("  ÉTAPE 7 — Stratégie épurée (sans b3 SMA20<SMA50 et b4 MACD<Signal)")
    print(f"{'='*110}{Style.RESET_ALL}")

    all_symbols = list(dict.fromkeys(s for syms in FOCUS_PORTS.values() for s in syms))
    print(f"\n{Fore.CYAN}Chargement données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, FOCUS_TF, verbose=False)
    for sym in all_symbols:
        for tf in FOCUS_TF:
            ms.get_df(sym, tf)
    print(f"  {len(all_symbols)} symboles × {len(FOCUS_TF)} TF")

    print(f"{Fore.CYAN}Fear & Greed...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # B&H cache
    bah_cache = {}
    for tf in FOCUS_TF:
        for port, syms in FOCUS_PORTS.items():
            for years in FOCUS_PERIODS:
                bah_cache[(port, tf, years)] = buy_and_hold(syms, tf, years)

    # Simulations
    total = len(VERSIONS) * len(FOCUS_PORTS) * len(FOCUS_TF) * len(FOCUS_PERIODS)
    done  = 0
    all_results = {}

    print(f"{Fore.CYAN}Simulations ({total} backtests)...{Style.RESET_ALL}")
    for ver_name, params in VERSIONS.items():
        for tf in FOCUS_TF:
            for port, syms in FOCUS_PORTS.items():
                for years in FOCUS_PERIODS:
                    try:
                        r = run(syms, tf, years,
                                use_triple_st=params["use_triple_st"],
                                use_sma_macd=params["use_sma_macd"],
                                min_score=params["min_score"],
                                fg=fg_data)
                    except Exception as e:
                        r = {}
                    all_results[(ver_name, port, tf, years)] = r
                    done += 1
                    print(f"\r  {done}/{total}", end="", flush=True)
    print()

    # Affichage par TF
    for tf in FOCUS_TF:
        print(f"\n{Fore.YELLOW}{'='*110}")
        print(f"  TF : {tf.upper()} | multi | sansSL | 1000 USDT")
        print(f"{'='*110}{Style.RESET_ALL}")

        headers = ["Portfolio", "Période"] + list(VERSIONS.keys()) + ["B&H"]
        rows = []
        for port in FOCUS_PORTS:
            for years, plabel in FOCUS_PERIODS.items():
                row = [PORT_LABELS[port], plabel]
                best_ret = max(
                    (all_results.get((v, port, tf, years), {}).get("return_%") or -999)
                    for v in VERSIONS
                )
                for ver_name in VERSIONS:
                    r   = all_results.get((ver_name, port, tf, years), {})
                    ret = r.get("return_%")
                    if ret is None:
                        row.append("—")
                    else:
                        bold  = ret == best_ret
                        color = Fore.GREEN if ret > 0 else (Fore.RED if ret < 0 else "")
                        s = f"{color}{ret:+.1f}%{Style.RESET_ALL}"
                        row.append(f"★{s}" if bold else s)
                bah = bah_cache.get((port, tf, years), 0.0)
                row.append(f"{bah:+.1f}%")
                rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # Résumé : meilleure version par config clé
    print(f"\n{Fore.CYAN}{'='*110}")
    print("  RÉSUMÉ — Meilleure version par config (★ = gagnante)")
    print(f"{'='*110}{Style.RESET_ALL}")

    KEY_CONFIGS = [
        ("Top20", "top20", "12h", 4),
        ("Top20", "top20", "12h", 3),
        ("Top20", "top20",  "6h", 4),
        ("Top5",  "top5",  "12h", 4),
        ("Top5",  "top5",  "12h", 3),
        ("BTC+ETH","btceth","12h",4),
    ]

    sum_headers = ["Config", "TF", "Période", "Meilleure version", "Return", "DD", "Win%", "Trades", "B&H", "Alpha"]
    sum_rows = []
    for label, port, tf, years in KEY_CONFIGS:
        best_ver, best_ret, best_r = None, None, {}
        for ver_name in VERSIONS:
            r   = all_results.get((ver_name, port, tf, years), {})
            ret = r.get("return_%")
            if ret is not None and (best_ret is None or ret > best_ret):
                best_ret = ret
                best_ver = ver_name
                best_r   = r
        bah = bah_cache.get((port, tf, years), 0.0)
        color = Fore.GREEN if best_ret and best_ret > 0 else Fore.RED
        sum_rows.append([
            label, tf, FOCUS_PERIODS[years],
            best_ver or "—",
            f"{color}{best_ret:+.1f}%{Style.RESET_ALL}" if best_ret else "—",
            f"{best_r.get('drawdown_%',0):+.1f}%",
            f"{best_r.get('win_%',0):.0f}%",
            best_r.get("trades", 0),
            f"{bah:+.1f}%",
            f"{round(best_ret-bah,1):+.1f}%" if best_ret else "—",
        ])
    print(tabulate(sum_rows, headers=sum_headers, tablefmt="rounded_outline"))
