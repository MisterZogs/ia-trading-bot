"""
Comparaison Étape 4 — Triple SuperTrend + ATR Trailing Stop
4 versions testées sur les meilleures configs identifiées :
  A : baseline (sansSL, sans Triple ST)
  B : + Triple SuperTrend (8ème condition)
  C : + ATR Trailing Stop (sans Triple ST)
  D : + Triple ST + ATR Trailing Stop

Portfolios : Top20, Top5, BTC+ETH
Timeframes : 6h, 12h (les plus performants)
Périodes   : 1an, 2ans, 3ans, 4ans
Colonne B&H pour référence
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators
import data_cache
import fear_greed
import multi_sim as ms

init(autoreset=True)

FOCUS_PORTS = {
    "top20":  config.SYMBOLS,
    "top5":   ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"],
    "btceth": ["BTC/USDT", "ETH/USDT"],
}

FOCUS_TF      = ["6h", "12h"]
FOCUS_PERIODS = {1: "1an", 2: "2ans", 3: "3ans", 4: "4ans"}
PORT_LABELS   = {"top20": "Top 20", "top5": "Top 5", "btceth": "BTC+ETH"}

VERSIONS = {
    "A — baseline":       {"trail_sl": False, "use_triple_st": False},
    "B — +TripleST":      {"trail_sl": False, "use_triple_st": True},
    "C — +ATRTrail":      {"trail_sl": True,  "use_triple_st": False},
    "D — +TripleST+ATR":  {"trail_sl": True,  "use_triple_st": True},
}


def run_version(port_syms, tf, years, trail_sl, use_triple_st, fg):
    return ms.sim_multi(
        port_syms, tf, years,
        use_sl=False,
        fg=fg,
        trail_sl=trail_sl,
        use_triple_st=use_triple_st,
    )


def buy_and_hold(symbols, tf, years):
    n = ms.CANDLES_PER_YEAR[tf] * years
    returns = []
    for symbol in symbols:
        df = ms.get_df(symbol, tf)
        if df is None or len(df) < 10:
            continue
        dp = df.tail(n)
        s, e = dp["close"].iloc[0], dp["close"].iloc[-1]
        if s > 0:
            returns.append((e - s) / s * 100)
    if not returns:
        return 0.0
    return round(sum(returns) / len(returns), 1)


if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  ÉTAPE 4 — Triple SuperTrend + ATR Trailing Stop")
    print(f"{'='*90}{Style.RESET_ALL}")

    # Chargement données
    all_symbols = list(dict.fromkeys(s for syms in FOCUS_PORTS.values() for s in syms))
    print(f"\n{Fore.CYAN}Chargement données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, FOCUS_TF, verbose=True)

    # Calcul indicateurs (cache mémoire commun — les colonnes ST sont toujours calculées)
    print(f"{Fore.CYAN}Calcul indicateurs...{Style.RESET_ALL}")
    total_pairs = len(all_symbols) * len(FOCUS_TF)
    done = 0
    for symbol in all_symbols:
        for tf in FOCUS_TF:
            ms.get_df(symbol, tf)
            done += 1
            print(f"\r  {done}/{total_pairs}", end="", flush=True)
    print()

    # Fear & Greed
    print(f"{Fore.CYAN}Chargement Fear & Greed...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=True)

    # Buy-and-hold cache
    bah_cache = {}
    for tf in FOCUS_TF:
        for port, syms in FOCUS_PORTS.items():
            for years in FOCUS_PERIODS:
                bah_cache[(port, tf, years)] = buy_and_hold(syms, tf, years)

    # -----------------------------------------------------------------------
    # Simulations — 4 versions × 3 ports × 2 TF × 4 périodes = 96 backtests
    # -----------------------------------------------------------------------
    all_results = {}
    total = len(VERSIONS) * len(FOCUS_PORTS) * len(FOCUS_TF) * len(FOCUS_PERIODS)
    done  = 0

    for ver_name, ver_params in VERSIONS.items():
        print(f"\n{Fore.YELLOW}Version {ver_name}...{Style.RESET_ALL}")
        for tf in FOCUS_TF:
            for port, syms in FOCUS_PORTS.items():
                for years in FOCUS_PERIODS:
                    try:
                        r = run_version(syms, tf, years,
                                        trail_sl=ver_params["trail_sl"],
                                        use_triple_st=ver_params["use_triple_st"],
                                        fg=fg_data)
                    except Exception as e:
                        print(f"  Erreur {port}/{tf}/{years}y: {e}")
                        r = {}
                    all_results[(ver_name, port, tf, years)] = r
                    done += 1
                    print(f"\r  {done}/{total}", end="", flush=True)
    print()

    # -----------------------------------------------------------------------
    # Affichage par TF
    # -----------------------------------------------------------------------
    for tf in FOCUS_TF:
        print(f"\n{Fore.YELLOW}{'='*90}")
        print(f"  TIMEFRAME : {tf.upper()} | mode multi | sansSL | frais 0.1% | Capital 1000 USDT")
        print(f"{'='*90}{Style.RESET_ALL}")

        headers = ["Portfolio", "Période"] + list(VERSIONS.keys()) + ["B&H"]
        rows = []

        for port in FOCUS_PORTS:
            for years, plabel in FOCUS_PERIODS.items():
                row = [PORT_LABELS[port], plabel]
                for ver_name in VERSIONS:
                    r   = all_results.get((ver_name, port, tf, years), {})
                    ret = r.get("return_%")
                    if ret is None:
                        row.append("—")
                    else:
                        color = Fore.GREEN if ret > 0 else (Fore.RED if ret < 0 else "")
                        row.append(f"{color}{ret:+.1f}%{Style.RESET_ALL}")
                bah = bah_cache.get((port, tf, years), 0.0)
                row.append(f"{bah:+.1f}%")
                rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # -----------------------------------------------------------------------
    # Résumé : meilleure version par TF + période (top20)
    # -----------------------------------------------------------------------
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  RÉSUMÉ — Meilleure version par TF × Période (mode multi, top20)")
    print(f"{'='*90}{Style.RESET_ALL}")

    sum_headers = ["TF", "Période", "Meilleure version", "Return", "B&H"]
    sum_rows = []
    for tf in FOCUS_TF:
        for years, plabel in FOCUS_PERIODS.items():
            best_ver, best_ret = None, None
            for ver_name in VERSIONS:
                r   = all_results.get((ver_name, "top20", tf, years), {})
                ret = r.get("return_%")
                if ret is not None and (best_ret is None or ret > best_ret):
                    best_ret = ret
                    best_ver = ver_name
            bah = bah_cache.get(("top20", tf, years), 0.0)
            sum_rows.append([tf, plabel,
                             best_ver or "—",
                             f"{best_ret:+.1f}%" if best_ret is not None else "—",
                             f"{bah:+.1f}%"])
    print(tabulate(sum_rows, headers=sum_headers, tablefmt="rounded_outline"))
