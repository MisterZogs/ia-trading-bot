"""
Walk-Forward Validation — 8 fenêtres annuelles glissantes
==========================================================
Teste les meilleures stratégies sur chacune des 8 années
d'historique pour vérifier la consistance et détecter l'overfitting.

Une stratégie robuste doit être profitable sur au moins 6/8 années.
Si elle n'est bonne que sur les 4 dernières années, c'est du surapprentissage.

Usage :
    python3 etape8_walkforward.py
"""

import sys
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import data_cache
import multi_sim
import fear_greed

init(autoreset=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TIMEFRAME = "12h"
N1Y = multi_sim.CANDLES_PER_YEAR[TIMEFRAME]   # 730 bougies par an

# Meilleures stratégies à tester (top du full_ranking)
CONFIGS = [
    {
        "label":         "Top20/12h/épurée",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "use_triple_st": False,
        "use_sma_macd":  False,
    },
    {
        "label":         "Top5/12h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "use_triple_st": False,
        "use_sma_macd":  True,
    },
    {
        "label":         "BTC+ETH/12h/+TriST",
        "symbols":       multi_sim.PORTFOLIOS["btceth"],
        "use_triple_st": True,
        "use_sma_macd":  True,
    },
    {
        "label":         "Top20/12h/épurée+ST",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "use_triple_st": True,
        "use_sma_macd":  False,
    },
]

# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------
def fmt_ret(v: float | None) -> str:
    if v is None:
        return "—"
    color = Fore.GREEN if v > 0 else Fore.RED
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"


def buy_and_hold_window(symbols: list[str], dfs: dict, start: int, end: int) -> float:
    """B&H moyen équipondéré sur la fenêtre [start:end]."""
    returns = []
    for s in symbols:
        if s not in dfs:
            continue
        sliced = dfs[s].iloc[start:end]
        if len(sliced) < 2:
            continue
        p0, p1 = sliced["close"].iloc[0], sliced["close"].iloc[-1]
        if p0 > 0:
            returns.append((p1 - p0) / p0 * 100)
    return round(sum(returns) / len(returns), 1) if returns else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  WALK-FORWARD VALIDATION — 8 fenêtres d'1 an sur données 8 ans")
    print(f"{'='*90}{Style.RESET_ALL}")

    # Tous les symboles nécessaires
    all_symbols = list(dict.fromkeys(
        s for cfg in CONFIGS for s in cfg["symbols"]
    ))

    # Chargement des données 8 ans
    print(f"\n{Fore.CYAN}Chargement des données 8 ans ({TIMEFRAME})...{Style.RESET_ALL}")
    data_cache.prefetch_all_8y(all_symbols, [TIMEFRAME], verbose=True)

    print(f"{Fore.CYAN}Calcul des indicateurs (+ ADX)...{Style.RESET_ALL}")
    full_dfs: dict = {}
    done = 0
    for symbol in all_symbols:
        df = multi_sim.get_df_8y(symbol, TIMEFRAME)
        if df is not None:
            full_dfs[symbol] = df
        done += 1
        print(f"\r  {done}/{len(all_symbols)}", end="", flush=True)
    print()

    if not full_dfs:
        print(f"{Fore.RED}Aucune donnée disponible.{Style.RESET_ALL}")
        sys.exit(1)

    max_len = max(len(df) for df in full_dfs.values())
    n_windows = max_len // N1Y
    if n_windows > 8:
        n_windows = 8
    print(f"  Données max : {max_len} bougies → {n_windows} fenêtres d'1 an\n")

    # Fear & Greed
    print(f"{Fore.CYAN}Chargement Fear & Greed Index...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # ---------------------------------------------------------------------------
    # Calcul fenêtre par fenêtre (de la plus ancienne à la plus récente)
    # ---------------------------------------------------------------------------
    results: dict[str, list] = {cfg["label"]: [] for cfg in CONFIGS}
    bah_vals: list = []
    window_labels: list = []

    for w in range(n_windows):
        # Fenêtre w=0 = la plus ancienne
        start_idx = max_len - (n_windows - w) * N1Y
        end_idx   = max_len - (n_windows - w - 1) * N1Y
        start_idx = max(0, start_idx)

        # Étiquette année (approximative — avril 2026 = dernier point)
        year_end   = 2026 - (n_windows - w - 1)
        year_start = year_end - 1
        window_labels.append(f"{year_start}-{str(year_end)[2:]}")

        # B&H de référence (Top20 sur cette fenêtre)
        bah_w = buy_and_hold_window(
            multi_sim.PORTFOLIOS["top20"], full_dfs, start_idx, end_idx
        )
        bah_vals.append(bah_w)

        for cfg in CONFIGS:
            # Slicer uniquement les symboles disponibles avec >= 50% de couverture
            window_dfs: dict = {}
            for symbol in cfg["symbols"]:
                if symbol not in full_dfs:
                    continue
                df = full_dfs[symbol]
                if end_idx > len(df):
                    sliced = df.iloc[start_idx:].reset_index(drop=True)
                else:
                    sliced = df.iloc[start_idx:end_idx].reset_index(drop=True)
                min_candles = int(N1Y * 0.5)
                if len(sliced) < min_candles:
                    continue
                window_dfs[symbol] = sliced

            if not window_dfs:
                results[cfg["label"]].append(None)
                continue

            r = multi_sim.sim_multi_on_dfs(
                window_dfs,
                use_sl=False,
                fg=fg_data,
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
            )
            results[cfg["label"]].append(r.get("return_%"))

    # ---------------------------------------------------------------------------
    # Tableau principal
    # ---------------------------------------------------------------------------
    headers = ["Stratégie"] + window_labels + ["Moy", "Pos/8", "Min", "B&H moy"]
    rows = []

    bah_avg = round(sum(bah_vals) / len(bah_vals), 1) if bah_vals else 0.0

    for cfg in CONFIGS:
        vals  = results[cfg["label"]]
        valid = [v for v in vals if v is not None]
        row   = [cfg["label"]]
        for v in vals:
            row.append(fmt_ret(v))

        if valid:
            avg = round(sum(valid) / len(valid), 1)
            pos = sum(1 for v in valid if v > 0)
            mn  = min(valid)
            color_pos = Fore.GREEN if pos >= 6 else (Fore.YELLOW if pos >= 4 else Fore.RED)
            row.append(f"{avg:+.1f}%")
            row.append(f"{color_pos}{pos}/{len(valid)}{Style.RESET_ALL}")
            row.append(f"{mn:+.1f}%")
        else:
            row += ["—", "—", "—"]

        row.append(f"{bah_avg:+.1f}%")
        rows.append(row)

    # Ligne B&H de référence
    bah_row = [f"{Fore.BLUE}Buy & Hold (Top20){Style.RESET_ALL}"]
    bah_row += [fmt_ret(v) for v in bah_vals]
    if bah_vals:
        avg_bah = round(sum(bah_vals) / len(bah_vals), 1)
        pos_bah = sum(1 for v in bah_vals if v > 0)
        mn_bah  = min(bah_vals)
        bah_row += [f"{avg_bah:+.1f}%", f"{pos_bah}/8", f"{mn_bah:+.1f}%", "—"]
    else:
        bah_row += ["—", "—", "—", "—"]
    rows.append(bah_row)

    print(f"{Fore.YELLOW}Walk-Forward : Return % par année et par stratégie (12h / multi / sansSL){Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Tableau détaillé par stratégie
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Détail par stratégie :{Style.RESET_ALL}")
    detail_headers = ["Stratégie", "Return moy", "Pos/8", "Min année", "Verdict"]
    detail_rows = []

    for cfg in CONFIGS:
        vals  = results[cfg["label"]]
        valid = [v for v in vals if v is not None]
        if not valid:
            detail_rows.append([cfg["label"], "—", "—", "—", "Données insuffisantes"])
            continue

        avg = round(sum(valid) / len(valid), 1)
        pos = sum(1 for v in valid if v > 0)
        mn  = min(valid)

        if pos >= 6 and avg > 10 and mn > -30:
            verdict = f"{Fore.GREEN}Robuste — stratégie fiable{Style.RESET_ALL}"
        elif pos >= 5 and avg > 0:
            verdict = f"{Fore.YELLOW}Acceptable — quelques années négatives{Style.RESET_ALL}"
        elif pos <= 3:
            verdict = f"{Fore.RED}Surapprentissage suspecté{Style.RESET_ALL}"
        else:
            verdict = f"{Fore.RED}Instable — prudence{Style.RESET_ALL}"

        detail_rows.append([
            cfg["label"],
            f"{avg:+.1f}%",
            f"{pos}/8",
            f"{mn:+.1f}%",
            verdict,
        ])

    print(tabulate(detail_rows, headers=detail_headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Légende
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print("  • Pos/8 ≥ 6      → stratégie robuste (profitable sur ≥ 75% des années)")
    print("  • Pos/8 ≤ 3      → surapprentissage suspecté (profitable < 50% du temps)")
    print("  • Min > -30%     → risque acceptable (perte annuelle max contrôlée)")
    print("  • Return moy > 10% → stratégie viable à long terme")
    print()
    print(f"  {Fore.YELLOW}Note :{Style.RESET_ALL} les fenêtres anciennes ont moins de symboles disponibles")
    print("  (les altcoins n'existaient pas avant 2020) → la fenêtre peut ne contenir")
    print("  que BTC/ETH/BNB pour les années 2018-2021.")


if __name__ == "__main__":
    main()
