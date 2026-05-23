"""
Monte Carlo Stress Test
=======================
Teste la robustesse de la stratégie en perturbant aléatoirement le timing
des entrées de 0 à 2 bougies (simulation de "on était en retard d'une bougie").

500 simulations → distribution des returns et drawdowns.

Répond à : "le +81% dépend-il d'un timing exact ou est-il robuste ?"
  • Si P5 > 50%  → très robuste, le résultat ne dépend pas du timing
  • Si P5 < 0%   → fragile, le timing est critique

Usage :
    python3 etape11_montecarlo.py
"""

import numpy as np
from numpy.random import default_rng
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import indicators
import multi_sim
import fear_greed

init(autoreset=True)

N_RUNS     = 500    # nombre de simulations Monte Carlo
MAX_SHIFT  = 2      # décalage max en bougies (0, 1 ou 2)
SEED       = 42     # reproductibilité

# ---------------------------------------------------------------------------
# Configurations à tester
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "label":         "Top20/12h/épurée",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  False,
    },
    {
        "label":         "BTC+ETH/12h/+TriST",
        "symbols":       multi_sim.PORTFOLIOS["btceth"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": True,
        "use_sma_macd":  True,
    },
    {
        "label":         "Top5/12h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  True,
    },
]


# ---------------------------------------------------------------------------
# Perturbation des signaux : décale les BUY de 0 à max_shift bougies
# ---------------------------------------------------------------------------
def perturb_sigs(sigs: np.ndarray, max_shift: int, rng) -> np.ndarray:
    """
    Pour chaque BUY au rang i, le déplace à i + k (k ∈ [0, max_shift]).
    Simule un retard d'exécution de 0 à 2 bougies.
    """
    out = sigs.copy()
    n   = len(out)
    i   = 0
    while i < n:
        if out[i] == "BUY":
            shift = int(rng.integers(0, max_shift + 1))
            if shift > 0:
                out[i] = "HOLD"
                j = i + shift
                if j < n and out[j] == "HOLD":
                    out[j] = "BUY"
            i += max(shift, 1)
        else:
            i += 1
    return out


# ---------------------------------------------------------------------------
# Histogramme ASCII
# ---------------------------------------------------------------------------
def ascii_hist(values: list, n_bins: int = 10, width: int = 40) -> str:
    arr  = np.array(values)
    lo, hi = arr.min(), arr.max()
    if lo == hi:
        return f"  Toutes les valeurs = {lo:.1f}%"
    bins  = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(arr, bins=bins)
    max_c = counts.max()
    lines = []
    for k in range(n_bins):
        bar_len = int(counts[k] / max_c * width)
        bar     = "█" * bar_len
        label   = f"{bins[k]:+6.1f}% à {bins[k+1]:+6.1f}%"
        lines.append(f"  {label} │{bar} {counts[k]}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"  MONTE CARLO STRESS TEST — {N_RUNS} simulations, décalage ±{MAX_SHIFT} bougies")
    print(f"{'='*80}{Style.RESET_ALL}")

    all_symbols = list(dict.fromkeys(s for cfg in CONFIGS for s in cfg["symbols"]))
    all_tfs     = list(dict.fromkeys(cfg["timeframe"] for cfg in CONFIGS))

    print(f"\n{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, all_tfs, verbose=True)

    print(f"{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    done, total = 0, len(all_symbols) * len(all_tfs)
    for sym in all_symbols:
        for tf in all_tfs:
            multi_sim.get_df(sym, tf)
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)
    print()

    print(f"{Fore.CYAN}Chargement Fear & Greed...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    rng = default_rng(SEED)

    summary_headers = [
        "Config", "Baseline", "P5", "P25", "Médiane", "P75", "P95",
        "% > 0%", "% > 50%", "DD médian", "DD P95",
    ]
    summary_rows = []

    for cfg in CONFIGS:
        tf  = cfg["timeframe"]
        n   = multi_sim.CANDLES_PER_YEAR[tf] * cfg["years"]

        # Préparer les DataFrames slicés
        dfs: dict = {}
        for sym in cfg["symbols"]:
            df = multi_sim.get_df(sym, tf)
            if df is not None and len(df) >= 10:
                dfs[sym] = df.tail(n).reset_index(drop=True)
        if not dfs:
            continue

        # Calculer les signaux de base une seule fois
        base_sigs = {
            s: indicators.vectorized_signals(
                dfs[s],
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
            ).values
            for s in dfs
        }

        # Run baseline (sans perturbation)
        r_base = multi_sim.sim_multi_on_dfs(
            dfs, use_sl=False, fg=fg_data,
            precomputed_sigs=base_sigs,
        )
        ret_base = r_base.get("return_%", 0)
        dd_base  = r_base.get("drawdown_%", 0)

        # Monte Carlo : N_RUNS avec signaux perturbés
        print(f"\n{Fore.CYAN}{cfg['label']} — {N_RUNS} runs...{Style.RESET_ALL}", end="", flush=True)
        mc_returns   = []
        mc_drawdowns = []

        for run in range(N_RUNS):
            perturbed = {s: perturb_sigs(base_sigs[s], MAX_SHIFT, rng) for s in base_sigs}
            r = multi_sim.sim_multi_on_dfs(
                dfs, use_sl=False, fg=fg_data,
                precomputed_sigs=perturbed,
            )
            mc_returns.append(r.get("return_%", 0))
            mc_drawdowns.append(r.get("drawdown_%", 0))
            if (run + 1) % 100 == 0:
                print(f" {run+1}", end="", flush=True)
        print()

        arr_r  = np.array(mc_returns)
        arr_dd = np.array(mc_drawdowns)

        p5   = np.percentile(arr_r, 5)
        p25  = np.percentile(arr_r, 25)
        p50  = np.percentile(arr_r, 50)
        p75  = np.percentile(arr_r, 75)
        p95  = np.percentile(arr_r, 95)
        pct_pos  = (arr_r > 0).mean() * 100
        pct_50   = (arr_r > 50).mean() * 100
        dd_p50   = np.percentile(arr_dd, 50)
        dd_p95   = np.percentile(arr_dd, 95)

        # Couleur selon robustesse
        def cr(v):
            return (Fore.GREEN if v >= 40 else (Fore.YELLOW if v >= 0 else Fore.RED)) + f"{v:+.1f}%{Style.RESET_ALL}"

        summary_rows.append([
            cfg["label"],
            f"{ret_base:+.1f}%",
            cr(p5), cr(p25), cr(p50), cr(p75), cr(p95),
            f"{pct_pos:.0f}%",
            f"{pct_50:.0f}%",
            f"{dd_p50:.1f}%",
            f"{dd_p95:.1f}%",
        ])

        # Histogramme des returns
        print(f"\n  Distribution des returns — {cfg['label']} (baseline = {ret_base:+.1f}%):")
        print(ascii_hist(mc_returns, n_bins=12, width=35))
        print(f"  σ = {arr_r.std():.1f}%  |  min = {arr_r.min():+.1f}%  |  max = {arr_r.max():+.1f}%")

    # ---------------------------------------------------------------------------
    # Tableau de synthèse
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  SYNTHÈSE MONTE CARLO")
    print(f"{'='*90}{Style.RESET_ALL}")
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print(f"  P5  = dans 95% des cas le return dépasse ce seuil (pire scénario réaliste)")
    print(f"  P95 = dans 5% des cas le return dépasse ce seuil (meilleur scénario)")
    print(f"  % > 0%  → proportion de runs profitables")
    print(f"  % > 50% → proportion de runs avec return > 50%")
    print()
    print(f"  {Fore.GREEN}Robuste{Style.RESET_ALL}  : P5 > +30% et % > 0% ≥ 95%")
    print(f"  {Fore.YELLOW}Correct{Style.RESET_ALL}  : P5 > 0%  et % > 0% ≥ 80%")
    print(f"  {Fore.RED}Fragile{Style.RESET_ALL}  : P5 < 0%  → timing critique, risque en live")


if __name__ == "__main__":
    main()
