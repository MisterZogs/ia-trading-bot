"""
Regime Detection — Filtre ADX sur les meilleures configurations
===============================================================
Compare les performances avec et sans filtre de régime basé sur l'ADX.

Logique :
  ADX > 25 → marché en tendance forte (directionnelle)
           → la mean-reversion est moins efficace → bloquer les BUY
  ADX ≤ 25 → marché en consolidation (ranging)
           → mean-reversion est plus fiable → laisser passer les BUY

Usage :
    python3 etape9_regime.py
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import data_cache
import multi_sim
import fear_greed

init(autoreset=True)

# ---------------------------------------------------------------------------
# Configurations à tester (4 dernières années, multi, sansSL)
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "label":         "Top20/12h/épurée",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  False,
        "use_sl":        False,
    },
    {
        "label":         "Top5/12h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  True,
        "use_sl":        False,
    },
    {
        "label":         "BTC+ETH/12h/+TriST",
        "symbols":       multi_sim.PORTFOLIOS["btceth"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": True,
        "use_sma_macd":  True,
        "use_sl":        False,
    },
    {
        "label":         "Top5/6h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "6h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  True,
        "use_sl":        False,
    },
    {
        "label":         "Top20/2h/épurée+ST",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "timeframe":     "2h",
        "years":         4,
        "use_triple_st": True,
        "use_sma_macd":  False,
        "use_sl":        False,
    },
]

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def fmt_ret(v: float) -> str:
    color = Fore.GREEN if v > 0 else Fore.RED
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"


def fmt_delta(v: float, positive_is_good: bool = True) -> str:
    if positive_is_good:
        color = Fore.GREEN if v > 1 else (Fore.RED if v < -1 else "")
    else:
        color = Fore.GREEN if v < -1 else (Fore.RED if v > 1 else "")
    sign = "+" if v >= 0 else ""
    return f"{color}{sign}{v:.1f}{Style.RESET_ALL}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  REGIME DETECTION — Filtre ADX(14) : BUY bloqué si ADX > 25")
    print(f"{'='*90}{Style.RESET_ALL}")
    print(f"  Principe : la mean-reversion est moins fiable en marché directionnel.")
    print(f"  Quand ADX > 25, le marché est en tendance → on reste flat.")
    print(f"  Quand ADX ≤ 25, le marché est en consolidation → signaux BUY autorisés.")

    # Tous les symboles et timeframes nécessaires
    all_symbols = list(dict.fromkeys(
        s for cfg in CONFIGS for s in cfg["symbols"]
    ))
    all_tfs = list(dict.fromkeys(cfg["timeframe"] for cfg in CONFIGS))

    print(f"\n{Fore.CYAN}Chargement des données (4 ans)...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, all_tfs, verbose=True)

    print(f"{Fore.CYAN}Calcul des indicateurs (avec ADX)...{Style.RESET_ALL}")
    done = 0
    total = len(all_symbols) * len(all_tfs)
    for symbol in all_symbols:
        for tf in all_tfs:
            multi_sim.get_df(symbol, tf)
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)
    print()

    print(f"{Fore.CYAN}Chargement Fear & Greed Index...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # ---------------------------------------------------------------------------
    # Test : sans vs avec filtre régime
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Comparaison sans / avec filtre régime ADX > 25 (4 ans / multi / sansSL){Style.RESET_ALL}")

    headers = [
        "Stratégie",
        "Sans filtre", "DD sans", "Trades",
        "Avec ADX",   "DD avec", "Trades",
        "Δ Return",   "Δ DD",    "Δ Trades",
    ]
    rows = []

    summary = []  # pour verdict final

    for cfg in CONFIGS:
        tf = cfg["timeframe"]
        n  = multi_sim.CANDLES_PER_YEAR[tf] * cfg["years"]

        # Slicer les DFs (4 dernières années)
        dfs: dict = {}
        for symbol in cfg["symbols"]:
            df = multi_sim.get_df(symbol, tf)
            if df is not None and len(df) >= 10:
                dfs[symbol] = df.tail(n).reset_index(drop=True)

        if not dfs:
            rows.append([cfg["label"]] + ["—"] * 9)
            continue

        # Sans filtre régime (baseline)
        r_base = multi_sim.sim_multi_on_dfs(
            dfs, use_sl=cfg["use_sl"], fg=fg_data,
            use_triple_st=cfg["use_triple_st"],
            use_sma_macd=cfg["use_sma_macd"],
            use_regime_filter=False,
        )

        # Avec filtre régime ADX
        r_reg = multi_sim.sim_multi_on_dfs(
            dfs, use_sl=cfg["use_sl"], fg=fg_data,
            use_triple_st=cfg["use_triple_st"],
            use_sma_macd=cfg["use_sma_macd"],
            use_regime_filter=True,
        )

        ret_base  = r_base.get("return_%", 0.0)
        dd_base   = r_base.get("drawdown_%", 0.0)
        t_base    = r_base.get("trades", 0)
        ret_reg   = r_reg.get("return_%", 0.0)
        dd_reg    = r_reg.get("drawdown_%", 0.0)
        t_reg     = r_reg.get("trades", 0)

        delta_ret = round(ret_reg - ret_base, 1)
        delta_dd  = round(dd_reg  - dd_base,  1)
        delta_t   = t_reg - t_base

        summary.append({
            "label":     cfg["label"],
            "ret_base":  ret_base,
            "ret_reg":   ret_reg,
            "dd_base":   dd_base,
            "dd_reg":    dd_reg,
            "delta_ret": delta_ret,
            "delta_dd":  delta_dd,
        })

        rows.append([
            cfg["label"],
            fmt_ret(ret_base),
            f"{dd_base:.1f}%",
            str(t_base),
            fmt_ret(ret_reg),
            f"{dd_reg:.1f}%",
            str(t_reg),
            fmt_delta(delta_ret, positive_is_good=True),
            fmt_delta(delta_dd,  positive_is_good=False),
            fmt_delta(delta_t,   positive_is_good=True),
        ])

    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Analyse du % de temps en régime trending (ADX > 25)
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Proportion du temps en régime trending (ADX > 25) :{Style.RESET_ALL}")
    adx_headers = ["Symbole", "Timeframe", "% temps ADX>25", "ADX moy", "ADX max"]
    adx_rows = []

    # Calculer sur BTC et ETH (représentatifs)
    for symbol in ["BTC/USDT", "ETH/USDT"]:
        for tf in all_tfs:
            df = multi_sim.get_df(symbol, tf)
            if df is None or "adx" not in df.columns:
                continue
            n = multi_sim.CANDLES_PER_YEAR[tf] * 4
            sliced = df.tail(n)
            adx = sliced["adx"].dropna()
            if len(adx) == 0:
                continue
            pct_trending = round((adx > 25).sum() / len(adx) * 100, 1)
            adx_mean = round(adx.mean(), 1)
            adx_max  = round(adx.max(), 1)
            adx_rows.append([symbol, tf, f"{pct_trending}%", str(adx_mean), str(adx_max)])

    if adx_rows:
        print(tabulate(adx_rows, headers=adx_headers, tablefmt="rounded_outline"))
    else:
        print("  Colonne ADX non disponible — relancer après calcul des indicateurs.")

    # ---------------------------------------------------------------------------
    # Verdict
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Verdict :{Style.RESET_ALL}")
    improvements = [s for s in summary if s["delta_ret"] > 2]
    degradations = [s for s in summary if s["delta_ret"] < -2]

    if len(improvements) > len(degradations):
        print(f"  {Fore.GREEN}Le filtre ADX améliore globalement les performances.{Style.RESET_ALL}")
        print(f"  Recommandation : activer use_regime_filter=True en production.")
    elif len(degradations) > len(improvements):
        print(f"  {Fore.RED}Le filtre ADX dégrade globalement les performances.{Style.RESET_ALL}")
        print(f"  Raison probable : le marché crypto est souvent en tendance (ADX > 25 > 40% du temps).")
        print(f"  Recommandation : conserver la stratégie sans filtre régime.")
    else:
        print(f"  {Fore.YELLOW}Résultats mixtes — le filtre ADX n'a pas d'impact significatif.{Style.RESET_ALL}")
        print(f"  Recommandation : tester un seuil ADX différent (ex. 30 ou 35).")

    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print("  • Δ Return > 0  → le filtre améliore le return (BUY évités en tendance = moins de faux signaux)")
    print("  • Δ Return < 0  → le filtre nuit (trop de BUY bloqués, marché souvent en tendance)")
    print("  • Δ DD < 0      → le filtre réduit le drawdown (positions évitées en marché adverse)")
    print("  • Δ Trades < 0  → moins de trades (normal : le filtre bloque des BUY)")
    print()
    print("  ADX > 25 = tendance forte. Pour crypto (marchés volatils) :")
    print("  → si ADX > 25 plus de 40% du temps, le filtre est trop restrictif")
    print("  → essayez ADX > 30 ou ADX > 35 pour moins d'exclusions")


if __name__ == "__main__":
    main()
