"""
ATR Adaptif — Sizing & Take-Profit dynamiques
==============================================
Compare 4 variantes sur les meilleures configs (4 ans / multi / sansSL) :

  Baseline  : position fixe 5%, TP fixe par symbole
  +Sizing   : position adaptée à la volatilité ATR (réduite si marché agité)
  +TP ATR   : TP = entrée + 2.5×ATR (s'adapte à l'amplitude des bougies)
  +Les deux : sizing ATR + TP ATR combinés

Usage :
    python3 etape10_atr.py
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import multi_sim
import fear_greed

init(autoreset=True)

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
        "label":         "Top5/12h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "12h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  True,
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
        "label":         "Top5/6h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "6h",
        "years":         4,
        "use_triple_st": False,
        "use_sma_macd":  True,
    },
]

VARIANTS = [
    {"label": "Baseline",   "atr_sizing": False, "atr_tp": False},
    {"label": "+Sizing ATR","atr_sizing": True,  "atr_tp": False},
    {"label": "+TP ATR",    "atr_sizing": False, "atr_tp": True},
    {"label": "+Les deux",  "atr_sizing": True,  "atr_tp": True},
]

# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------
def fmt_ret(v: float) -> str:
    color = Fore.GREEN if v > 0 else Fore.RED
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"


def fmt_delta(v: float, positive_is_good: bool = True) -> str:
    if abs(v) < 0.5:
        return f"{v:+.1f}"
    color = (Fore.GREEN if v > 0 else Fore.RED) if positive_is_good \
        else (Fore.GREEN if v < 0 else Fore.RED)
    return f"{color}{v:+.1f}{Style.RESET_ALL}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  ATR ADAPTATIF — Sizing volatilité + Take-Profit dynamique")
    print(f"{'='*90}{Style.RESET_ALL}")
    print(f"  Sizing ref  : ATR/prix = {config.ATR_SIZING_REF_PCT*100:.1f}%"
          f"  |  échelle [{config.ATR_SIZING_MIN_MULT:.1f}×, {config.ATR_SIZING_MAX_MULT:.1f}×]")
    print(f"  TP dynamique: TP = entrée + {config.ATR_TP_MULT}×ATR"
          f"  |  minimum garanti {config.ATR_TP_MIN_PCT*100:.0f}%")

    # Chargement données
    all_symbols = list(dict.fromkeys(s for cfg in CONFIGS for s in cfg["symbols"]))
    all_tfs     = list(dict.fromkeys(cfg["timeframe"] for cfg in CONFIGS))

    print(f"\n{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, all_tfs, verbose=True)

    print(f"{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    done, total = 0, len(all_symbols) * len(all_tfs)
    for symbol in all_symbols:
        for tf in all_tfs:
            multi_sim.get_df(symbol, tf)
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)
    print()

    print(f"{Fore.CYAN}Chargement Fear & Greed Index...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # ---------------------------------------------------------------------------
    # Tableau principal
    # ---------------------------------------------------------------------------
    all_rows   = []
    best_combo = {}  # pour le résumé final

    for cfg in CONFIGS:
        tf = cfg["timeframe"]
        n  = multi_sim.CANDLES_PER_YEAR[tf] * cfg["years"]

        dfs: dict = {}
        for symbol in cfg["symbols"]:
            df = multi_sim.get_df(symbol, tf)
            if df is not None and len(df) >= 10:
                dfs[symbol] = df.tail(n).reset_index(drop=True)

        if not dfs:
            continue

        variant_results = {}
        for v in VARIANTS:
            r = multi_sim.sim_multi_on_dfs(
                dfs, use_sl=False, fg=fg_data,
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
                atr_sizing=v["atr_sizing"],
                atr_tp=v["atr_tp"],
            )
            variant_results[v["label"]] = r

        # Ligne de séparation par config
        all_rows.append([f"{Fore.YELLOW}{cfg['label']}{Style.RESET_ALL}",
                         "", "", "", "", "", "", "", ""])

        base = variant_results["Baseline"]
        ret0 = base.get("return_%", 0)
        dd0  = base.get("drawdown_%", 0)
        pf0  = base.get("profit_factor", 0)

        best_ret   = ret0
        best_label = "Baseline"

        for v in VARIANTS:
            r    = variant_results[v["label"]]
            ret  = r.get("return_%", 0)
            dd   = r.get("drawdown_%", 0)
            pf   = r.get("profit_factor", 0)
            wins = r.get("win_%", 0)
            t    = r.get("trades", 0)

            if v["label"] == "Baseline":
                d_ret = "—"
                d_dd  = "—"
            else:
                d_ret = fmt_delta(ret - ret0, positive_is_good=True)
                d_dd  = fmt_delta(dd  - dd0,  positive_is_good=False)

            all_rows.append([
                f"  {v['label']}",
                fmt_ret(ret),
                f"{dd:.1f}%",
                f"{pf:.2f}",
                f"{wins:.1f}%",
                str(t),
                d_ret,
                d_dd,
                "",
            ])

            if ret > best_ret:
                best_ret   = ret
                best_label = v["label"]

        best_combo[cfg["label"]] = {"label": best_label, "ret": best_ret}

    headers = [
        "Config / Variante", "Return", "Drawdown", "PF",
        "Win%", "Trades", "Δ Return", "Δ DD", "",
    ]
    print(f"\n{Fore.YELLOW}Comparaison : Baseline vs variantes ATR (4 ans / multi / sansSL){Style.RESET_ALL}")
    print(tabulate(all_rows, headers=headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Résumé
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Meilleure variante par config :{Style.RESET_ALL}")
    sum_headers = ["Config", "Meilleure variante", "Return"]
    sum_rows    = []
    for cfg_label, best in best_combo.items():
        color = Fore.GREEN if best["ret"] > 0 else Fore.RED
        sum_rows.append([cfg_label, best["label"], f"{color}{best['ret']:+.1f}%{Style.RESET_ALL}"])
    print(tabulate(sum_rows, headers=sum_headers, tablefmt="rounded_outline"))

    # ---------------------------------------------------------------------------
    # Analyse du sizing ATR (visualisation)
    # ---------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}Distribution du sizing ATR sur BTC/USDT 12h (4 ans) :{Style.RESET_ALL}")
    df_btc = multi_sim.get_df("BTC/USDT", "12h")
    if df_btc is not None and "atr" in df_btc.columns:
        n  = multi_sim.CANDLES_PER_YEAR["12h"] * 4
        sl = df_btc.tail(n).copy()
        sl = sl.dropna(subset=["atr", "close"])
        sl["atr_pct"] = sl["atr"] / sl["close"] * 100
        sl["scale"]   = (config.ATR_SIZING_REF_PCT / (sl["atr_pct"] / 100)).clip(
            config.ATR_SIZING_MIN_MULT, config.ATR_SIZING_MAX_MULT
        )
        sl["pos_pct"] = sl["scale"] * config.POSITION_SIZE_PCT * 100

        atr_headers = ["Métrique", "Valeur"]
        atr_rows = [
            ["ATR/prix moyen (BTC 12h, 4 ans)", f"{sl['atr_pct'].mean():.2f}%"],
            ["ATR/prix médian",                  f"{sl['atr_pct'].median():.2f}%"],
            ["ATR/prix min",                     f"{sl['atr_pct'].min():.2f}%"],
            ["ATR/prix max",                     f"{sl['atr_pct'].max():.2f}%"],
            ["Taille position moyenne",           f"{sl['pos_pct'].mean():.1f}% du capital"],
            ["Taille position min (volatile)",    f"{sl['pos_pct'].min():.1f}% du capital"],
            ["Taille position max (calme)",       f"{sl['pos_pct'].max():.1f}% du capital"],
            ["% temps au plancher (×0.40)",       f"{(sl['scale'] <= config.ATR_SIZING_MIN_MULT + 0.01).mean()*100:.1f}%"],
            ["% temps au plafond (×2.00)",        f"{(sl['scale'] >= config.ATR_SIZING_MAX_MULT - 0.01).mean()*100:.1f}%"],
        ]
        print(tabulate(atr_rows, headers=atr_headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print("  Sizing ATR  : réduit les pertes en marché volatile → ↓ drawdown, parfois ↓ return")
    print("  TP ATR      : TP adapté à l'amplitude réelle → ↑ win rate, trades plus longs")
    print("  +Les deux   : combinaison — peut améliorer le ratio return/drawdown (Sharpe)")
    print()
    print("  Si Δ Return < 0 : le TP fixe était bien calibré pour ce portfolio")
    print("  Si Δ DD < 0     : le sizing ATR protège en période de forte volatilité")


if __name__ == "__main__":
    main()
