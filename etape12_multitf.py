"""
Confirmation Multi-Timeframe
=============================
Un signal BUY sur 12h n'est exécuté que si le timeframe inférieur (4h)
est aussi en signal BUY à ce moment-là.

Logique :
  12h BUY + 4h BUY  → on entre (double confirmation)
  12h BUY + 4h HOLD → on ignore (pas de confirmation)
  12h BUY + 4h SELL → on ignore (signal contraire)

Attentes :
  ↓ nombre de trades (on filtre les BUY non confirmés)
  ↑ win rate (moins de faux signaux)
  return incertain (on manque des bons trades aussi)

Usage :
    python3 etape12_multitf.py
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import data_cache
import indicators
import multi_sim
import fear_greed

init(autoreset=True)

CONFIRM_TF = "4h"   # timeframe de confirmation

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
]


# ---------------------------------------------------------------------------
# Confirmation multi-TF : filtre les BUY du primaire par le secondaire
# ---------------------------------------------------------------------------
def get_confirmed_sigs(symbol: str, primary_df: pd.DataFrame,
                       primary_sigs: "np.ndarray",
                       confirm_tf: str,
                       use_triple_st: bool, use_sma_macd: bool) -> "np.ndarray":
    """
    Pour chaque candle du primary_df, conserve le BUY seulement si
    le dernier signal du confirm_tf clos avant ce candle est aussi BUY.
    Utilise merge_asof pour l'alignement temporel.
    """
    import numpy as np

    df_sec = multi_sim.get_df(symbol, confirm_tf)
    if df_sec is None or "timestamp" not in df_sec.columns \
            or "timestamp" not in primary_df.columns:
        return primary_sigs   # pas de données 4h → pas de filtre

    sec_sigs = indicators.vectorized_signals(
        df_sec, use_triple_st=use_triple_st, use_sma_macd=use_sma_macd
    )

    df_sec_sig = pd.DataFrame({
        "timestamp": pd.to_datetime(df_sec["timestamp"]),
        "sig_conf":  sec_sigs.values,
    }).sort_values("timestamp")

    df_prim = pd.DataFrame({
        "timestamp": pd.to_datetime(primary_df["timestamp"]),
        "sig_prim":  primary_sigs,
    }).sort_values("timestamp")

    merged = pd.merge_asof(
        df_prim, df_sec_sig,
        on="timestamp",
        direction="backward",   # dernier signal 4h connu avant ce candle 12h
    )

    confirmed = merged["sig_prim"].copy()
    # BUY 12h bloqué si le 4h n'est pas aussi BUY
    mask = (merged["sig_prim"] == "BUY") & (merged["sig_conf"] != "BUY")
    confirmed[mask] = "HOLD"

    return confirmed.values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"  CONFIRMATION MULTI-TIMEFRAME — 12h confirmé par {CONFIRM_TF}")
    print(f"{'='*80}{Style.RESET_ALL}")

    all_symbols = list(dict.fromkeys(s for cfg in CONFIGS for s in cfg["symbols"]))
    all_tfs     = list(dict.fromkeys(
        tf for cfg in CONFIGS for tf in [cfg["timeframe"], CONFIRM_TF]
    ))

    print(f"\n{Fore.CYAN}Chargement des données ({', '.join(all_tfs)})...{Style.RESET_ALL}")
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

    headers = [
        "Config / Variante",
        "Return", "Drawdown", "PF", "Win%", "Trades",
        "Δ Return", "Δ DD", "Δ Trades",
    ]
    rows = []

    for cfg in CONFIGS:
        tf = cfg["timeframe"]
        n  = multi_sim.CANDLES_PER_YEAR[tf] * cfg["years"]

        # DataFrames slicés
        dfs: dict = {}
        for sym in cfg["symbols"]:
            df = multi_sim.get_df(sym, tf)
            if df is not None and len(df) >= 10:
                dfs[sym] = df.tail(n).reset_index(drop=True)
        if not dfs:
            continue

        # Signaux baseline (12h seul)
        base_sigs = {
            s: indicators.vectorized_signals(
                dfs[s],
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
            ).values
            for s in dfs
        }

        # Signaux confirmés (12h + 4h)
        conf_sigs = {
            s: get_confirmed_sigs(
                s, dfs[s], base_sigs[s],
                CONFIRM_TF,
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
            )
            for s in dfs
        }

        # Statistique : combien de BUY filtrés ?
        import numpy as np
        total_buy   = sum((base_sigs[s] == "BUY").sum() for s in base_sigs)
        total_conf  = sum((conf_sigs[s] == "BUY").sum() for s in conf_sigs)
        pct_kept    = total_conf / total_buy * 100 if total_buy else 0

        # Run baseline
        r_base = multi_sim.sim_multi_on_dfs(
            dfs, use_sl=False, fg=fg_data,
            precomputed_sigs=base_sigs,
        )

        # Run avec confirmation 4h
        r_conf = multi_sim.sim_multi_on_dfs(
            dfs, use_sl=False, fg=fg_data,
            precomputed_sigs=conf_sigs,
        )

        def fmt_ret(v):
            c = Fore.GREEN if v > 0 else Fore.RED
            return f"{c}{v:+.1f}%{Style.RESET_ALL}"

        def fmt_delta(v, good_positive=True):
            if abs(v) < 0.5:
                return f"{v:+.1f}"
            c = (Fore.GREEN if v > 0 else Fore.RED) if good_positive \
                else (Fore.GREEN if v < 0 else Fore.RED)
            return f"{c}{v:+.1f}{Style.RESET_ALL}"

        # Séparateur de config
        rows.append([
            f"{Fore.YELLOW}{cfg['label']}{Style.RESET_ALL}"
            f"  ({total_buy} BUY → {total_conf} confirmés = {pct_kept:.0f}% gardés)",
            "", "", "", "", "", "", "", "",
        ])

        for label, r in [("Baseline (12h seul)", r_base),
                          (f"+ Confirmation {CONFIRM_TF}", r_conf)]:
            ret  = r.get("return_%", 0)
            dd   = r.get("drawdown_%", 0)
            pf   = r.get("profit_factor", 0)
            win  = r.get("win_%", 0)
            t    = r.get("trades", 0)

            if label.startswith("Baseline"):
                rows.append([
                    f"  {label}",
                    fmt_ret(ret), f"{dd:.1f}%", f"{pf:.2f}",
                    f"{win:.1f}%", str(t), "—", "—", "—",
                ])
                ret0, dd0, t0 = ret, dd, t
            else:
                rows.append([
                    f"  {label}",
                    fmt_ret(ret), f"{dd:.1f}%", f"{pf:.2f}",
                    f"{win:.1f}%", str(t),
                    fmt_delta(ret - ret0, good_positive=True),
                    fmt_delta(dd  - dd0,  good_positive=False),
                    fmt_delta(t   - t0,   good_positive=True),
                ])

    print(f"\n{Fore.YELLOW}Comparaison Baseline vs Confirmation {CONFIRM_TF} (4 ans / multi / sansSL){Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print(f"  • Δ Return > 0 → la confirmation 4h filtre les mauvais BUY → gain net")
    print(f"  • Δ Return < 0 → on filtre trop de bons BUY → perte nette")
    print(f"  • Δ DD < 0     → drawdown réduit (moins de trades risqués)")
    print(f"  • % BUY gardés → si < 30%, le filtre est trop restrictif")
    print(f"  • Si Δ Return > 0 ET Δ DD < 0 : confirmer ce timeframe en production")


if __name__ == "__main__":
    main()
