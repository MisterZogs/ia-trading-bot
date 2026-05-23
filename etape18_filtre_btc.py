"""
Étape 18 — Filtre BTC momentum

Problème identifié en étape 17 :
  2024 Q4 — BTC +46.8% mais stratégie -9.3%.
  En bull run fort, BTC entraîne tous les altcoins vers le haut.
  La mean-reversion ouvre des shorts implicites sur des actifs en plein élan → perd.

Solution testée :
  Bloquer les BUY quand BTC est en fort bull run,
  similaire au veto Fear & Greed.

Méthode de détection du bull run BTC :
  btc_momentum = (close_btc - SMA_N_btc) / SMA_N_btc × 100
  Si btc_momentum > seuil → VETO BUY (sig = "HOLD")

Paramètres testés :
  SMA windows  : 20, 50 bougies (12h)
  Seuils       : +5%, +10%, +15%, +20%, +25%, +30% au-dessus de la SMA
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import fear_greed
import indicators
import multi_sim as ms

init(autoreset=True)

SYMBOLS   = config.SYMBOLS
TIMEFRAME = "12h"

# Seuils et fenêtres à tester
SMA_WINDOWS = [20, 50]
THRESHOLDS  = [5, 10, 15, 20, 25, 30]   # % au-dessus de la SMA BTC

# Config de référence : Top 20 / épurée / sansSL (notre meilleure config)
USE_TRIPLE_ST = False
USE_SMA_MACD  = False


# ---------------------------------------------------------------------------
# Construction du signal BTC momentum
# ---------------------------------------------------------------------------
def build_btc_momentum(df_btc: pd.DataFrame, sma_window: int) -> dict:
    """
    Retourne un dict {date -> momentum%} où momentum = (close - SMA) / SMA * 100.
    Utilisé comme le dict Fear & Greed : {date -> valeur}.
    """
    close = df_btc["close"].values
    sma   = np.full(len(close), np.nan)
    for i in range(sma_window - 1, len(close)):
        sma[i] = close[i - sma_window + 1 : i + 1].mean()

    momentum = {}
    if "timestamp" not in df_btc.columns:
        return momentum
    for i, row in df_btc.iterrows():
        if np.isnan(sma[i]):
            continue
        if sma[i] == 0:
            continue
        date = row["timestamp"].date()
        momentum[date] = (close[i] - sma[i]) / sma[i] * 100
    return momentum


# ---------------------------------------------------------------------------
# Simulation avec filtre BTC momentum
# ---------------------------------------------------------------------------
def sim_with_btc_filter(dfs: dict, btc_momentum: dict, threshold: float,
                        fg: dict | None, tf: str) -> dict:
    """
    sim_multi_on_dfs avec veto BUY supplémentaire quand BTC momentum > threshold.
    Reproduit la logique de sim_multi_on_dfs en ajoutant le filtre BTC.
    """
    import math
    if not dfs:
        return {}

    sigs = {s: indicators.vectorized_signals(
        dfs[s], use_triple_st=USE_TRIPLE_ST, use_sma_macd=USE_SMA_MACD).values
        for s in dfs}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT,
                                               "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values \
                if "timestamp" in dfs[ref_sym].columns else None

    capital   = ms.INITIAL_CAPITAL
    positions: dict[str, list] = {s: [] for s in dfs}
    trades, durations, equity  = [], [], [capital]
    max_len = max(arr_len.values())

    for i in range(2, max_len):
        fg_val  = None
        btc_val = None
        if arr_dates is not None and i < len(arr_dates):
            d = arr_dates[i]
            if fg is not None:
                fg_val = fg.get(d)
            btc_val = btc_momentum.get(d)

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            risk  = risks[symbol]

            still_open = []
            for pos in positions[symbol]:
                if high >= pos["tp"]:
                    fee = pos["tp"] * pos["size"] * ms.FEE_RATE
                    pnl = (pos["tp"] - pos["entry"]) * pos["size"] - fee
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                else:
                    still_open.append(pos)
            positions[symbol] = still_open

            sig = sigs[symbol][i]

            # Veto Fear & Greed
            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            # Veto BTC momentum
            if sig == "BUY" and btc_val is not None:
                if btc_val > threshold:
                    sig = "HOLD"

            pos_val  = capital * config.POSITION_SIZE_PCT
            deployed = sum(p["entry"] * p["size"]
                           for plist in positions.values() for p in plist)

            if sig == "BUY" and capital - deployed >= pos_val:
                capital -= pos_val * ms.FEE_RATE
                size     = pos_val / close
                positions[symbol].append({
                    "entry":   close, "size": size, "sl": 0.0,
                    "tp":      close * (1 + risk["tp"]),
                    "entry_i": i,
                })
            elif sig == "SELL" and positions[symbol]:
                for pos in positions[symbol]:
                    fee = close * pos["size"] * ms.FEE_RATE
                    pnl = (close - pos["entry"]) * pos["size"] - fee
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                positions[symbol] = []

        equity.append(capital)

    unclosed = sum(len(pl) for pl in positions.values())
    for symbol, plist in positions.items():
        last = arr_close[symbol][-1]
        for pos in plist:
            fee = last * pos["size"] * ms.FEE_RATE
            pnl = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)
            durations.append(max_len - 1 - pos["entry_i"])

    return ms._stats(trades, capital, equity, durations, unclosed, tf)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt(v):
    if v is None:
        return "—"
    color = Fore.GREEN if v > 0 else (Fore.RED if v < 0 else "")
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"


def load_period(symbols, tf, years):
    n = ms.CANDLES_PER_YEAR[tf] * years
    dfs = {}
    for sym in symbols:
        df = ms.get_df(sym, tf)
        if df is not None and len(df) >= 10:
            dfs[sym] = df.tail(n).reset_index(drop=True)
    return dfs


def load_year(symbols, tf, year):
    dfs = {}
    for sym in symbols:
        df = ms.get_df_for_year(sym, tf, year)
        if df is not None:
            dfs[sym] = df
    return dfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  ÉTAPE 18 — Filtre BTC Momentum")
    print(f"  Veto BUY quand BTC > SMA_N + seuil%")
    print(f"{'='*90}{Style.RESET_ALL}")

    # ---- Chargement données ------------------------------------------------
    print(f"\n{Fore.CYAN}Chargement données...{Style.RESET_ALL}")
    all_syms = list(dict.fromkeys(["BTC/USDT"] + SYMBOLS))
    data_cache.prefetch_all(all_syms, [TIMEFRAME], verbose=False)
    for sym in all_syms:
        ms.get_df(sym, TIMEFRAME)
    # Pour les années calendaires anciennes
    data_cache.prefetch_all_8y(all_syms, [TIMEFRAME], verbose=False)
    data_cache.prefetch_all_10y(all_syms, [TIMEFRAME], verbose=False)
    for sym in all_syms:
        ms.get_df_8y(sym, TIMEFRAME)
        ms.get_df_10y(sym, TIMEFRAME)

    fg_data = fear_greed.load(verbose=False)
    print(f"  OK — Fear & Greed : {len(fg_data)} jours")

    # ---- Signal BTC momentum -----------------------------------------------
    print(f"{Fore.CYAN}Calcul du momentum BTC...{Style.RESET_ALL}")
    df_btc_full = ms.get_df("BTC/USDT", TIMEFRAME)
    btc_signals = {}   # {(sma_window, threshold): {date: momentum}}
    for w in SMA_WINDOWS:
        mom = build_btc_momentum(df_btc_full, sma_window=w)
        btc_signals[w] = mom
        # Stats rapides
        vals = list(mom.values())
        pct_above = {t: sum(1 for v in vals if v > t) / len(vals) * 100
                     for t in THRESHOLDS}
        print(f"  SMA{w:2d} — % jours en bull run : " +
              "  ".join(f">{t}%: {pct_above[t]:.0f}%" for t in THRESHOLDS))

    # ---- Référence sans filtre BTC -----------------------------------------
    print(f"\n{Fore.CYAN}Simulation de référence (sans filtre BTC)...{Style.RESET_ALL}")
    dfs_4y = load_period(SYMBOLS, TIMEFRAME, 4)
    r_ref  = ms.sim_multi_on_dfs(dfs_4y, use_sl=False, fg=fg_data,
                                  use_triple_st=False, use_sma_macd=False,
                                  tf=TIMEFRAME)
    ref_ret = r_ref.get("return_%", 0)
    ref_dd  = r_ref.get("drawdown_%", 0)
    print(f"  Référence 4 ans : {fmt(ref_ret)} | DD {ref_dd:.1f}%")

    # =========================================================================
    # Tableau 1 — Comparaison seuils × SMA (4 ans)
    # =========================================================================
    print(f"\n{Fore.CYAN}Simulations (4 ans) × {len(SMA_WINDOWS)} SMA × {len(THRESHOLDS)} seuils...{Style.RESET_ALL}")

    results_grid = {}   # {(w, t): stats}
    total = len(SMA_WINDOWS) * len(THRESHOLDS)
    done  = 0
    for w in SMA_WINDOWS:
        for t in THRESHOLDS:
            r = sim_with_btc_filter(dfs_4y, btc_signals[w], t, fg_data, TIMEFRAME)
            results_grid[(w, t)] = r
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)
    print()

    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  IMPACT SUR 4 ANS — Return% / Drawdown%")
    print(f"  Référence sans filtre : {fmt(ref_ret)} | DD {ref_dd:.1f}%")
    print(f"{'='*90}{Style.RESET_ALL}")

    grid_headers = ["SMA \\ Seuil"] + [f">{t}%" for t in THRESHOLDS]
    for w in SMA_WINDOWS:
        rows_ret = [f"SMA{w} Ret%"]
        rows_dd  = [f"SMA{w} DD%"]
        for t in THRESHOLDS:
            r   = results_grid[(w, t)]
            ret = r.get("return_%")
            dd  = r.get("drawdown_%", 0)
            delta = round(ret - ref_ret, 1) if ret is not None else None
            arrow = "▲" if delta and delta > 0 else ("▼" if delta and delta < 0 else "")
            color = Fore.GREEN if delta and delta > 0 else (Fore.RED if delta and delta < 0 else "")
            rows_ret.append(f"{fmt(ret)} ({color}{delta:+.1f}{Style.RESET_ALL}{arrow})")
            dd_delta = round(dd - ref_dd, 1)
            dd_color = Fore.GREEN if dd_delta < 0 else (Fore.RED if dd_delta > 0 else "")
            rows_dd.append(f"{dd:.1f}% ({dd_color}{dd_delta:+.1f}{Style.RESET_ALL})")
        print(tabulate([rows_ret, rows_dd], headers=grid_headers, tablefmt="rounded_outline"))
        print()

    # =========================================================================
    # Tableau 2 — Meilleure config par année calendaire
    # =========================================================================
    import datetime
    cal_years = list(range(2018, datetime.date.today().year + 1))

    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  PAR ANNÉE CALENDAIRE — Meilleure config filtre vs référence")
    print(f"{'='*90}{Style.RESET_ALL}")

    # Trouver la meilleure combo (w, t) selon la somme des returns sur 4 ans
    best_w, best_t = max(results_grid, key=lambda k: results_grid[k].get("return_%") or -999)
    best_ret_4y    = results_grid[(best_w, best_t)].get("return_%")
    print(f"  Meilleure config : SMA{best_w} / seuil >{best_t}% → {fmt(best_ret_4y)} sur 4 ans\n")

    # Précalcul du momentum BTC par année
    btc_mom_best = btc_signals[best_w]

    year_headers = ["Année", "Référence", f"Filtre SMA{best_w}>{best_t}%", "Delta", "% bougies vetoed"]
    year_rows    = []

    for year in cal_years:
        dfs_y = load_year(SYMBOLS, TIMEFRAME, year)
        if not dfs_y:
            continue

        # Référence
        r_ref_y = ms.sim_multi_on_dfs(dfs_y, use_sl=False, fg=fg_data,
                                       use_triple_st=False, use_sma_macd=False,
                                       tf=TIMEFRAME)
        # Avec filtre BTC
        r_filt_y = sim_with_btc_filter(dfs_y, btc_mom_best, best_t, fg_data, TIMEFRAME)

        ret_ref  = r_ref_y.get("return_%")
        ret_filt = r_filt_y.get("return_%")
        delta    = round(ret_filt - ret_ref, 1) if ret_ref is not None and ret_filt is not None else None

        # % de bougies où le filtre BTC était actif
        df_btc_y = ms.get_df_for_year("BTC/USDT", TIMEFRAME, year)
        pct_veto = "—"
        if df_btc_y is not None and "timestamp" in df_btc_y.columns:
            dates_y   = df_btc_y["timestamp"].dt.date
            n_veto    = sum(1 for d in dates_y if btc_mom_best.get(d, 0) > best_t)
            pct_veto  = f"{n_veto / len(dates_y) * 100:.0f}%"

        delta_str = "—"
        if delta is not None:
            color     = Fore.GREEN if delta > 0 else (Fore.RED if delta < 0 else "")
            delta_str = f"{color}{delta:+.1f}%{Style.RESET_ALL}"

        year_rows.append([str(year), fmt(ret_ref), fmt(ret_filt), delta_str, pct_veto])

    print(tabulate(year_rows, headers=year_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # Résumé
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  RÉSUMÉ — Verdict filtre BTC")
    print(f"{'='*90}{Style.RESET_ALL}")

    sum_headers = ["Config", "4 ans", "DD%", "Trades", "Win%", "vs référence"]
    sum_rows    = [["Référence (sans filtre)",
                    fmt(r_ref.get("return_%")),
                    f"{r_ref.get('drawdown_%', 0):.1f}%",
                    r_ref.get("trades", "—"),
                    f"{r_ref.get('win_%', 0):.1f}%",
                    "—"]]

    for w in SMA_WINDOWS:
        for t in THRESHOLDS:
            r   = results_grid[(w, t)]
            ret = r.get("return_%", 0)
            dd  = r.get("drawdown_%", 0)
            delta = round(ret - ref_ret, 1)
            color = Fore.GREEN if delta > 1 else (Fore.RED if delta < -1 else "")
            sum_rows.append([
                f"SMA{w} / >{t}%",
                fmt(ret),
                f"{dd:.1f}%",
                r.get("trades", "—"),
                f"{r.get('win_%', 0):.1f}%",
                f"{color}{delta:+.1f}%{Style.RESET_ALL}",
            ])

    print(tabulate(sum_rows, headers=sum_headers, tablefmt="rounded_outline"))
    print()
