"""
Étape 5 — Ablation Study des conditions BUY/SELL

Pour chaque condition, on la retire et on mesure l'impact sur les résultats.
Si retirer une condition améliore les résultats → elle était nuisible (faux signaux).
Si retirer une condition dégrade les résultats → elle est utile.

On teste aussi chaque condition seule (score minimum = 1) pour voir leur valeur isolée.

Config de référence : les meilleures configs identifiées
  - Top20 / multi / sansSL / 12h / 4ans  → best alpha vs B&H
  - Top5  / multi / sansSL / 12h / 4ans  → best return absolu
  - BTC+ETH / multi / sansSL / 12h / 4ans
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators
import data_cache
import fear_greed
import multi_sim as ms

init(autoreset=True)

# ---------------------------------------------------------------------------
# Configs de référence
# ---------------------------------------------------------------------------
CONFIGS = {
    "Top20 / 12h / 4ans":  (config.SYMBOLS,                                          "12h", 4),
    "Top5  / 12h / 4ans":  (["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"], "12h", 4),
    "BTC+ETH / 12h / 4ans":(["BTC/USDT","ETH/USDT"],                                  "12h", 4),
    "Top20 / 12h / 3ans":  (config.SYMBOLS,                                           "12h", 3),
    "Top5  / 6h / 3ans":   (["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"], "6h",  3),
}

# Noms des 8 conditions (index dans vectorized_signals)
CONDITIONS = [
    "b1 — Prix en baisse >2%",
    "b2 — RSI < 30",
    "b3 — SMA20 < SMA50",
    "b4 — MACD < Signal",
    "b5 — Stoch < 20",
    "b6 — Prix < BB basse",
    "b7 — Volume > Vol_SMA",
    "b8 — Triple ST dip",
]


# ---------------------------------------------------------------------------
# Version patchée de vectorized_signals avec masque de conditions
# ---------------------------------------------------------------------------
def vectorized_signals_masked(df: pd.DataFrame, mask_buy: list[bool],
                               mask_sell: list[bool], min_score: int = None) -> pd.Series:
    """
    Comme vectorized_signals mais on peut désactiver des conditions individuellement.
    mask_buy[i] = False → condition i désactivée côté BUY
    mask_sell[i] = False → condition i désactivée côté SELL
    """
    close = df["close"]
    prev  = close.shift(1)
    chg   = (close - prev) / prev * 100

    b = [None] * 8
    b[0] = chg < -config.PRICE_CHANGE_THRESHOLD_PCT
    b[1] = df["rsi"].notna() & (df["rsi"] < config.RSI_OVERSOLD)
    b[2] = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] < df["sma_slow"])
    b[3] = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] < df["macd_signal"])
    b[4] = df["stoch_k"].notna() & (df["stoch_k"] < config.STOCH_OVERSOLD)
    b[5] = df["bb_lower"].notna() & (close < df["bb_lower"])
    b[6] = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])
    b[7] = (df["st_dir_7"].fillna(0) == -1) & (df["st_dir_21"].fillna(0) == 1)

    s = [None] * 8
    s[0] = chg > config.PRICE_CHANGE_THRESHOLD_PCT
    s[1] = df["rsi"].notna() & (df["rsi"] > config.RSI_OVERBOUGHT)
    s[2] = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] > df["sma_slow"])
    s[3] = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] > df["macd_signal"])
    s[4] = df["stoch_k"].notna() & (df["stoch_k"] > config.STOCH_OVERBOUGHT)
    s[5] = df["bb_upper"].notna() & (close > df["bb_upper"])
    s[6] = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])
    s[7] = (df["st_dir_7"].fillna(0) == 1)

    zero = pd.Series(False, index=df.index)
    buy_score  = sum((b[i] if mask_buy[i]  else zero).astype(int) for i in range(8))
    sell_score = sum((s[i] if mask_sell[i] else zero).astype(int) for i in range(8))

    in_uptrend = df["sma200"].isna() | (close > df["sma200"])
    ms_val = min_score if min_score is not None else config.MIN_SCORE_TO_TRADE

    is_buy  = in_uptrend & (buy_score >= ms_val) & (buy_score > sell_score)
    is_sell = (sell_score >= ms_val) & (sell_score > buy_score)

    signals = pd.Series("HOLD", index=df.index, dtype=object)
    signals[is_buy]  = "BUY"
    signals[is_sell] = "SELL"
    signals[is_buy & is_sell] = "BUY"
    return signals


def sim_masked(syms, tf, years, mask_buy, mask_sell, fg, min_score=None):
    """Simulation multi/sansSL avec masque de conditions."""
    import math
    n = ms.CANDLES_PER_YEAR[tf] * years
    dfs  = {}
    sigs = {}
    for sym in syms:
        df = ms.get_df(sym, tf)
        if df is not None and len(df) >= 10:
            sliced = df.tail(n).reset_index(drop=True)
            dfs[sym]  = sliced
            sigs[sym] = vectorized_signals_masked(sliced, mask_buy, mask_sell, min_score).values
    if not dfs:
        return {}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    min_slot  = ms.INITIAL_CAPITAL * ms.POSITION_SIZE_PCT
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = ms.INITIAL_CAPITAL
    positions: dict = {s: [] for s in dfs}
    trades    = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for sym in dfs:
            if i >= arr_len[sym]:
                continue
            low   = arr_low[sym][i]
            high  = arr_high[sym][i]
            close = arr_close[sym][i]

            still_open = []
            for pos in positions[sym]:
                tp_hit = high >= pos["tp"]
                if tp_hit:
                    fee_exit = pos["tp"] * pos["size"] * ms.FEE_RATE
                    pnl      = (pos["tp"] - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                else:
                    still_open.append(pos)
            positions[sym] = still_open

            sig = sigs[sym][i]
            if sig == "BUY" and fg_val is not None and fg_val > fear_greed.FG_GREED_VETO:
                sig = "HOLD"

            deployed  = sum(p["entry"] * p["size"] for plist in positions.values() for p in plist)
            available = capital - deployed

            if sig == "BUY" and available >= min_slot:
                risk    = risks[sym]
                pos_val = min_slot
                capital -= pos_val * ms.FEE_RATE
                size    = pos_val / close
                positions[sym].append({
                    "entry": close, "size": size,
                    "sl": 0.0,
                    "tp": close * (1 + risk["tp"]),
                })
            elif sig == "SELL" and positions[sym]:
                for pos in positions[sym]:
                    fee_exit = close * pos["size"] * ms.FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                positions[sym] = []

        equity.append(capital)

    for sym, plist in positions.items():
        last = arr_close[sym][-1]
        for pos in plist:
            fee  = last * pos["size"] * ms.FEE_RATE
            pnl  = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)

    return ms._stats(trades, capital, equity)


def bah(syms, tf, years):
    n = ms.CANDLES_PER_YEAR[tf] * years
    returns = []
    for sym in syms:
        df = ms.get_df(sym, tf)
        if df is None or len(df) < 10:
            continue
        dp = df.tail(n)
        s, e = dp["close"].iloc[0], dp["close"].iloc[-1]
        if s > 0:
            returns.append((e - s) / s * 100)
    return round(sum(returns) / len(returns), 1) if returns else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*100}")
    print("  ÉTAPE 5 — Ablation Study : impact de chaque condition BUY/SELL")
    print(f"{'='*100}{Style.RESET_ALL}")

    # Chargement
    all_syms = list(dict.fromkeys(s for syms, _, _ in CONFIGS.values() for s in syms))
    all_tfs  = list(dict.fromkeys(tf for _, tf, _ in CONFIGS.values()))
    print(f"\n{Fore.CYAN}Chargement données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_syms, all_tfs, verbose=False)
    for sym in all_syms:
        for tf in all_tfs:
            ms.get_df(sym, tf)
    print(f"  Indicateurs calculés ({len(all_syms)} symboles × {len(all_tfs)} TF)")

    print(f"{Fore.CYAN}Chargement Fear & Greed...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # -----------------------------------------------------------------------
    # Test 1 — Ablation : retirer une condition à la fois
    # -----------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*100}")
    print("  TEST 1 — Ablation : impact du retrait de chaque condition")
    print(f"{'='*100}{Style.RESET_ALL}")
    print("  (référence = toutes les 8 conditions actives, MIN_SCORE=3)\n")

    for cfg_name, (syms, tf, years) in CONFIGS.items():
        bah_val = bah(syms, tf, years)

        # Référence : toutes conditions actives
        ref = sim_masked(syms, tf, years,
                         [True]*8, [True]*8, fg_data)
        ref_ret = ref.get("return_%", 0.0)

        rows = []
        for i, cond_name in enumerate(CONDITIONS):
            mask_b = [True]*8
            mask_s = [True]*8
            mask_b[i] = False
            mask_s[i] = False
            r = sim_masked(syms, tf, years, mask_b, mask_s, fg_data)
            ret = r.get("return_%", 0.0)
            delta = round(ret - ref_ret, 1)
            verdict = (f"{Fore.GREEN}+++ AMÉLIORE (+{delta:.1f}pts){Style.RESET_ALL}"
                       if delta > 2 else
                       f"{Fore.RED}--- DÉGRADE ({delta:.1f}pts){Style.RESET_ALL}"
                       if delta < -2 else
                       f"  ~ neutre ({delta:+.1f}pts)")
            rows.append([
                cond_name,
                f"{ret:+.1f}%",
                f"{r.get('drawdown_%', 0):+.1f}%",
                f"{r.get('win_%', 0):.0f}%",
                f"{r.get('trades', 0)}",
                verdict,
            ])

        headers = ["Condition retirée", "Return", "Max DD", "Win%", "Trades", "Verdict"]
        print(f"{Fore.YELLOW}  {cfg_name}  |  Référence: {ref_ret:+.1f}%  |  B&H: {bah_val:+.1f}%{Style.RESET_ALL}")
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print()

    # -----------------------------------------------------------------------
    # Test 2 — Chaque condition seule (MIN_SCORE=1)
    # -----------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*100}")
    print("  TEST 2 — Chaque condition seule (MIN_SCORE=1)")
    print(f"{'='*100}{Style.RESET_ALL}")
    print("  Montre la valeur intrinsèque de chaque signal pris isolément\n")

    for cfg_name, (syms, tf, years) in CONFIGS.items():
        bah_val = bah(syms, tf, years)
        rows = []
        for i, cond_name in enumerate(CONDITIONS):
            mask_b = [False]*8
            mask_s = [False]*8
            mask_b[i] = True
            mask_s[i] = True
            r = sim_masked(syms, tf, years, mask_b, mask_s, fg_data, min_score=1)
            ret = r.get("return_%", 0.0)
            alpha = round(ret - bah_val, 1)
            color = Fore.GREEN if ret > 5 else (Fore.RED if ret < -5 else "")
            rows.append([
                cond_name,
                f"{color}{ret:+.1f}%{Style.RESET_ALL}",
                f"{r.get('drawdown_%', 0):+.1f}%",
                f"{r.get('win_%', 0):.0f}%",
                f"{r.get('trades', 0)}",
                f"{r.get('profit_factor', 0):.2f}",
                f"{alpha:+.1f}%",
            ])
        # Trier par return décroissant
        rows.sort(key=lambda x: float(x[1].replace("%","").replace("+","")
                  .replace(f"{Fore.GREEN}","").replace(f"{Fore.RED}","")
                  .replace(f"{Style.RESET_ALL}","").strip()), reverse=True)

        headers = ["Condition seule", "Return", "Max DD", "Win%", "Trades", "PF", "Alpha"]
        print(f"{Fore.YELLOW}  {cfg_name}  |  B&H: {bah_val:+.1f}%{Style.RESET_ALL}")
        print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
        print()

    # -----------------------------------------------------------------------
    # Test 3 — Combinaisons optimales (greedy forward selection)
    # -----------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*100}")
    print("  TEST 3 — Sélection greedy : meilleure combinaison de conditions")
    print(f"{'='*100}{Style.RESET_ALL}")
    print("  On part de 0 et on ajoute la condition qui améliore le plus, une par une\n")

    cfg_name = "Top20 / 12h / 4ans"
    syms, tf, years = CONFIGS[cfg_name]
    bah_val = bah(syms, tf, years)

    selected   = []
    remaining  = list(range(8))
    best_ret_so_far = -999

    print(f"  Config : {cfg_name}  |  B&H: {bah_val:+.1f}%\n")
    step_rows = []

    for step in range(8):
        best_cond, best_ret, best_r = None, -999, {}
        for idx in remaining:
            test = selected + [idx]
            mask_b = [i in test for i in range(8)]
            mask_s = [i in test for i in range(8)]
            r = sim_masked(syms, tf, years, mask_b, mask_s, fg_data)
            ret = r.get("return_%", -999)
            if ret > best_ret:
                best_ret, best_cond, best_r = ret, idx, r
        if best_cond is None:
            break
        selected.append(best_cond)
        remaining.remove(best_cond)
        delta = round(best_ret - best_ret_so_far, 1) if best_ret_so_far > -999 else 0
        best_ret_so_far = best_ret
        step_rows.append([
            f"Étape {step+1}",
            CONDITIONS[best_cond],
            f"{best_ret:+.1f}%",
            f"{delta:+.1f}pts" if step > 0 else "—",
            f"{best_r.get('drawdown_%',0):+.1f}%",
            f"{best_r.get('win_%',0):.0f}%",
            f"{best_r.get('trades',0)}",
            f"{best_r.get('profit_factor',0):.2f}",
        ])

    headers = ["Étape", "Condition ajoutée", "Return cumulé", "Delta", "Max DD", "Win%", "Trades", "PF"]
    print(tabulate(step_rows, headers=headers, tablefmt="rounded_outline"))
    print(f"\n  Ordre optimal : {' → '.join(CONDITIONS[i].split('—')[0].strip() for i in selected)}")
    print(f"  Meilleure combinaison greedy : {best_ret_so_far:+.1f}%  (B&H: {bah_val:+.1f}%)")
