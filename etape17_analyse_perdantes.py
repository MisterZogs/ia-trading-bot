"""
Étape 17 — Analyse des années perdantes (2018 et 2024)

5 axes d'analyse :
  1. Contribution par symbole — qui perd, qui sauve
  2. Découpage trimestriel — quand ça perd
  3. Fréquence des signaux BUY — trop de trades en bear market ?
  4. Corrélation avec BTC — sensibilité au marché baissier
  5. Quelle config résiste le mieux — stratégies × SL sur années perdantes
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

SYMBOLS    = config.SYMBOLS
TIMEFRAME  = "12h"
BAD_YEARS  = [2018, 2024]
GOOD_YEARS = [2020, 2021, 2023]

STRATEGIES = [
    ("épurée",    False, False),
    ("baseline",  False, True),
    ("+TripleST", True,  True),
    ("épurée+ST", True,  False),
]

def fmt(v, width=6):
    if v is None:
        return "—"
    color = Fore.GREEN if v > 0 else (Fore.RED if v < 0 else "")
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"

def load_year(year):
    dfs = {}
    for sym in SYMBOLS:
        df = ms.get_df_for_year(sym, TIMEFRAME, year)
        if df is not None:
            dfs[sym] = df
    return dfs

def load_quarter(year, quarter):
    """Charge les DFs filtrés sur un trimestre (Q1=mois 1-3, etc.)."""
    month_start = (quarter - 1) * 3 + 1
    month_end   = quarter * 3
    dfs = {}
    for sym in SYMBOLS:
        df = ms.get_df_for_year(sym, TIMEFRAME, year)
        if df is None:
            continue
        mask = df["timestamp"].dt.month.between(month_start, month_end)
        sliced = df[mask].reset_index(drop=True)
        if len(sliced) >= 10:
            dfs[sym] = sliced
    return dfs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  ÉTAPE 17 — Analyse des années perdantes (2018 & 2024)")
    print(f"{'='*90}{Style.RESET_ALL}")

    # ---- Chargement données ------------------------------------------------
    print(f"\n{Fore.CYAN}Chargement données...{Style.RESET_ALL}")
    data_cache.prefetch_all(SYMBOLS, [TIMEFRAME], verbose=False)
    data_cache.prefetch_all_8y(SYMBOLS, [TIMEFRAME], verbose=False)
    data_cache.prefetch_all_10y(SYMBOLS, [TIMEFRAME], verbose=False)
    for sym in SYMBOLS:
        ms.get_df(sym, TIMEFRAME)
        ms.get_df_8y(sym, TIMEFRAME)
        ms.get_df_10y(sym, TIMEFRAME)

    fg_data = fear_greed.load(verbose=False)
    print(f"  OK — Fear & Greed : {len(fg_data)} jours")

    # =========================================================================
    # AXE 1 — Contribution par symbole
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  AXE 1 — Contribution par symbole sur les années perdantes")
    print(f"{'='*90}{Style.RESET_ALL}")

    sym_results = {}  # {sym: {year: return_%}}
    for year in BAD_YEARS + GOOD_YEARS:
        for sym in SYMBOLS:
            df_year = ms.get_df_for_year(sym, TIMEFRAME, year)
            if df_year is None:
                continue
            r = ms.sim_multi_on_dfs(
                {sym: df_year}, use_sl=False, fg=fg_data,
                use_triple_st=False, use_sma_macd=False, tf=TIMEFRAME,
            )
            ret = r.get("return_%")
            if ret is not None:
                sym_results.setdefault(sym, {})[year] = ret

    ax1_headers = ["Symbole", "2018", "2024", "Moy perdantes", "2020", "2021", "2023", "Moy bonnes", "Δ (bon-mauvais)"]
    ax1_rows = []
    for sym in SYMBOLS:
        data = sym_results.get(sym, {})
        bad_vals  = [data[y] for y in BAD_YEARS  if y in data]
        good_vals = [data[y] for y in GOOD_YEARS if y in data]
        moy_bad   = sum(bad_vals)  / len(bad_vals)  if bad_vals  else None
        moy_good  = sum(good_vals) / len(good_vals) if good_vals else None
        delta     = round(moy_good - moy_bad, 1) if moy_bad is not None and moy_good is not None else None
        ax1_rows.append((sym, data, moy_bad, moy_good, delta))

    # Trier par moy_bad (les pires en premier)
    ax1_rows.sort(key=lambda x: x[2] if x[2] is not None else 999)

    table_ax1 = []
    for sym, data, moy_bad, moy_good, delta in ax1_rows:
        delta_str = "—"
        if delta is not None:
            color = Fore.GREEN if delta > 0 else Fore.RED
            delta_str = f"{color}{delta:+.1f}%{Style.RESET_ALL}"
        table_ax1.append([
            sym.replace("/USDT", ""),
            fmt(data.get(2018)), fmt(data.get(2024)),
            fmt(moy_bad),
            fmt(data.get(2020)), fmt(data.get(2021)), fmt(data.get(2023)),
            fmt(moy_good),
            delta_str,
        ])
    print(tabulate(table_ax1, headers=ax1_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # AXE 2 — Découpage trimestriel
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  AXE 2 — Performance trimestrielle (portfolio complet, config épurée)")
    print(f"{'='*90}{Style.RESET_ALL}")

    ax2_headers = ["Année", "Q1 (jan-mar)", "Q2 (avr-jun)", "Q3 (jul-sep)", "Q4 (oct-déc)", "Total"]
    ax2_rows = []

    for year in BAD_YEARS + GOOD_YEARS:
        row = [str(year)]
        for q in [1, 2, 3, 4]:
            dfs_q = load_quarter(year, q)
            if not dfs_q:
                row.append("—")
                continue
            r = ms.sim_multi_on_dfs(
                dfs_q, use_sl=False, fg=fg_data,
                use_triple_st=False, use_sma_macd=False, tf=TIMEFRAME,
            )
            row.append(fmt(r.get("return_%")))
        # Total année
        dfs_y = load_year(year)
        r_tot = ms.sim_multi_on_dfs(
            dfs_y, use_sl=False, fg=fg_data,
            use_triple_st=False, use_sma_macd=False, tf=TIMEFRAME,
        )
        row.append(fmt(r_tot.get("return_%")))
        ax2_rows.append(row)

    print(tabulate(ax2_rows, headers=ax2_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # AXE 3 — Fréquence des signaux BUY
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  AXE 3 — Fréquence des signaux BUY (bonnes vs mauvaises années)")
    print(f"{'='*90}{Style.RESET_ALL}")

    ax3_headers = ["Année", "Type", "Total bougies", "Signaux BUY", "Ratio BUY%", "Signaux SELL", "Ratio SELL%"]
    ax3_rows = []

    for year in BAD_YEARS + GOOD_YEARS:
        total_candles = 0
        total_buy = 0
        total_sell = 0
        for sym in SYMBOLS:
            df_year = ms.get_df_for_year(sym, TIMEFRAME, year)
            if df_year is None:
                continue
            sigs = indicators.vectorized_signals(df_year, use_triple_st=False, use_sma_macd=False)
            total_candles += len(sigs)
            total_buy  += (sigs == "BUY").sum()
            total_sell += (sigs == "SELL").sum()
        if total_candles == 0:
            continue
        typ = f"{Fore.RED}Perdante{Style.RESET_ALL}" if year in BAD_YEARS else f"{Fore.GREEN}Bonne{Style.RESET_ALL}"
        ratio_buy  = total_buy  / total_candles * 100
        ratio_sell = total_sell / total_candles * 100
        color_buy  = Fore.RED if ratio_buy > 5 else (Fore.GREEN if ratio_buy < 3 else "")
        ax3_rows.append([
            str(year), typ,
            f"{total_candles:,}",
            f"{total_buy:,}",
            f"{color_buy}{ratio_buy:.1f}%{Style.RESET_ALL}",
            f"{total_sell:,}",
            f"{ratio_sell:.1f}%",
        ])

    print(tabulate(ax3_rows, headers=ax3_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # AXE 4 — Corrélation avec BTC
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  AXE 4 — Corrélation avec BTC par trimestre (années perdantes)")
    print(f"{'='*90}{Style.RESET_ALL}")

    ax4_headers = ["Année", "Trimestre", "BTC début", "BTC fin", "BTC %", "Stratégie %", "Corrélé ?"]
    ax4_rows = []

    for year in BAD_YEARS:
        df_btc_full = ms.get_df_for_year("BTC/USDT", TIMEFRAME, year)
        for q in [1, 2, 3, 4]:
            month_start = (q - 1) * 3 + 1
            month_end   = q * 3
            # BTC performance sur le trimestre
            btc_ret = None
            if df_btc_full is not None:
                mask   = df_btc_full["timestamp"].dt.month.between(month_start, month_end)
                df_btc = df_btc_full[mask]
                if len(df_btc) > 0:
                    p_start = df_btc["close"].iloc[0]
                    p_end   = df_btc["close"].iloc[-1]
                    btc_ret = round((p_end - p_start) / p_start * 100, 1)
                    btc_start_str = f"${p_start:,.0f}"
                    btc_end_str   = f"${p_end:,.0f}"
                else:
                    btc_start_str = btc_end_str = "—"
            else:
                btc_start_str = btc_end_str = "—"

            # Stratégie sur le trimestre
            dfs_q = load_quarter(year, q)
            strat_ret = None
            if dfs_q:
                r = ms.sim_multi_on_dfs(
                    dfs_q, use_sl=False, fg=fg_data,
                    use_triple_st=False, use_sma_macd=False, tf=TIMEFRAME,
                )
                strat_ret = r.get("return_%")

            corr = "—"
            if btc_ret is not None and strat_ret is not None:
                if btc_ret < -10 and strat_ret < 0:
                    corr = f"{Fore.RED}OUI — bear market{Style.RESET_ALL}"
                elif btc_ret > 10 and strat_ret > 0:
                    corr = f"{Fore.GREEN}OUI — bull market{Style.RESET_ALL}"
                elif btc_ret < -10 and strat_ret >= 0:
                    corr = f"{Fore.GREEN}Résiste au bear{Style.RESET_ALL}"
                elif btc_ret >= 0 and strat_ret < 0:
                    corr = f"{Fore.YELLOW}Perd malgré hausse{Style.RESET_ALL}"
                else:
                    corr = "Neutre"

            ax4_rows.append([
                str(year), f"Q{q}",
                btc_start_str, btc_end_str,
                fmt(btc_ret),
                fmt(strat_ret),
                corr,
            ])

    print(tabulate(ax4_rows, headers=ax4_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # AXE 5 — Quelle config résiste le mieux
    # =========================================================================
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  AXE 5 — Comparaison des configs sur les années perdantes")
    print(f"{'='*90}{Style.RESET_ALL}")

    ax5_headers = ["Config", "SL", "2018", "2024", "Moy perdantes", "2020", "2021", "2023", "Moy bonnes"]
    ax5_rows = []

    for strat_name, use_ts, use_sm in STRATEGIES:
        for sl_label, use_sl in [("sansSL", False), ("avecSL", True)]:
            row = [strat_name, sl_label]
            bad_vals  = []
            good_vals = []
            for year in BAD_YEARS + GOOD_YEARS:
                dfs = load_year(year)
                if not dfs:
                    row.append("—")
                    continue
                r = ms.sim_multi_on_dfs(
                    dfs, use_sl=use_sl, fg=fg_data,
                    use_triple_st=use_ts, use_sma_macd=use_sm, tf=TIMEFRAME,
                )
                ret = r.get("return_%")
                row.append(fmt(ret))
                if year in BAD_YEARS and ret is not None:
                    bad_vals.append(ret)
                elif year in GOOD_YEARS and ret is not None:
                    good_vals.append(ret)

            # Insérer moyennes aux bonnes positions
            moy_bad  = sum(bad_vals)  / len(bad_vals)  if bad_vals  else None
            moy_good = sum(good_vals) / len(good_vals) if good_vals else None
            # row est [strat, sl, 2018, 2024, 2020, 2021, 2023] → insérer moy_bad après 2024
            row.insert(4, fmt(moy_bad))
            row.append(fmt(moy_good))
            ax5_rows.append(row)

    # Trier par moy perdantes (extraire la valeur numérique)
    def moy_bad_val(row):
        # La moy perdantes est à l'index 4, mais contient des codes ANSI
        for strat_name, use_ts, use_sm in STRATEGIES:
            for sl_label, use_sl in [("sansSL", False), ("avecSL", True)]:
                pass
        return 0

    print(tabulate(ax5_rows, headers=ax5_headers, tablefmt="rounded_outline"))

    # =========================================================================
    # SYNTHÈSE
    # =========================================================================
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  SYNTHÈSE — Pourquoi 2018 et 2024 perdent ?")
    print(f"{'='*90}{Style.RESET_ALL}")

    # Compter les symboles perdants vs gagnants par année
    for year in BAD_YEARS:
        perdants = [(s, sym_results[s][year]) for s in SYMBOLS
                    if s in sym_results and year in sym_results[s] and sym_results[s][year] < 0]
        gagnants = [(s, sym_results[s][year]) for s in SYMBOLS
                    if s in sym_results and year in sym_results[s] and sym_results[s][year] > 0]
        perdants.sort(key=lambda x: x[1])
        gagnants.sort(key=lambda x: x[1], reverse=True)

        print(f"\n  {Fore.RED}▶ {year}{Style.RESET_ALL} — {len(perdants)} symboles perdants / {len(gagnants)} gagnants")
        if perdants:
            worst = ", ".join(f"{s.replace('/USDT','')} ({v:+.1f}%)" for s, v in perdants[:5])
            print(f"    Pires : {Fore.RED}{worst}{Style.RESET_ALL}")
        if gagnants:
            best = ", ".join(f"{s.replace('/USDT','')} ({v:+.1f}%)" for s, v in gagnants[:5])
            print(f"    Meilleurs : {Fore.GREEN}{best}{Style.RESET_ALL}")
    print()
