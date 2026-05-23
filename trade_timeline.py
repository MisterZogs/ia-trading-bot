"""
Génère un graphe de distribution temporelle des trades pour chacune des
42 stratégies commentées dans full_ranking_results.csv.

Sortie : trade_timeline.pdf  (42 pages, une par stratégie)

Usage : python trade_timeline.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

import config
import indicators
import fear_greed
import data_cache
import multi_sim as ms
from multi_sim import CANDLES_PER_YEAR, FEE_RATE, INITIAL_CAPITAL

GAP_DAYS = 60  # zone jaune si aucun trade pendant X jours

CAPITAL_MAP = {
    "5%/illim":  (0.05, 9999),
    "10%/max10": (0.10,   10),
    "20%/max5":  (0.20,    5),
    "50%/max2":  (0.50,    2),
    "100%/max1": (1.00,    1),
}

PORTFOLIO_SYMBOLS = {
    "Top 20":  config.SYMBOLS,
    "Top 10":  ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT",
                "ADA/USDT","AVAX/USDT","DOT/USDT","LINK/USDT","MATIC/USDT"],
    "Top 5":   ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"],
    "BTC+ETH": ["BTC/USDT","ETH/USDT"],
    "BTC":     ["BTC/USDT"],
    "ETH":     ["ETH/USDT"],
}

STRAT_MAP = {
    "baseline":  (False, True),
    "+TripleST": (True,  True),
    "épurée":    (False, False),
    "épurée+ST": (True,  False),
}


def _sim_timeline(symbols, tf, years, mode, use_sl,
                  use_triple_st, use_sma_macd,
                  pos_pct, max_trades, fg_data):
    """
    Rejoue la simulation et retourne une liste de trade events avec timestamps.
    Chaque event : {exit_ts, entry_ts, pnl, symbol, reason}
    """
    n = CANDLES_PER_YEAR[tf] * years
    dfs = {}
    sigs = {}
    for sym in symbols:
        df = ms.get_df(sym, tf)
        if df is not None and len(df) >= 10:
            sliced = df.tail(n).reset_index(drop=True)
            dfs[sym]  = sliced
            sigs[sym] = indicators.vectorized_signals(
                sliced, use_triple_st=use_triple_st,
                use_sma_macd=use_sma_macd).values
    if not dfs:
        return []

    arr_close = {s: dfs[s]["close"].values    for s in dfs}
    arr_low   = {s: dfs[s]["low"].values      for s in dfs}
    arr_high  = {s: dfs[s]["high"].values     for s in dfs}
    arr_ts    = {s: dfs[s]["timestamp"].values for s in dfs}
    arr_len   = {s: len(dfs[s])               for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT,
                                                "tp": config.TAKE_PROFIT_PCT})
                 for s in dfs}

    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values

    capital      = INITIAL_CAPITAL
    positions    = {s: [] for s in dfs}
    trade_events = []
    max_len      = max(arr_len.values())

    for i in range(2, max_len):
        fg_val     = fg_data.get(arr_dates[i]) if i < len(arr_dates) else None
        total_open = sum(len(p) for p in positions.values())

        for sym in dfs:
            if i >= arr_len[sym]:
                continue
            low   = arr_low[sym][i]
            high  = arr_high[sym][i]
            close = arr_close[sym][i]
            risk  = risks[sym]
            ts_i  = pd.Timestamp(arr_ts[sym][i])

            # Vérification TP / SL
            still_open = []
            for pos in positions[sym]:
                tp_hit = high >= pos["tp"]
                sl_hit = use_sl and low <= pos["sl"]
                if tp_hit or sl_hit:
                    exit_px  = pos["tp"] if tp_hit else pos["sl"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trade_events.append({
                        "exit_ts":  ts_i,
                        "entry_ts": pos["entry_ts"],
                        "pnl":      pnl,
                        "symbol":   sym,
                        "reason":   "tp" if tp_hit else "sl",
                    })
                    total_open -= 1
                else:
                    still_open.append(pos)
            positions[sym] = still_open

            sig = sigs[sym][i]
            if sig == "BUY" and fg_val is not None and fg_val > fear_greed.FG_GREED_VETO:
                sig = "HOLD"

            pos_val  = capital * pos_pct
            can_buy  = sig == "BUY" and total_open < max_trades and pos_val > 0 and close > 0
            if mode == "single":
                can_buy = can_buy and not positions[sym]

            if can_buy:
                capital -= pos_val * FEE_RATE
                size       = pos_val / close
                initial_sl = close * (1 - risk["sl"]) if use_sl else 0.0
                positions[sym].append({
                    "entry":    close,
                    "size":     size,
                    "sl":       initial_sl,
                    "tp":       close * (1 + risk["tp"]),
                    "entry_ts": ts_i,
                })
                total_open += 1

            elif sig == "SELL" and positions[sym]:
                for pos in positions[sym]:
                    fee_exit = close * pos["size"] * FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trade_events.append({
                        "exit_ts":  ts_i,
                        "entry_ts": pos["entry_ts"],
                        "pnl":      pnl,
                        "symbol":   sym,
                        "reason":   "signal",
                    })
                    total_open -= 1
                positions[sym] = []

    # Clôture finale
    for sym, plist in positions.items():
        if not plist:
            continue
        last_close = arr_close[sym][-1]
        last_ts    = pd.Timestamp(arr_ts[sym][-1])
        for pos in plist:
            fee = last_close * pos["size"] * FEE_RATE
            pnl = (last_close - pos["entry"]) * pos["size"] - fee
            trade_events.append({
                "exit_ts":  last_ts,
                "entry_ts": pos["entry_ts"],
                "pnl":      pnl,
                "symbol":   sym,
                "reason":   "end",
            })

    return trade_events


def _plot_strategy(fig, ax1, ax2, events, row):
    """Dessine les 2 sous-graphes pour une stratégie."""
    if not events:
        ax1.text(0.5, 0.5, "Aucun trade", transform=ax1.transAxes,
                 ha="center", va="center", fontsize=12, color="gray")
        return

    df = pd.DataFrame(events)
    df["month"] = df["exit_ts"].dt.to_period("M")
    df["win"]   = df["pnl"] > 0

    # ── Barres mensuelles (wins verts / losses rouges) ─────────────────────
    monthly_wins   = df[df["win"]].groupby("month").size()
    monthly_losses = df[~df["win"]].groupby("month").size()
    all_months     = sorted(set(monthly_wins.index) | set(monthly_losses.index))
    xs        = [m.to_timestamp() for m in all_months]
    wins_y    = [int(monthly_wins.get(m, 0))   for m in all_months]
    losses_y  = [int(monthly_losses.get(m, 0)) for m in all_months]

    ax1.bar(xs, wins_y,   width=20, color="#27ae60", alpha=0.85, label="Win")
    ax1.bar(xs, losses_y, width=20, color="#e74c3c", alpha=0.85,
            bottom=wins_y, label="Loss")

    # Zones de gap > GAP_DAYS
    sorted_dates = sorted(df["exit_ts"])
    for j in range(1, len(sorted_dates)):
        gap = (sorted_dates[j] - sorted_dates[j - 1]).days
        if gap > GAP_DAYS:
            ax1.axvspan(sorted_dates[j - 1], sorted_dates[j],
                        color="gold", alpha=0.18)
            mid = sorted_dates[j - 1] + (sorted_dates[j] - sorted_dates[j - 1]) / 2
            ax1.text(mid, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 1,
                     f"{gap}j", ha="center", va="top", fontsize=6, color="darkorange")

    ax1.set_ylabel("Trades / mois", fontsize=8)
    ax1.legend(loc="upper left", fontsize=7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax1.grid(axis="y", alpha=0.3)

    # ── Courbe cumulative + PnL coloré ─────────────────────────────────────
    df_s = df.sort_values("exit_ts").copy()
    df_s["cumul"] = range(1, len(df_s) + 1)

    # Courbe générale
    ax2.step(df_s["exit_ts"], df_s["cumul"],
             where="post", color="#2980b9", linewidth=1.5, zorder=2)

    # Points colorés win/loss
    ax2.scatter(df_s.loc[df_s["win"], "exit_ts"],
                df_s.loc[df_s["win"], "cumul"],
                color="#27ae60", s=8, zorder=3, alpha=0.7)
    ax2.scatter(df_s.loc[~df_s["win"], "exit_ts"],
                df_s.loc[~df_s["win"], "cumul"],
                color="#e74c3c", s=8, zorder=3, alpha=0.7)

    ax2.set_ylabel("Trades cumulés", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)
    ax2.grid(alpha=0.3)

    # Résumé stats
    n          = len(df)
    wr         = df["win"].mean() * 100
    delta_days = (df["exit_ts"].max() - df["exit_ts"].min()).days
    rate       = n / (delta_days / 30) if delta_days > 0 else 0
    gaps       = sum(1 for j in range(1, len(sorted_dates))
                     if (sorted_dates[j] - sorted_dates[j - 1]).days > GAP_DAYS)
    ax2.text(0.99, 0.04,
             f"{n} trades | WR {wr:.0f}% | ~{rate:.1f}/mois | gaps>{GAP_DAYS}j: {gaps}",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8, color="#555555",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))


def main():
    # ── Chargement des 42 stratégies commentées ────────────────────────────
    df_csv  = pd.read_csv("full_ranking_results.csv", low_memory=False)
    strats  = df_csv[
        df_csv["commentaire"].notna() &
        (df_csv["commentaire"].astype(str).str.strip() != "")
    ].copy()
    print(f"{len(strats)} stratégies avec commentaire.\n")

    # ── Chargement des données (cache disque → mémoire) ────────────────────
    all_syms   = list(dict.fromkeys(s for syms in PORTFOLIO_SYMBOLS.values() for s in syms))
    needed_tfs = sorted(strats["timeframe"].unique())
    print(f"Chargement données : {len(all_syms)} symboles × {needed_tfs}...")
    data_cache.prefetch_all(all_syms, needed_tfs, verbose=False)
    total = len(all_syms) * len(needed_tfs)
    done  = 0
    for sym in all_syms:
        for tf in needed_tfs:
            ms.get_df(sym, tf)
            done += 1
            print(f"\r  indicateurs {done}/{total}", end="", flush=True)
    print()

    # ── Fear & Greed ───────────────────────────────────────────────────────
    fg_data = fear_greed.load(verbose=False)

    # ── Génération PDF ─────────────────────────────────────────────────────
    output = "trade_timeline.pdf"
    print(f"\nGénération {len(strats)} graphes → {output}\n")

    with PdfPages(output) as pdf:
        for idx, (_, row) in enumerate(strats.iterrows(), 1):
            portfolio = str(row["portfolio"])
            mode_raw  = str(row["mode"])
            sl_raw    = str(row["sl"])
            tf        = str(row["timeframe"])
            strat     = str(row["stratégie"])
            cap_label = str(row["capital"]) if "capital" in row and pd.notna(row.get("capital")) else "5%/illim"

            mode            = "single" if mode_raw == "1pos" else "multi"
            use_sl          = sl_raw == "avecSL"
            symbols         = PORTFOLIO_SYMBOLS.get(portfolio, config.SYMBOLS)
            use_triple_st, use_sma_macd = STRAT_MAP.get(strat, (False, False))
            pos_pct, max_trades         = CAPITAL_MAP.get(cap_label, (0.05, 9999))

            print(f"  [{idx:2d}/{len(strats)}] {portfolio} | {mode_raw} | {sl_raw} | "
                  f"{tf} | {strat} | {cap_label}...", end=" ", flush=True)

            events = _sim_timeline(symbols, tf, 4, mode, use_sl,
                                   use_triple_st, use_sma_macd,
                                   pos_pct, max_trades, fg_data)
            print(f"{len(events)} trades")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

            comment = str(row.get("commentaire", ""))[:90]
            ret     = row.get("return_%", "?")
            dd      = row.get("drawdown_%", "?")
            wr_csv  = row.get("win_%", "?")
            title   = (f"[{idx}/{len(strats)}]  {portfolio} | {mode_raw} | {sl_raw} | "
                       f"TF: {tf} | {strat} | cap: {cap_label}\n"
                       f"Return: {ret}% | MaxDD: {dd}% | Win: {wr_csv}%   —   {comment}")
            fig.suptitle(title, fontsize=9, fontweight="bold", wrap=True)

            _plot_strategy(fig, ax1, ax2, events, row)

            plt.tight_layout(rect=[0, 0, 1, 0.91])
            pdf.savefig(fig, dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"\nPDF généré : {output}  ({len(strats)} pages)")


if __name__ == "__main__":
    main()
