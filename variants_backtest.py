"""
Backtest comparatif des variantes proposées après l'analyse du run live
(8 juin → 4 août 2026, PnL réalisé -55.56 USDT sur 636 trades).

Réplique fidèlement le moteur du bot live (bot.py + risk_manager.py) :
  - multi-positions par symbole, pondéré-strict, sansSL
  - TP par symbole depuis config.SYMBOL_RISK
  - sortie TP évaluée sur le CLOSE de la bougie fermée (comme le live),
    pas sur le high intrabar (comme multi_sim.py) — voir TP_ON_CLOSE
  - gate capital de RiskManager.can_buy() reproduit tel quel, bug inclus
    (le gate teste pos_pct global alors que open_position dimensionne
     avec le poids du symbole) — la variante "gate_fix" mesure son impact

Variantes testées : voir VARIANTS en bas de fichier.

Usage:
    python variants_backtest.py                # fenêtre live + années 2023-2025
    python variants_backtest.py --live-only    # fenêtre live seulement
"""

import sys
import copy
import pickle
import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import indicators
import data_cache
import fear_greed
import multi_sim as ms

FEE_RATE        = 0.001
INITIAL_CAPITAL = 1000.0
TIMEFRAME       = "30m"
MAX_POSITIONS   = 10       # LIVE_MAX_POS
FALLBACK_POS_PCT = 0.10    # LIVE_POS_PCT

LIVE_START = pd.Timestamp("2026-06-08 10:00:00")
LIVE_END   = pd.Timestamp("2026-08-04 09:30:00")

SYMBOLS = config.SYMBOLS


# --------------------------------------------------------------------------- #
# Signaux — réplique de indicators.vectorized_signals mais min_score paramétrable
# --------------------------------------------------------------------------- #
def gen_signals(df: pd.DataFrame, min_score: int,
                use_triple_st: bool = True, use_sma_macd: bool = True) -> np.ndarray:
    close = df["close"]
    prev  = close.shift(1)
    chg   = (close - prev) / prev * 100

    b1 = chg < -config.PRICE_CHANGE_THRESHOLD_PCT
    b2 = df["rsi"].notna() & (df["rsi"] < config.RSI_OVERSOLD)
    if use_sma_macd:
        b3 = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] < df["sma_slow"])
        b4 = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] < df["macd_signal"])
    else:
        b3 = b4 = pd.Series(False, index=df.index)
    b5 = df["stoch_k"].notna() & (df["stoch_k"] < config.STOCH_OVERSOLD)
    b6 = df["bb_lower"].notna() & (close < df["bb_lower"])
    b7 = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])
    b8 = ((df["st_dir_7"].fillna(0) == -1) & (df["st_dir_21"].fillna(0) == 1)) \
        if use_triple_st else pd.Series(False, index=df.index)

    buy_score = sum(x.astype(int) for x in (b1, b2, b3, b4, b5, b6, b7, b8))

    s1 = chg > config.PRICE_CHANGE_THRESHOLD_PCT
    s2 = df["rsi"].notna() & (df["rsi"] > config.RSI_OVERBOUGHT)
    if use_sma_macd:
        s3 = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] > df["sma_slow"])
        s4 = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] > df["macd_signal"])
    else:
        s3 = s4 = pd.Series(False, index=df.index)
    s5 = df["stoch_k"].notna() & (df["stoch_k"] > config.STOCH_OVERBOUGHT)
    s6 = df["bb_upper"].notna() & (close > df["bb_upper"])
    s7 = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])
    s8 = (df["st_dir_7"].fillna(0) == 1) if use_triple_st else pd.Series(False, index=df.index)

    sell_score = sum(x.astype(int) for x in (s1, s2, s3, s4, s5, s6, s7, s8))

    in_uptrend = df["sma200"].isna() | (close > df["sma200"])
    # SELL garde le seuil d'origine : on ne durcit que l'entrée
    is_buy  = in_uptrend & (buy_score >= min_score) & (buy_score > sell_score)
    is_sell = (sell_score >= config.MIN_SCORE_TO_TRADE) & (sell_score > buy_score)

    sig = np.full(len(df), "HOLD", dtype=object)
    sig[is_sell.values] = "SELL"
    sig[is_buy.values]  = "BUY"      # BUY prioritaire, comme vectorized_signals
    return sig


# --------------------------------------------------------------------------- #
# Paramètres d'une variante
# --------------------------------------------------------------------------- #
@dataclass
class Params:
    label:            str
    min_score:        int   = 3
    exclude:          tuple = ()
    max_pos_per_sym:  int   = 10_000
    tp_override:      float | None = None    # ex. 0.03 → TP fixe +3%
    trail_atr_mult:   float | None = None    # trailing stop = N × ATR
    fg_fallback:      bool  = False          # veto F&G avec repli sur J-1
    gate_fix:         bool  = False          # can_buy() teste la vraie taille
    tp_on_close:      bool  = True           # fidélité live


# --------------------------------------------------------------------------- #
# Moteur de simulation
# --------------------------------------------------------------------------- #
def simulate(dfs: dict, weights: dict, p: Params, fg: dict | None) -> dict:
    syms = [s for s in dfs if s not in p.exclude]
    if not syms:
        return {}

    sigs   = {s: gen_signals(dfs[s], p.min_score) for s in syms}
    close  = {s: dfs[s]["close"].values for s in syms}
    high   = {s: dfs[s]["high"].values  for s in syms}
    low    = {s: dfs[s]["low"].values   for s in syms}
    atr    = {s: dfs[s]["atr"].values if "atr" in dfs[s].columns else None for s in syms}
    n      = {s: len(dfs[s]) for s in syms}
    risk   = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT,
                                            "tp": config.TAKE_PROFIT_PCT}) for s in syms}

    ref    = max(syms, key=lambda s: n[s])
    dates  = dfs[ref]["timestamp"].dt.date.values

    capital   = INITIAL_CAPITAL
    positions = {s: [] for s in syms}
    trades    = []          # (pnl, reason, symbol, bars_held)
    equity    = [capital]
    fees_paid = 0.0
    refused   = 0
    max_len   = max(n.values())

    def deployed():
        return sum(q["entry"] * q["size"] for pl in positions.values() for q in pl)

    def n_open():
        return sum(len(pl) for pl in positions.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and i < len(dates):
            d = dates[i]
            fg_val = fg.get(d)
            if fg_val is None and p.fg_fallback:
                fg_val = fg.get(d - dt.timedelta(days=1))

        for s in syms:
            if i >= n[s]:
                continue
            px = close[s][i]

            # --- Sorties TP / trailing ---
            still = []
            for q in positions[s]:
                exit_px, reason = None, None
                if p.trail_atr_mult is not None:
                    a = atr[s][i] if atr[s] is not None else np.nan
                    if not np.isnan(a):
                        q["peak"] = max(q["peak"], px)
                        trail = q["peak"] - p.trail_atr_mult * a
                        if px <= trail:
                            exit_px, reason = px, "trail"
                if exit_px is None:
                    hit = (px >= q["tp"]) if p.tp_on_close else (high[s][i] >= q["tp"])
                    if hit:
                        exit_px = px if p.tp_on_close else q["tp"]
                        reason  = "take_profit"
                if exit_px is not None:
                    fee = exit_px * q["size"] * FEE_RATE
                    fees_paid += fee
                    pnl = (exit_px - q["entry"]) * q["size"] - fee
                    capital += pnl
                    trades.append((pnl, reason, s, i - q["i"]))
                else:
                    still.append(q)
            positions[s] = still

            sig = sigs[s][i]
            if sig == "BUY" and fg_val is not None and fg_val > fear_greed.FG_GREED_VETO:
                sig = "HOLD"

            # --- Entrée ---
            if sig == "BUY":
                pct      = weights.get(s) or FALLBACK_POS_PCT
                pos_val  = capital * pct
                gate_val = pos_val if p.gate_fix else capital * FALLBACK_POS_PCT
                ok = (n_open() < MAX_POSITIONS
                      and len(positions[s]) < p.max_pos_per_sym
                      and capital - deployed() >= gate_val)
                if ok:
                    fee = pos_val * FEE_RATE
                    fees_paid += fee
                    capital -= fee
                    tp_pct = p.tp_override if p.tp_override is not None else risk[s]["tp"]
                    positions[s].append({
                        "entry": px, "size": pos_val / px,
                        "tp": px * (1 + tp_pct), "i": i, "peak": px,
                    })
                else:
                    refused += 1

            elif sig == "SELL" and positions[s]:
                for q in positions[s]:
                    fee = px * q["size"] * FEE_RATE
                    fees_paid += fee
                    pnl = (px - q["entry"]) * q["size"] - fee
                    capital += pnl
                    trades.append((pnl, "signal", s, i - q["i"]))
                positions[s] = []

        equity.append(capital)

    # Liquidation des positions ouvertes en fin de période
    n_unclosed = n_open()
    for s, pl in positions.items():
        last = close[s][-1]
        for q in pl:
            fee = last * q["size"] * FEE_RATE
            fees_paid += fee
            pnl = (last - q["entry"]) * q["size"] - fee
            capital += pnl
            trades.append((pnl, "eod", s, max_len - 1 - q["i"]))

    return stats(trades, capital, equity, fees_paid, refused, n_unclosed, p.label)


def stats(trades, capital, equity, fees, refused, unclosed, label):
    pnls = [t[0] for t in trades]
    wins = [x for x in pnls if x > 0]
    loss = [x for x in pnls if x <= 0]
    eq   = np.array(equity)
    peak = np.maximum.accumulate(eq)
    dd   = float(((peak - eq) / peak).max() * 100) if len(eq) else 0.0

    gross_w = sum(wins)
    gross_l = abs(sum(loss))
    by_reason = {}
    for pnl, reason, _, _ in trades:
        by_reason.setdefault(reason, [0, 0.0])
        by_reason[reason][0] += 1
        by_reason[reason][1] += pnl

    return {
        "label":    label,
        "return_%": (capital / INITIAL_CAPITAL - 1) * 100,
        "pnl":      capital - INITIAL_CAPITAL,
        "trades":   len(pnls),
        "win_%":    (len(wins) / len(pnls) * 100) if pnls else 0.0,
        "pf":       (gross_w / gross_l) if gross_l else float("inf"),
        "avg_win":  (gross_w / len(wins)) if wins else 0.0,
        "avg_loss": (-gross_l / len(loss)) if loss else 0.0,
        "be_win_%": (gross_l / len(loss)) / ((gross_w / len(wins)) + (gross_l / len(loss))) * 100
                    if wins and loss else 0.0,
        "max_dd_%": dd,
        "fees":     fees,
        "refused":  refused,
        "unclosed": unclosed,
        "by_reason": by_reason,
    }


# --------------------------------------------------------------------------- #
# Données et poids
# --------------------------------------------------------------------------- #
# Colonnes réellement utilisées par gen_signals / simulate / multi_sim.
# Tout garder fait ~500 Mo par jeu de 20 symboles et envoie la machine en swap.
KEEP_COLS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "rsi", "sma_fast", "sma_slow", "macd", "macd_signal", "stoch_k",
    "bb_lower", "bb_upper", "volume_sma", "st_dir_7", "st_dir_21",
    "sma200", "atr",
]

_DF_CACHE = {}


def _slim(df):
    cols = [c for c in KEEP_COLS if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype == "float64":
            out[c] = out[c].astype("float32")
    return out


def load_dfs(symbols, start=None, end=None, warmup_bars=250):
    """
    DFs avec indicateurs, tronqués sur [start, end] avec warmup pour SMA200.
    Partage le cache de multi_sim (ms._ind_cache) : sans ça les indicateurs sont
    calculés et stockés deux fois (ici + compute_weights), ce qui sature la RAM.
    """
    out = {}
    for s in symbols:
        if s not in _DF_CACHE:
            try:
                raw = data_cache.fetch_ohlcv(s, TIMEFRAME, verbose=False)
                df  = indicators.compute_all(raw.reset_index())
                df  = _slim(df).dropna().reset_index(drop=True)
                _DF_CACHE[s] = df
                ms._ind_cache[f"{s}_{TIMEFRAME}"] = df   # évite le 2e calcul
            except Exception as e:
                print(f"  [err] {s}: {e}")
                _DF_CACHE[s] = None
        df = _DF_CACHE[s]
        if df is None or df.empty:
            continue
        if start is not None:
            i0 = df.index[df["timestamp"] >= start]
            if len(i0) == 0:
                continue
            lo = max(0, int(i0[0]) - warmup_bars)
            df = df.iloc[lo:]
        if end is not None:
            df = df[df["timestamp"] <= end]
        df = df.reset_index(drop=True)
        if len(df) > warmup_bars:
            out[s] = df
    return out


def compute_weights(symbols, fg, max_year):
    """
    Poids pondéré-strict — réplique EXACTE de bot._compute_weights.
    Le bot live utilise SCORE_YEARS = range(2018, 2025), donc années < max_year
    (2025 exclu pour la fenêtre live). Toute déviation change radicalement les
    poids : inclure 2025 (année très négative) fait exploser le poids relatif
    de MATIC, qui n'a plus de données après sept. 2024.
    """
    years = [y for y in range(2018, max_year)]
    scores = {}
    for i_s, s in enumerate(symbols, 1):
        print(f"    [{i_s}/{len(symbols)}] {s}", flush=True)
        rets = []
        for y in years:
            df_y = ms.get_df_for_year(s, TIMEFRAME, y)
            if df_y is None or len(df_y) < 10:
                continue
            r = ms.sim_multi_on_dfs({s: df_y}, use_sl=False, fg=fg,
                                    use_triple_st=True, use_sma_macd=True, tf=TIMEFRAME)
            if r.get("return_%") is not None:
                rets.append(r["return_%"])
        if len(rets) >= 2:
            moy = sum(rets) / len(rets)
            pct_pos = sum(1 for v in rets if v > 0) / len(rets)
            scores[s] = moy * (pct_pos ** 2)
        else:
            scores[s] = 0.0
    pos = {s: max(scores.get(s, 0.0), 0.0) for s in symbols}
    tot = sum(pos.values())
    if tot == 0:
        return {s: 1.0 / len(symbols) for s in symbols}
    return {s: pos[s] / tot for s in symbols}


# --------------------------------------------------------------------------- #
# Variantes
# --------------------------------------------------------------------------- #
VARIANTS = [
    Params("V0  baseline (config live)"),
    Params("V1  score >= 4",                     min_score=4),
    Params("V1b score >= 5",                     min_score=5),
    Params("V2  sans SOL/ALGO/TRX",              exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT")),
    Params("V3  max 2 pos/symbole",              max_pos_per_sym=2),
    Params("V3b max 3 pos/symbole",              max_pos_per_sym=3),
    Params("V4  TP fixe +2%",                    tp_override=0.02),
    Params("V4b TP fixe +3%",                    tp_override=0.03),
    Params("V4c TP fixe +5%",                    tp_override=0.05),
    Params("V4d trailing 2x ATR",                trail_atr_mult=2.0),
    Params("V5  veto F&G reparé (repli J-1)",    fg_fallback=True),
    Params("V6  gate capital corrigé",           gate_fix=True),
]

COMBOS = [
    Params("C1  V1+V2 (score4 + exclusions)",
           min_score=4, exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT")),
    Params("C2  V1+V2+V3b",
           min_score=4, exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT"), max_pos_per_sym=3),
    Params("C3  C2 + TP +3%",
           min_score=4, exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT"),
           max_pos_per_sym=3, tp_override=0.03),
    Params("C4  C2 + trailing 2xATR",
           min_score=4, exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT"),
           max_pos_per_sym=3, trail_atr_mult=2.0),
    Params("C5  C2 + gate corrigé + F&G",
           min_score=4, exclude=("SOL/USDT", "ALGO/USDT", "TRX/USDT"),
           max_pos_per_sym=3, gate_fix=True, fg_fallback=True),
]


HDR = f"{'variante':<34} {'ret%':>7} {'PnL':>8} {'trades':>7} {'win%':>6} {'PF':>5} {'BE%':>6} {'DD%':>6} {'fees':>7}"


def line(r):
    return (f"{r['label']:<34} {r['return_%']:+7.2f} {r['pnl']:+8.1f} {r['trades']:7} "
            f"{r['win_%']:6.1f} {r['pf']:5.2f} {r['be_win_%']:6.1f} {r['max_dd_%']:6.2f} "
            f"{r['fees']:7.1f}")


def run_block(title, dfs, weights, fg, variants):
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    print(HDR)
    print("-" * 100)
    results = []
    for p in variants:
        r = simulate(dfs, weights, p, fg)
        if r:
            results.append(r)
            print(line(r), flush=True)
    return results


def main():
    live_only = "--live-only" in sys.argv
    fg = fear_greed.load(verbose=False)

    print("Chargement des données 30m…", flush=True)
    dfs_all = load_dfs(SYMBOLS)
    for s, d in list(dfs_all.items())[:1]:
        print(f"  {s}: {len(d)} bougies, {d['timestamp'].min()} → {d['timestamp'].max()}")

    # --- Fenêtre live : poids identiques à ceux du bot live (années 2018-2024) ---
    print("\nCalcul des poids pondéré-strict (2018-2024, comme le bot live)…", flush=True)
    w_live = compute_weights(SYMBOLS, fg, max_year=2025)
    top = sorted(w_live.items(), key=lambda x: -x[1])
    print("  poids : " + " | ".join(f"{s.replace('/USDT','')}: {w*100:.1f}%" for s, w in top if w > 0.001))
    print("  live  : SOL: 15.9% | NEAR: 13.2% | ATOM: 12.8% | OP: 10.1% | ARB: 10.0% (log du bot)")

    dfs_live = load_dfs(SYMBOLS, start=LIVE_START, end=LIVE_END)
    all_res = {}
    all_res["live"] = run_block(
        f"FENÊTRE LIVE  {LIVE_START.date()} → {LIVE_END.date()}  (réel bot : -55.56 USDT, 636 trades, 62.7% win)",
        dfs_live, w_live, fg, VARIANTS + COMBOS)

    if live_only:
        return

    # --- Années calendaires : poids recalculés sur années antérieures ---
    for year in (2023, 2024, 2025):
        w = compute_weights(SYMBOLS, fg, max_year=year - 1)
        dfs_y = load_dfs(SYMBOLS,
                         start=pd.Timestamp(f"{year}-01-01"),
                         end=pd.Timestamp(f"{year}-12-31 23:59"))
        if not dfs_y:
            print(f"\n[{year}] pas de données — ignoré")
            continue
        all_res[year] = run_block(f"ANNÉE {year}  (poids calculés sur ≤ {year-1})",
                                  dfs_y, w, fg, VARIANTS + COMBOS)

    # --- Synthèse ---
    print(f"\n{'='*100}\nSYNTHÈSE — rendement % par période\n{'='*100}")
    periods = list(all_res.keys())
    print(f"{'variante':<34}" + "".join(f"{str(p):>12}" for p in periods) + f"{'moy':>10}")
    print("-" * 100)
    labels = [p.label for p in VARIANTS + COMBOS]
    for lab in labels:
        row, vals = f"{lab:<34}", []
        for per in periods:
            m = next((r for r in all_res[per] if r["label"] == lab), None)
            if m:
                row += f"{m['return_%']:+12.2f}"
                vals.append(m["return_%"])
            else:
                row += f"{'—':>12}"
        row += f"{(sum(vals)/len(vals) if vals else 0):+10.2f}"
        print(row)


if __name__ == "__main__":
    main()
