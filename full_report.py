"""
Rapport complet :
- Toutes les timeframes (30m, 1h, 2h, 4h, 6h, 12h, 1d) par crypto
- Toutes les périodes (1 an, 2 ans, 3 ans, 4 ans)
- SL/TP optimisés par symbole (depuis config.py)
- Données téléchargées une seule fois puis découpées par période
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators
import data_cache

init(autoreset=True)

ALL_TIMEFRAMES = ["30m", "1h", "2h", "4h", "6h", "12h", "1d"]

# Bougies pour découper les périodes depuis la fin
CANDLES_PER_YEAR = {
    "30m": 17520, "1h": 8760, "2h": 4380,
    "4h": 2190,   "6h": 1460, "12h": 730,  "1d": 365,
}

PERIODS = {1: "1 an", 2: "2 ans", 3: "3 ans", 4: "4 ans"}

# Frais de trading : 0.1% à l'achat + 0.1% à la vente
FEE_RATE = 0.001

# Cache données enrichies par (symbol, timeframe)
_raw_cache: dict[str, pd.DataFrame] = {}


def fetch_and_cache(symbol: str, timeframe: str) -> pd.DataFrame:
    key = f"{symbol}_{timeframe}"
    if key in _raw_cache:
        return _raw_cache[key]

    raw = data_cache.fetch_ohlcv(symbol, timeframe)
    df = raw.reset_index()
    df = indicators.compute_all(df)
    df = df.dropna().reset_index(drop=True)
    _raw_cache[key] = df
    return df


def simulate(df_full: pd.DataFrame, symbol: str, years: int, timeframe: str) -> dict:
    """
    Simule sur les N dernières années de données.
    - Frais : FEE_RATE à l'achat (sur la valeur de position) + FEE_RATE à la vente
    - Volatility targeting : position réduite quand la volatilité (ATR/close) est haute
    """
    n_candles = CANDLES_PER_YEAR[timeframe] * years
    df = df_full.tail(n_candles).reset_index(drop=True)
    if len(df) < 10:
        return {}

    risk_params = config.SYMBOL_RISK.get(symbol, {
        "sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT
    })
    sl = risk_params["sl"]
    tp = risk_params["tp"]
    capital = 1000.0
    position = None
    trades = []
    equity = [capital]

    # Precompute signals vectorisés pour toute la période
    signals = indicators.vectorized_signals(df).values  # numpy array

    for i in range(2, len(df)):
        candle = df.iloc[i]
        low, high, close = candle["low"], candle["high"], candle["close"]

        if position:
            sl_hit = low  <= position["sl"]
            tp_hit = high >= position["tp"]
            if sl_hit or tp_hit:
                exit_price = position["sl"] if sl_hit else position["tp"]
                fee_exit   = exit_price * position["size"] * FEE_RATE
                pnl        = (exit_price - position["entry"]) * position["size"] - fee_exit
                capital   += pnl
                trades.append(pnl)
                position   = None
                equity.append(capital)
                continue

        sig = signals[i]

        if sig == "BUY" and position is None:
            # Volatility targeting : réduit la position si ATR/close > seuil
            atr   = candle.get("atr", float("nan"))
            vol   = (atr / close) if (close > 0 and pd.notna(atr)) else config.POSITION_SIZE_PCT
            # Cible : risque ATR = 2% du capital max par trade
            # size_pct adaptatif : entre 2% et POSITION_SIZE_PCT (5%)
            target_risk_pct = 0.02
            vol_size_pct = min(config.POSITION_SIZE_PCT,
                               target_risk_pct / vol if vol > 0 else config.POSITION_SIZE_PCT)

            fee_buy  = capital * vol_size_pct * FEE_RATE
            capital -= fee_buy
            size     = (capital * vol_size_pct) / close
            position = {"entry": close, "size": size,
                        "sl": close * (1 - sl), "tp": close * (1 + tp)}

        elif sig == "SELL" and position is not None:
            fee_exit = close * position["size"] * FEE_RATE
            pnl      = (close - position["entry"]) * position["size"] - fee_exit
            capital += pnl
            trades.append(pnl)
            position = None

        equity.append(capital)

    if position:
        last     = df.iloc[-1]["close"]
        fee_exit = last * position["size"] * FEE_RATE
        pnl      = (last - position["entry"]) * position["size"] - fee_exit
        capital += pnl
        trades.append(pnl)

    if not trades:
        return {"trades": 0, "return_%": 0.0, "win_%": 0.0,
                "profit_factor": 0.0, "drawdown_%": 0.0}

    wins   = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")
    ret    = (capital - 1000.0) / 1000.0 * 100

    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        peak = max(peak, v)
        dd = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)

    return {
        "trades": len(trades),
        "win_%": round(len(wins) / len(trades) * 100, 1),
        "return_%": round(ret, 2),
        "profit_factor": round(pf, 2),
        "drawdown_%": round(max_dd, 2),
    }


def portfolio_return(results_by_symbol: dict) -> float:
    """Somme des gains USDT de tous les symboles sur capital commun de 1000 USDT."""
    return sum(r.get("return_%", 0) for r in results_by_symbol.values())


def print_period_table(period_results: dict, years: int):
    """Tableau : lignes = symboles, colonnes = timeframes."""
    label = PERIODS[years]
    print(f"\n{Fore.YELLOW}{'='*100}")
    print(f"  PÉRIODE : {label.upper()} | Capital initial : 1000 USDT | SL/TP optimisés par symbole")
    print(f"{'='*100}{Style.RESET_ALL}")

    # En-têtes : symbol | 30m | 1h | 2h | 4h | 6h | 12h | 1d | TOTAL
    headers = ["Symbole"] + [f"Return {tf}" for tf in ALL_TIMEFRAMES]
    rows = []

    for symbol in config.SYMBOLS:
        row = [symbol]
        for tf in ALL_TIMEFRAMES:
            r = period_results.get((symbol, tf, years), {})
            ret = r.get("return_%", None)
            trades = r.get("trades", 0)
            if ret is None or trades == 0:
                row.append("—")
            else:
                row.append(f"{ret:+.1f}%\n({trades}t)")
        rows.append(row)

    # Ligne total portefeuille par timeframe
    total_row = ["TOTAL PORTF."]
    for tf in ALL_TIMEFRAMES:
        total = sum(
            period_results.get((s, tf, years), {}).get("return_%", 0)
            for s in config.SYMBOLS
        )
        n_pos = sum(
            1 for s in config.SYMBOLS
            if period_results.get((s, tf, years), {}).get("return_%", 0) > 0
        )
        total_row.append(f"{total:+.1f}%\n({n_pos}/20✓)")
    rows.append(total_row)

    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


def print_best_table(all_results: dict):
    """Tableau récapitulatif : meilleure timeframe par symbole par période."""
    print(f"\n{Fore.CYAN}{'='*80}")
    print("  MEILLEURE TIMEFRAME PAR SYMBOLE × PÉRIODE (return %)")
    print(f"{'='*80}{Style.RESET_ALL}")

    headers = ["Symbole", "SL%", "TP%"] + [f"{PERIODS[y]} (best TF)" for y in PERIODS]
    rows = []

    for symbol in config.SYMBOLS:
        risk = config.SYMBOL_RISK.get(symbol, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT})
        row = [symbol, f"{risk['sl']*100:.0f}%", f"{risk['tp']*100:.0f}%"]
        for years in PERIODS:
            best_ret = None
            best_tf  = None
            for tf in ALL_TIMEFRAMES:
                r = all_results.get((symbol, tf, years), {})
                ret = r.get("return_%")
                if ret is not None and (best_ret is None or ret > best_ret):
                    best_ret = ret
                    best_tf  = tf
            if best_ret is not None:
                color = "+" if best_ret >= 0 else ""
                row.append(f"{color}{best_ret:.1f}% [{best_tf}]")
            else:
                row.append("—")
        rows.append(row)

    # Ligne résumé portefeuille (meilleure TF par symbole)
    total_row = ["TOTAL", "", ""]
    for years in PERIODS:
        total = 0
        for symbol in config.SYMBOLS:
            best = max(
                (all_results.get((symbol, tf, years), {}).get("return_%", -999) for tf in ALL_TIMEFRAMES),
                default=0
            )
            total += max(best, 0)  # Ne compte que les positifs
        total_row.append(f"+{total:.1f}% (pos. seuls)")
    rows.append(total_row)

    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


if __name__ == "__main__":
    all_results: dict = {}

    # Téléchargement/cache des données (une fois par symbol/timeframe)
    print(f"\n{Fore.CYAN}Chargement des données 4 ans ({len(config.SYMBOLS)} cryptos × {len(ALL_TIMEFRAMES)} timeframes)...{Style.RESET_ALL}")
    data_cache.prefetch_all(config.SYMBOLS, ALL_TIMEFRAMES)

    print(f"{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    for symbol in config.SYMBOLS:
        for tf in ALL_TIMEFRAMES:
            try:
                fetch_and_cache(symbol, tf)
            except Exception as e:
                print(f"  {Fore.RED}Erreur {symbol}/{tf}: {e}{Style.RESET_ALL}")

    # Simulations : symbol × timeframe × période
    print(f"\n{Fore.CYAN}Simulation en cours...{Style.RESET_ALL}")
    total = len(config.SYMBOLS) * len(ALL_TIMEFRAMES) * len(PERIODS)
    done = 0
    for symbol in config.SYMBOLS:
        for tf in ALL_TIMEFRAMES:
            df = _raw_cache.get(f"{symbol}_{tf}")
            if df is None:
                continue
            for years in PERIODS:
                result = simulate(df, symbol, years, tf)
                all_results[(symbol, tf, years)] = result
                done += 1
            print(f"\r  Progression : {done}/{total}", end="", flush=True)

    print()

    # Affichage des tableaux par période
    for years in PERIODS:
        print_period_table(all_results, years)

    # Tableau récapitulatif meilleure TF par symbole
    print_best_table(all_results)
