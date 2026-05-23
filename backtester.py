"""
Backtester étendu :
- Toutes les timeframes (30m, 1h, 2h, 4h, 6h, 12h, 1d)
- 4 ans de données (pagination Binance)
- SL/TP déclenchés sur les hauts/bas intracandle (réaliste)
- Mode multi-timeframe : signal valide seulement si TF court ET TF long s'accordent
"""

import time
import ccxt
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators
from risk_manager import RiskManager

init(autoreset=True)

ALL_TIMEFRAMES = ["30m", "1h", "2h", "4h", "6h", "12h", "1d"]

# Nombre de bougies à récupérer par timeframe pour couvrir ~4 ans
CANDLES_PER_TF = {
    "30m": 70080,   # 4 ans
    "1h":  35040,
    "2h":  17520,
    "4h":  8760,
    "6h":  5840,
    "12h": 2920,
    "1d":  1460,
}

# Paires multi-timeframe à tester (court, long)
MULTI_TF_PAIRS = [
    ("1h", "4h"),
    ("1h", "1d"),
    ("2h", "1d"),
    ("4h", "1d"),
    ("6h", "1d"),
]


def fetch_historical(symbol: str, timeframe: str) -> pd.DataFrame:
    """Récupère jusqu'à 2 ans de données en paginant si nécessaire."""
    exchange = ccxt.binance({"enableRateLimit": True})
    target = CANDLES_PER_TF.get(timeframe, 500)
    limit_per_call = 1000
    all_candles = []

    # Calcul du timestamp de départ
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - target * tf_ms

    print(f"  Téléchargement {symbol} {timeframe} ({target} bougies)...", end=" ", flush=True)

    while len(all_candles) < target:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit_per_call)
        except Exception as e:
            print(f"Erreur: {e}")
            break
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + tf_ms
        if len(batch) < limit_per_call:
            break
        time.sleep(exchange.rateLimit / 1000)

    print(f"{len(all_candles)} reçues")

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp").sort_index()
    return df


def _simulate(df_ready: pd.DataFrame, symbol: str, timeframe: str, initial_capital: float) -> dict:
    """
    Simulation sur un DataFrame déjà enrichi avec indicateurs.
    SL/TP déclenchés sur les hauts/bas intracandle (plus réaliste) :
      - SL : déclenché si candle LOW  <= stop_loss_price  → exit au prix SL
      - TP : déclenché si candle HIGH >= take_profit_price → exit au prix TP
      - Si les deux dans la même bougie : SL en priorité (cas le plus conservateur)
    """
    df = df_ready.dropna().reset_index()
    if len(df) < 10:
        return {"symbol": symbol, "timeframe": timeframe, "trades": 0, "note": "Pas assez de données"}

    risk = RiskManager(initial_capital)
    trades      = []
    durations   = []
    entry_index = {}   # symbol -> candle index d'entrée
    entry_ts    = {}   # symbol -> timestamp d'entrée
    equity_curve = [initial_capital]

    for i in range(2, len(df)):
        candle = df.iloc[i]
        candle_low   = candle["low"]
        candle_high  = candle["high"]
        close_price  = candle["close"]
        candle_ts    = candle["timestamp"]

        # Vérification SL/TP sur hauts/bas intracandle
        if symbol in risk.positions:
            pos = risk.positions[symbol][0]
            sl_hit = candle_low  <= pos.stop_loss
            tp_hit = candle_high >= pos.take_profit

            if sl_hit or tp_hit:
                # SL prioritaire si les deux sont touchés dans la même bougie
                exit_price = pos.stop_loss if sl_hit else pos.take_profit
                exit_reason = "stop_loss" if sl_hit else "take_profit"
                result = risk.close_position(symbol, pos, exit_price)
                if result:
                    result["reason"]   = exit_reason
                    result["entry_ts"] = entry_ts.pop(symbol, None)
                    result["exit_ts"]  = candle_ts
                    trades.append(result)
                    if symbol in entry_index:
                        durations.append(i - entry_index.pop(symbol))
                equity_curve.append(risk.total_capital)
                continue

        window = df.iloc[: i + 1].set_index("timestamp")
        score = indicators.score_signal(window)

        if score["signal"] == "BUY":
            opened = risk.open_position(symbol, close_price)
            if opened:
                entry_index[symbol] = i
                entry_ts[symbol]    = candle_ts
        elif score["signal"] == "SELL" and symbol in risk.positions:
            pos = risk.positions[symbol][0]
            result = risk.close_position(symbol, pos, close_price)
            if result:
                result["reason"]   = "signal"
                result["entry_ts"] = entry_ts.pop(symbol, None)
                result["exit_ts"]  = candle_ts
                trades.append(result)
                if symbol in entry_index:
                    durations.append(i - entry_index.pop(symbol))

        equity_curve.append(risk.total_capital)

    # Ferme les positions restantes au dernier prix de clôture
    unclosed   = len(risk.positions)
    last_price = df.iloc[-1]["close"]
    last_ts    = df.iloc[-1]["timestamp"]
    last_i     = len(df) - 1
    for sym in list(risk.positions.keys()):
        for pos in list(risk.positions[sym]):
            result = risk.close_position(sym, pos, last_price)
            if result:
                result["reason"]   = "end"
                result["entry_ts"] = entry_ts.pop(sym, None)
                result["exit_ts"]  = last_ts
                trades.append(result)
                if sym in entry_index:
                    durations.append(last_i - entry_index.pop(sym))

    return _compute_stats(trades, initial_capital, risk.total_capital, equity_curve,
                          symbol, timeframe, durations, unclosed, raw_trades=trades)


def run_single_tf(symbol: str, timeframe: str, initial_capital: float = 1000.0) -> dict:
    """Backtest sur une seule timeframe."""
    df = fetch_historical(symbol, timeframe)
    df = indicators.compute_all(df)
    return _simulate(df, symbol, timeframe, initial_capital)


def run_multi_tf(symbol: str, tf_short: str, tf_long: str, initial_capital: float = 1000.0) -> dict:
    """
    Backtest multi-timeframe :
    Signal BUY validé seulement si tf_short=BUY ET tf_long=BUY.
    Signal SELL validé seulement si tf_short=SELL ET tf_long=SELL.
    """
    label = f"{tf_short}+{tf_long}"
    print(f"  Multi-TF {symbol} [{label}]...", end=" ", flush=True)

    df_short = fetch_historical(symbol, tf_short)
    df_long = fetch_historical(symbol, tf_long)

    df_short = indicators.compute_all(df_short)
    df_long = indicators.compute_all(df_long)
    df_short = df_short.dropna().reset_index()
    df_long = df_long.dropna().reset_index()

    if len(df_short) < 10:
        return {"symbol": symbol, "timeframe": label, "trades": 0, "note": "Pas assez de données"}

    risk = RiskManager(initial_capital)
    trades = []
    equity_curve = [initial_capital]

    for i in range(2, len(df_short)):
        window_short = df_short.iloc[: i + 1].set_index("timestamp")
        score_short = indicators.score_signal(window_short)
        price = score_short["price"]
        ts_short = df_short.iloc[i]["timestamp"]

        # Fenêtre long : bougies jusqu'au timestamp courant
        long_up_to = df_long[df_long["timestamp"] <= ts_short]
        if len(long_up_to) < 3:
            equity_curve.append(risk.total_capital)
            continue
        score_long = indicators.score_signal(long_up_to.set_index("timestamp"))

        # Signal final : accord entre les deux TF
        if score_short["signal"] == "BUY" and score_long["signal"] == "BUY":
            final_signal = "BUY"
        elif score_short["signal"] == "SELL" and score_long["signal"] == "SELL":
            final_signal = "SELL"
        else:
            final_signal = "HOLD"

        exit_reason = risk.check_exits(symbol, price)
        if exit_reason in ("stop_loss", "take_profit"):
            result = risk.close_position(symbol, price)
            if result:
                risk.update_capital(result["pnl_usdt"])
                result["reason"] = exit_reason
                trades.append(result)
            equity_curve.append(risk.total_capital)
            continue

        if final_signal == "BUY":
            risk.open_position(symbol, price)
        elif final_signal == "SELL" and symbol in risk.positions:
            result = risk.close_position(symbol, price)
            if result:
                risk.update_capital(result["pnl_usdt"])
                result["reason"] = "signal"
                trades.append(result)

        equity_curve.append(risk.total_capital)

    last_price = df_short.iloc[-1]["close"]
    for sym in list(risk.positions.keys()):
        result = risk.close_position(sym, last_price)
        if result:
            risk.update_capital(result["pnl_usdt"])
            result["reason"] = "end"
            trades.append(result)

    print(f"{len(trades)} trades")
    return _compute_stats(trades, initial_capital, risk.total_capital, equity_curve, symbol, label)


def _compute_stats(trades, initial_capital, final_capital, equity_curve,
                   symbol, timeframe, durations=None, unclosed=0, raw_trades=None) -> dict:
    if not trades:
        return {"symbol": symbol, "timeframe": timeframe, "trades": 0, "note": "Aucun trade"}

    pnls = [t["pnl_usdt"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_return_pct = (final_capital - initial_capital) / initial_capital * 100
    win_rate = len(wins) / len(pnls) * 100 if pnls else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
    tf_hours = {"30m": 0.5, "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24}
    if durations and timeframe in tf_hours:
        avg_duration_j = round(sum(durations) / len(durations) * tf_hours[timeframe] / 24, 1)
    else:
        avg_duration_j = 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        "symbol":         symbol,
        "timeframe":      timeframe,
        "trades":         len(trades),
        "win_%":          round(win_rate, 1),
        "return_%":       round(total_return_pct, 2),
        "capital_final":  round(final_capital, 2),
        "profit_factor":  round(profit_factor, 2),
        "avg_win_$":      round(avg_win, 2),
        "avg_loss_$":     round(avg_loss, 2),
        "drawdown_%":     round(max_dd, 2),
        "avg_duration_j": avg_duration_j,
        "unclosed":       unclosed,
        "_raw_trades":    raw_trades or [],   # trades bruts avec timestamps (usage interne)
    }


def print_results(results: list[dict], title: str):
    valid = [r for r in results if "return_%" in r]
    if not valid:
        print(f"\n{title}: Aucun trade effectué.")
        return

    valid.sort(key=lambda x: x["return_%"], reverse=True)
    headers = list(valid[0].keys())
    rows = [[r.get(h, "—") for h in headers] for r in valid]

    print(f"\n{Fore.YELLOW}=== {title} ==={Style.RESET_ALL}")
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline", floatfmt=".2f"))

    best = valid[0]
    print(
        f"{Fore.GREEN}Meilleur: {best['symbol']} | {best['timeframe']} | "
        f"Return: {best['return_%']}% | Win: {best['win_%']}% | "
        f"Trades: {best['trades']} | Drawdown: {best['drawdown_%']}%{Style.RESET_ALL}"
    )


if __name__ == "__main__":
    CAPITAL = 1000.0

    # Backtest portefeuille 20 cryptos — chaque symbole avec son timeframe optimal
    print(f"\n{Fore.CYAN}{'='*60}")
    print("  BACKTEST PORTEFEUILLE 20 CRYPTOS — 4 ans de données")
    print(f"  Capital initial : {CAPITAL} USDT | Position size : {config.POSITION_SIZE_PCT*100:.0f}% par trade")
    print(f"{'='*60}{Style.RESET_ALL}")

    results = []
    for symbol in config.SYMBOLS:
        tf = config.SYMBOL_TIMEFRAMES.get(symbol, config.TIMEFRAME)
        risk = config.SYMBOL_RISK.get(symbol, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT})
        try:
            stats = run_single_tf(symbol, tf, CAPITAL)
            stats["sl_%"] = risk["sl"] * 100
            stats["tp_%"] = risk["tp"] * 100
            results.append(stats)
            r = stats.get("return_%", "N/A")
            t = stats.get("trades", 0)
            print(f"  -> {symbol} {tf}: Return {r}% | {t} trades")
        except Exception as e:
            print(f"{Fore.RED}  Erreur {symbol}/{tf}: {e}{Style.RESET_ALL}")

    print_results(results, "PORTEFEUILLE 20 CRYPTOS — 2 ans")
