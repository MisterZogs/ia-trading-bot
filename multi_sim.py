"""
24 simulations comparatives :
  6 portfolios × 2 modes (1 pos max / multi-positions) × 2 SL (avec / sans)
  × 7 timeframes × 4 périodes = 672 backtests

Plus colonne buy-and-hold pour chaque configuration.

Portfolios :
  top20  — 20 cryptos actuelles
  top10  — BTC ETH BNB SOL XRP ADA AVAX DOT LINK MATIC
  top5   — BTC ETH BNB SOL XRP
  btceth — BTC ETH
  btc    — BTC seulement
  eth    — ETH seulement

Modes :
  single — 1 position max par paire (comportement actuel)
  multi  — plusieurs positions par paire (capital max 100%)

SL :
  with_sl    — stop-loss activé (actuel)
  without_sl — pas de stop-loss, sortie sur signal SELL ou TP uniquement
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import indicators
import data_cache
import fear_greed

init(autoreset=True)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
FEE_RATE        = 0.001  # 0.1% par côté
ATR_TRAIL_MULT  = 2.0    # multiplicateur ATR pour le trailing stop

ALL_TIMEFRAMES = ["30m", "1h", "2h", "4h", "6h", "12h", "1d"]

CANDLES_PER_YEAR = {
    "30m": 17520, "1h": 8760, "2h": 4380,
    "4h":   2190, "6h": 1460, "12h":  730, "1d":  365,
}

TF_HOURS = {
    "30m": 0.5, "1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24,
}

PERIODS = {1: "1an", 2: "2ans", 3: "3ans", 4: "4ans"}

INITIAL_CAPITAL = 1000.0
POSITION_SIZE_PCT = config.POSITION_SIZE_PCT   # 5%

# ---------------------------------------------------------------------------
# Définition des 6 portfolios
# ---------------------------------------------------------------------------
TOP10 = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
]
TOP5  = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT"]

PORTFOLIOS = {
    "top20":  config.SYMBOLS,
    "top10":  TOP10,
    "top5":   TOP5,
    "btceth": ["BTC/USDT", "ETH/USDT"],
    "btc":    ["BTC/USDT"],
    "eth":    ["ETH/USDT"],
}

PORTFOLIO_LABELS = {
    "top20":  "Top 20",
    "top10":  "Top 10",
    "top5":   "Top 5",
    "btceth": "BTC+ETH",
    "btc":    "BTC",
    "eth":    "ETH",
}

# ---------------------------------------------------------------------------
# Cache indicateurs en mémoire
# ---------------------------------------------------------------------------
_ind_cache: dict = {}
_ind_cache_8y: dict = {}
_ind_cache_10y: dict = {}

def get_df(symbol: str, timeframe: str) -> pd.DataFrame | None:
    key = f"{symbol}_{timeframe}"
    if key in _ind_cache:
        return _ind_cache[key]
    try:
        raw = data_cache.fetch_ohlcv(symbol, timeframe, verbose=False)
        df  = raw.reset_index()
        df  = indicators.compute_all(df)
        df  = df.dropna().reset_index(drop=True)
        _ind_cache[key] = df
        return df
    except Exception as e:
        print(f"  [err] {symbol} {timeframe}: {e}")
        return None


def get_df_10y(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Charge 10 ans de données et calcule les indicateurs (cache mémoire)."""
    key = f"{symbol}_{timeframe}"
    if key in _ind_cache_10y:
        return _ind_cache_10y[key]
    try:
        raw = data_cache.fetch_ohlcv_10y(symbol, timeframe, verbose=False)
        df  = raw.reset_index()
        df  = indicators.compute_all(df)
        df  = df.dropna().reset_index(drop=True)
        _ind_cache_10y[key] = df
        return df
    except Exception as e:
        _ind_cache_10y[key] = None
        return None


def get_df_for_year(symbol: str, tf: str, year: int) -> pd.DataFrame | None:
    """
    Retourne le DF avec indicateurs filtré sur l'année calendaire `year`.
    Choisit automatiquement le bon tier de données (4y / 8y / 10y).
    """
    import datetime
    years_back = datetime.date.today().year - year

    if years_back <= 4:
        df = get_df(symbol, tf)
    elif years_back <= 8 and tf in data_cache.TF_WITH_8Y:
        df = get_df_8y(symbol, tf)
    elif tf in data_cache.TF_WITH_10Y:
        df = get_df_10y(symbol, tf)
    else:
        return None  # 30m/1h pas disponibles si > 4 ans

    if df is None or "timestamp" not in df.columns:
        return None

    filtered = df[df["timestamp"].dt.year == year].reset_index(drop=True)
    min_candles = max(10, int(CANDLES_PER_YEAR[tf] * 0.3))  # au moins 30% de l'année
    return filtered if len(filtered) >= min_candles else None


def get_df_8y(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Charge 8 ans de données (via data_cache._8y) et calcule les indicateurs."""
    key = f"{symbol}_{timeframe}"
    if key in _ind_cache_8y:
        return _ind_cache_8y[key]
    try:
        raw = data_cache.fetch_ohlcv_8y(symbol, timeframe, verbose=False)
        df  = raw.reset_index()
        df  = indicators.compute_all(df)
        df  = df.dropna().reset_index(drop=True)
        _ind_cache_8y[key] = df
        return df
    except Exception as e:
        # Certains altcoins n'ont pas 8 ans d'historique
        _ind_cache_8y[key] = None
        return None


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------
def buy_and_hold(symbols: list[str], timeframe: str, years: int) -> float:
    """
    Retourne le return % moyen équipondéré si on avait acheté et gardé
    pendant toute la période.
    """
    returns = []
    n = CANDLES_PER_YEAR[timeframe] * years
    for symbol in symbols:
        df = get_df(symbol, timeframe)
        if df is None or len(df) < 10:
            continue
        df_period = df.tail(n).reset_index(drop=True)
        start = df_period["close"].iloc[0]
        end   = df_period["close"].iloc[-1]
        if start > 0:
            returns.append((end - start) / start * 100)
    if not returns:
        return 0.0
    return round(sum(returns) / len(returns), 1)


# ---------------------------------------------------------------------------
# Simulation — mode single (1 position max par paire)
# ---------------------------------------------------------------------------
def sim_single(symbols: list[str], timeframe: str, years: int,
               use_sl: bool, fg: dict | None = None,
               trail_sl: bool = False, use_triple_st: bool = True,
               use_sma_macd: bool = True, old_period: bool = False) -> dict:
    """
    Simule le portefeuille avec 1 position max par paire.
    use_sl        : True = stop-loss fixe, False = pas de SL
    trail_sl      : True = ATR trailing stop (remplace use_sl)
    fg            : dict Fear & Greed (optionnel)
    use_triple_st : inclure ou non la condition Triple SuperTrend
    use_sma_macd  : inclure ou non SMA20<SMA50 (b3) et MACD<Signal (b4)
    old_period    : True = utilise les données 8y, slice années -5 à -(4+years)
    """
    n    = CANDLES_PER_YEAR[timeframe] * years
    n4y  = CANDLES_PER_YEAR[timeframe] * 4
    dfs  = {}
    sigs = {}
    for symbol in symbols:
        df = get_df_8y(symbol, timeframe) if old_period else get_df(symbol, timeframe)
        if df is not None and len(df) >= 10:
            if old_period:
                if len(df) <= n4y:
                    continue  # pas assez d'historique pour la frontière récente
                sliced = df.iloc[:-n4y].tail(n).reset_index(drop=True)
                if len(sliced) < int(n * 0.9):
                    continue  # moins de 90% de la période → non représentatif
            else:
                sliced = df.tail(n).reset_index(drop=True)
            dfs[symbol]  = sliced
            sigs[symbol] = indicators.vectorized_signals(
                sliced, use_triple_st=use_triple_st, use_sma_macd=use_sma_macd).values
    if not dfs:
        return {}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    import math
    capital   = INITIAL_CAPITAL
    positions = {}
    trades    = []
    durations = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            low   = arr_low[symbol][i]
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            atr   = arr_atr[symbol][i]

            if symbol in positions:
                pos = positions[symbol]

                # Mise à jour du trailing stop (monte avec le prix, jamais descend)
                if trail_sl and not math.isnan(atr):
                    new_trail = close - ATR_TRAIL_MULT * atr
                    pos["sl"] = max(pos["sl"], new_trail)

                tp_hit = high >= pos["tp"]
                sl_hit = (trail_sl or use_sl) and low <= pos["sl"]
                if sl_hit or tp_hit:
                    exit_px  = pos["sl"] if sl_hit else pos["tp"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                    del positions[symbol]
                    continue

            sig = sigs[symbol][i]

            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            if sig == "BUY" and symbol not in positions:
                risk    = risks[symbol]
                pos_val = capital * POSITION_SIZE_PCT
                capital -= pos_val * FEE_RATE
                size    = pos_val / close
                # SL initial : trailing basé sur ATR, ou fixe, ou absent
                if trail_sl and not math.isnan(atr):
                    initial_sl = close - ATR_TRAIL_MULT * atr
                elif use_sl:
                    initial_sl = close * (1 - risk["sl"])
                else:
                    initial_sl = 0.0  # pas de SL
                positions[symbol] = {
                    "entry": close, "size": size,
                    "sl": initial_sl,
                    "tp": close * (1 + risk["tp"]),
                    "entry_i": i,
                }
            elif sig == "SELL" and symbol in positions:
                pos      = positions[symbol]
                fee_exit = close * pos["size"] * FEE_RATE
                pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                capital += pnl
                trades.append(pnl)
                durations.append(i - pos["entry_i"])
                del positions[symbol]

        equity.append(capital)

    # Clôture finale
    unclosed = len(positions)
    for symbol, pos in positions.items():
        last = arr_close[symbol][-1]
        fee  = last * pos["size"] * FEE_RATE
        pnl  = (last - pos["entry"]) * pos["size"] - fee
        capital += pnl
        trades.append(pnl)
        durations.append(max_len - 1 - pos["entry_i"])

    return _stats(trades, capital, equity, durations, unclosed, timeframe)


# ---------------------------------------------------------------------------
# Simulation — mode multi (plusieurs positions par paire)
# ---------------------------------------------------------------------------
def sim_multi(symbols: list[str], timeframe: str, years: int,
              use_sl: bool, fg: dict | None = None,
              trail_sl: bool = False, use_triple_st: bool = True,
              use_sma_macd: bool = True, old_period: bool = False) -> dict:
    """
    Simule le portefeuille avec plusieurs positions possibles par paire.
    use_sl        : stop-loss fixe | trail_sl : ATR trailing stop
    use_triple_st : inclure ou non la condition Triple SuperTrend
    use_sma_macd  : inclure ou non SMA20<SMA50 (b3) et MACD<Signal (b4)
    old_period    : True = utilise les données 8y, slice années -5 à -(4+years)
    """
    import math
    n    = CANDLES_PER_YEAR[timeframe] * years
    n4y  = CANDLES_PER_YEAR[timeframe] * 4
    dfs  = {}
    sigs = {}
    for symbol in symbols:
        df = get_df_8y(symbol, timeframe) if old_period else get_df(symbol, timeframe)
        if df is not None and len(df) >= 10:
            if old_period:
                if len(df) <= n4y:
                    continue  # pas assez d'historique pour la frontière récente
                sliced = df.iloc[:-n4y].tail(n).reset_index(drop=True)
                if len(sliced) < int(n * 0.9):
                    continue  # moins de 90% de la période → non représentatif
            else:
                sliced = df.tail(n).reset_index(drop=True)
            dfs[symbol]  = sliced
            sigs[symbol] = indicators.vectorized_signals(
                sliced, use_triple_st=use_triple_st, use_sma_macd=use_sma_macd).values
    if not dfs:
        return {}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = INITIAL_CAPITAL
    positions: dict[str, list] = {s: [] for s in dfs}
    trades    = []
    durations = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            low   = arr_low[symbol][i]
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            atr   = arr_atr[symbol][i]

            # SL / TP check + mise à jour trailing stop
            still_open = []
            for pos in positions[symbol]:
                if trail_sl and not math.isnan(atr):
                    pos["sl"] = max(pos["sl"], close - ATR_TRAIL_MULT * atr)
                tp_hit = high >= pos["tp"]
                sl_hit = (trail_sl or use_sl) and low <= pos["sl"]
                if sl_hit or tp_hit:
                    exit_px  = pos["sl"] if sl_hit else pos["tp"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                else:
                    still_open.append(pos)
            positions[symbol] = still_open

            sig = sigs[symbol][i]

            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            pos_val   = capital * POSITION_SIZE_PCT
            deployed  = sum(p["entry"] * p["size"] for plist in positions.values() for p in plist)
            available = capital - deployed

            if sig == "BUY" and available >= pos_val:
                risk    = risks[symbol]
                capital -= pos_val * FEE_RATE
                size    = pos_val / close
                if trail_sl and not math.isnan(atr):
                    initial_sl = close - ATR_TRAIL_MULT * atr
                elif use_sl:
                    initial_sl = close * (1 - risk["sl"])
                else:
                    initial_sl = 0.0
                positions[symbol].append({
                    "entry": close, "size": size,
                    "sl": initial_sl,
                    "tp": close * (1 + risk["tp"]),
                    "entry_i": i,
                })

            elif sig == "SELL" and positions[symbol]:
                for pos in positions[symbol]:
                    fee_exit = close * pos["size"] * FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                positions[symbol] = []

        equity.append(capital)

    # Clôture finale
    unclosed = sum(len(plist) for plist in positions.values())
    for symbol, plist in positions.items():
        last = arr_close[symbol][-1]
        for pos in plist:
            fee  = last * pos["size"] * FEE_RATE
            pnl  = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)
            durations.append(max_len - 1 - pos["entry_i"])

    return _stats(trades, capital, equity, durations, unclosed, timeframe)


# ---------------------------------------------------------------------------
# Simulation multi avec concentration paramétrable
# ---------------------------------------------------------------------------
def sim_concentration(dfs: dict, pos_pct: float, max_trades: int,
                      fg: dict | None = None,
                      use_triple_st: bool = False, use_sma_macd: bool = False,
                      use_sl: bool = False, tf: str | None = None,
                      single: bool = False,
                      use_tp: bool = True,
                      tp_pct: float | None = None,
                      atr_tp: bool = False,
                      precomputed_sigs: dict | None = None) -> dict:
    """
    Multi-position avec limite globale de trades simultanés et taille de
    position fixe en % du capital.

    pos_pct          : fraction du capital par trade (ex. 0.10 = 10%)
    max_trades       : nombre max de positions ouvertes simultanément (tous symboles)
    use_sl           : activer le stop-loss fixe (par défaut False = sansSL)
    single           : si True, au plus 1 position ouverte par symbole (mode 1pos)
    use_tp           : False = pas de TP (sortie sur signal SELL ou SL uniquement)
    tp_pct           : TP fixe uniforme pour tous les symboles (ex: 0.10 = 10%)
    atr_tp           : TP dynamique = entrée + ATR_TP_MULT × ATR
    precomputed_sigs : signaux déjà calculés {symbol: np.array} — évite le recalcul
    """
    import math
    if not dfs:
        return {}

    if precomputed_sigs is not None:
        sigs = {s: precomputed_sigs[s] for s in dfs if s in precomputed_sigs}
    else:
        sigs = {s: indicators.vectorized_signals(
            dfs[s], use_triple_st=use_triple_st, use_sma_macd=use_sma_macd).values
            for s in dfs}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = INITIAL_CAPITAL
    positions: dict[str, list] = {s: [] for s in dfs}
    trades    = []
    durations = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val     = None
        total_open = sum(len(plist) for plist in positions.values())

        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            low   = arr_low[symbol][i]
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            atr   = arr_atr[symbol][i]
            risk  = risks[symbol]

            still_open = []
            for pos in positions[symbol]:
                tp_hit = high >= pos["tp"]
                sl_hit = use_sl and low <= pos["sl"]
                if tp_hit or sl_hit:
                    exit_px  = pos["tp"] if tp_hit else pos["sl"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                    total_open -= 1
                else:
                    still_open.append(pos)
            positions[symbol] = still_open

            sig = sigs[symbol][i]
            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            can_buy = total_open < max_trades and (not single or len(positions[symbol]) == 0)
            if sig == "BUY" and can_buy:
                pos_val = capital * pos_pct
                if pos_val > 0 and close > 0:
                    capital -= pos_val * FEE_RATE
                    size    = pos_val / close
                    initial_sl = close * (1 - risk["sl"]) if use_sl else 0.0
                    if not use_tp:
                        tp_price = math.inf
                    elif atr_tp and not math.isnan(atr) and atr > 0:
                        tp_price = close + config.ATR_TP_MULT * atr
                        tp_price = max(tp_price, close * (1 + config.ATR_TP_MIN_PCT))
                    elif tp_pct is not None:
                        tp_price = close * (1 + tp_pct)
                    else:
                        tp_price = close * (1 + risk["tp"])
                    positions[symbol].append({
                        "entry":   close,
                        "size":    size,
                        "sl":      initial_sl,
                        "tp":      tp_price,
                        "entry_i": i,
                    })
                    total_open += 1

            elif sig == "SELL" and positions[symbol]:
                for pos in positions[symbol]:
                    fee_exit = close * pos["size"] * FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                    total_open -= 1
                positions[symbol] = []

        equity.append(capital)

    unclosed = sum(len(plist) for plist in positions.values())
    for symbol, plist in positions.items():
        last = arr_close[symbol][-1]
        for pos in plist:
            fee  = last * pos["size"] * FEE_RATE
            pnl  = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)
            durations.append(max_len - 1 - pos["entry_i"])

    return _stats(trades, capital, equity, durations, unclosed, tf)


# ---------------------------------------------------------------------------
# Simulation sur DataFrames pré-construits (pour walk-forward)
# ---------------------------------------------------------------------------
def sim_multi_on_dfs(dfs: dict, use_sl: bool, fg: dict | None = None,
                     trail_sl: bool = False, use_triple_st: bool = True,
                     use_sma_macd: bool = True,
                     use_regime_filter: bool = False,
                     atr_sizing: bool = False,
                     atr_tp: bool = False,
                     use_tp: bool = True,
                     tp_pct: float | None = None,
                     precomputed_sigs: dict | None = None,
                     tf: str | None = None) -> dict:
    """
    Simule le portefeuille multi sur des DataFrames déjà découpés.
    Contrairement à sim_multi(), ne charge pas de données — prend les DFs directement.
    Utilisé par le walk-forward (etape8), le regime test (etape9) et le test ATR (etape10).

    dfs               : dict {symbol: DataFrame déjà slicé et avec indicateurs}
    use_sl            : stop-loss fixe
    fg                : dict Fear & Greed (optionnel)
    trail_sl          : ATR trailing stop
    use_triple_st     : inclure la condition Triple SuperTrend
    use_sma_macd      : inclure SMA20<SMA50 (b3) et MACD<Signal (b4)
    use_regime_filter : bloquer les BUY si ADX > 25 (marché en tendance forte)
    atr_sizing        : taille de position adaptée à la volatilité ATR courante
    atr_tp            : take-profit dynamique = entrée + ATR_TP_MULT × ATR
    use_tp            : False = pas de TP (sortie sur signal SELL ou SL uniquement)
    tp_pct            : TP fixe uniforme pour tous les symboles (ex: 0.10 = 10%)
                        priorité sur SYMBOL_RISK si fourni ; ignoré si atr_tp=True ou use_tp=False
    """
    import math
    if not dfs:
        return {}

    if precomputed_sigs is not None:
        sigs = {s: precomputed_sigs[s] for s in dfs if s in precomputed_sigs}
    else:
        sigs = {s: indicators.vectorized_signals(
            dfs[s], use_triple_st=use_triple_st, use_sma_macd=use_sma_macd,
            use_regime_filter=use_regime_filter).values
            for s in dfs}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = INITIAL_CAPITAL
    positions: dict[str, list] = {s: [] for s in dfs}
    trades    = []
    durations = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            low   = arr_low[symbol][i]
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            atr   = arr_atr[symbol][i]

            still_open = []
            for pos in positions[symbol]:
                if trail_sl and not math.isnan(atr):
                    pos["sl"] = max(pos["sl"], close - ATR_TRAIL_MULT * atr)
                tp_hit = high >= pos["tp"]
                sl_hit = (trail_sl or use_sl) and low <= pos["sl"]
                if sl_hit or tp_hit:
                    exit_px  = pos["sl"] if sl_hit else pos["tp"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                else:
                    still_open.append(pos)
            positions[symbol] = still_open

            sig = sigs[symbol][i]

            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            base_slot = capital * POSITION_SIZE_PCT
            deployed  = sum(p["entry"] * p["size"] for plist in positions.values() for p in plist)
            available = capital - deployed

            if sig == "BUY" and available >= base_slot:
                risk = risks[symbol]

                # --- Sizing adaptatif à la volatilité ATR ---
                if atr_sizing and not math.isnan(atr) and atr > 0 and close > 0:
                    atr_pct = atr / close
                    scale   = config.ATR_SIZING_REF_PCT / atr_pct
                    scale   = min(max(scale, config.ATR_SIZING_MIN_MULT),
                                  config.ATR_SIZING_MAX_MULT)
                else:
                    scale = 1.0
                pos_val  = base_slot * scale
                capital -= pos_val * FEE_RATE
                size     = pos_val / close

                # --- Take-Profit ---
                if not use_tp:
                    tp_price = math.inf
                elif atr_tp and not math.isnan(atr) and atr > 0:
                    tp_price = close + config.ATR_TP_MULT * atr
                    tp_price = max(tp_price, close * (1 + config.ATR_TP_MIN_PCT))
                elif tp_pct is not None:
                    tp_price = close * (1 + tp_pct)
                else:
                    tp_price = close * (1 + risk["tp"])

                if trail_sl and not math.isnan(atr):
                    initial_sl = close - ATR_TRAIL_MULT * atr
                elif use_sl:
                    initial_sl = close * (1 - risk["sl"])
                else:
                    initial_sl = 0.0
                positions[symbol].append({
                    "entry": close, "size": size,
                    "sl": initial_sl,
                    "tp": tp_price,
                    "entry_i": i,
                })

            elif sig == "SELL" and positions[symbol]:
                for pos in positions[symbol]:
                    fee_exit = close * pos["size"] * FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                positions[symbol] = []

        equity.append(capital)

    # Clôture finale
    unclosed = sum(len(plist) for plist in positions.values())
    for symbol, plist in positions.items():
        last = arr_close[symbol][-1]
        for pos in plist:
            fee  = last * pos["size"] * FEE_RATE
            pnl  = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)
            durations.append(max_len - 1 - pos["entry_i"])

    return _stats(trades, capital, equity, durations, unclosed, tf)


# ---------------------------------------------------------------------------
# Simulation multi avec pondération du capital par symbole
# ---------------------------------------------------------------------------
def sim_multi_weighted(dfs: dict, pos_pct_per_symbol: dict[str, float],
                       fg: dict | None = None,
                       use_triple_st: bool = False, use_sma_macd: bool = False,
                       use_sl: bool = False, tf: str | None = None) -> dict:
    """
    Multi-position avec taille de position différente par symbole.

    pos_pct_per_symbol : dict {symbol: fraction_du_capital}
                         ex. {"BTC/USDT": 0.08, "ALGO/USDT": 0.02}
    Les symboles absents reçoivent POSITION_SIZE_PCT par défaut.
    Sans limite globale de trades simultanés.
    """
    import math
    if not dfs:
        return {}

    sigs = {s: indicators.vectorized_signals(
        dfs[s], use_triple_st=use_triple_st, use_sma_macd=use_sma_macd).values
        for s in dfs}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = INITIAL_CAPITAL
    positions: dict[str, list] = {s: [] for s in dfs}
    trades    = []
    durations = []
    equity    = [capital]
    pos_pcts_used: list[float] = []  # pos_pct réel à chaque BUY effectif
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            risk  = risks[symbol]

            still_open = []
            for pos in positions[symbol]:
                tp_hit = high >= pos["tp"]
                sl_hit = use_sl and arr_low[symbol][i] <= pos["sl"]
                if tp_hit or sl_hit:
                    exit_px  = pos["tp"] if tp_hit else pos["sl"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                else:
                    still_open.append(pos)
            positions[symbol] = still_open

            sig = sigs[symbol][i]
            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            pos_pct = pos_pct_per_symbol.get(symbol, POSITION_SIZE_PCT)
            deployed = sum(p["entry"] * p["size"] for plist in positions.values() for p in plist)
            available = capital - deployed
            pos_val = capital * pos_pct

            if sig == "BUY" and available >= pos_val:
                pos_pcts_used.append(pos_pct)
                capital -= pos_val * FEE_RATE
                size    = pos_val / close
                initial_sl = close * (1 - risk["sl"]) if use_sl else 0.0
                positions[symbol].append({
                    "entry":   close,
                    "size":    size,
                    "sl":      initial_sl,
                    "tp":      close * (1 + risk["tp"]),
                    "entry_i": i,
                })

            elif sig == "SELL" and positions[symbol]:
                for pos in positions[symbol]:
                    fee_exit = close * pos["size"] * FEE_RATE
                    pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                positions[symbol] = []

        equity.append(capital)

    unclosed = sum(len(plist) for plist in positions.values())
    for symbol, plist in positions.items():
        last = arr_low[symbol][-1]  # approximation conservative
        last = arr_close[symbol][-1]
        for pos in plist:
            fee  = last * pos["size"] * FEE_RATE
            pnl  = (last - pos["entry"]) * pos["size"] - fee
            capital += pnl
            trades.append(pnl)
            durations.append(max_len - 1 - pos["entry_i"])

    result = _stats(trades, capital, equity, durations, unclosed, tf)
    if pos_pcts_used:
        result["avg_pos_pct"] = sum(pos_pcts_used) / len(pos_pcts_used)
        result["total_pos_pct"] = sum(pos_pcts_used)
    else:
        result["avg_pos_pct"] = 0.0
        result["total_pos_pct"] = 0.0
    return result


# ---------------------------------------------------------------------------
# Simulation — mode single sur DataFrames pré-construits (années calendaires)
# ---------------------------------------------------------------------------
def sim_single_on_dfs(dfs: dict, use_sl: bool, fg: dict | None = None,
                      trail_sl: bool = False, use_triple_st: bool = True,
                      use_sma_macd: bool = True,
                      precomputed_sigs: dict | None = None,
                      tf: str | None = None) -> dict:
    """Version single-position de sim_multi_on_dfs (1 position max par paire)."""
    import math
    if not dfs:
        return {}

    if precomputed_sigs is not None:
        sigs = {s: precomputed_sigs[s] for s in dfs if s in precomputed_sigs}
    else:
        sigs = {s: indicators.vectorized_signals(
            dfs[s], use_triple_st=use_triple_st, use_sma_macd=use_sma_macd).values
            for s in dfs}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_atr   = {s: dfs[s]["atr"].values   for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = INITIAL_CAPITAL
    positions = {}
    trades    = []
    durations = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for symbol in dfs:
            if i >= arr_len[symbol]:
                continue
            low   = arr_low[symbol][i]
            high  = arr_high[symbol][i]
            close = arr_close[symbol][i]
            atr   = arr_atr[symbol][i]

            if symbol in positions:
                pos = positions[symbol]
                if trail_sl and not math.isnan(atr):
                    pos["sl"] = max(pos["sl"], close - ATR_TRAIL_MULT * atr)
                tp_hit = high >= pos["tp"]
                sl_hit = (trail_sl or use_sl) and low <= pos["sl"]
                if sl_hit or tp_hit:
                    exit_px  = pos["sl"] if sl_hit else pos["tp"]
                    fee_exit = exit_px * pos["size"] * FEE_RATE
                    pnl      = (exit_px - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                    durations.append(i - pos["entry_i"])
                    del positions[symbol]
                    continue

            sig = sigs[symbol][i]
            if sig == "BUY" and fg_val is not None:
                if fg_val > fear_greed.FG_GREED_VETO:
                    sig = "HOLD"

            if sig == "BUY" and symbol not in positions:
                risk    = risks[symbol]
                pos_val = capital * POSITION_SIZE_PCT
                capital -= pos_val * FEE_RATE
                size    = pos_val / close
                if trail_sl and not math.isnan(atr):
                    initial_sl = close - ATR_TRAIL_MULT * atr
                elif use_sl:
                    initial_sl = close * (1 - risk["sl"])
                else:
                    initial_sl = 0.0
                positions[symbol] = {
                    "entry": close, "size": size,
                    "sl": initial_sl,
                    "tp": close * (1 + risk["tp"]),
                    "entry_i": i,
                }
            elif sig == "SELL" and symbol in positions:
                pos      = positions[symbol]
                fee_exit = close * pos["size"] * FEE_RATE
                pnl      = (close - pos["entry"]) * pos["size"] - fee_exit
                capital += pnl
                trades.append(pnl)
                durations.append(i - pos["entry_i"])
                del positions[symbol]

        equity.append(capital)

    unclosed = len(positions)
    for symbol, pos in positions.items():
        last = arr_close[symbol][-1]
        fee  = last * pos["size"] * FEE_RATE
        pnl  = (last - pos["entry"]) * pos["size"] - fee
        capital += pnl
        trades.append(pnl)
        durations.append(max_len - 1 - pos["entry_i"])

    return _stats(trades, capital, equity, durations, unclosed, tf)


# ---------------------------------------------------------------------------
# Simulation sur une année calendaire
# ---------------------------------------------------------------------------
def sim_year(symbols: list[str], tf: str, year: int, mode: str,
             use_sl: bool, fg: dict | None = None,
             use_triple_st: bool = True, use_sma_macd: bool = True) -> dict:
    """
    Simule le portefeuille sur une année calendaire précise (ex. 2021).
    Choisit automatiquement le bon tier de données selon l'ancienneté.
    """
    dfs = {}
    for symbol in symbols:
        df = get_df_for_year(symbol, tf, year)
        if df is not None:
            dfs[symbol] = df
    if not dfs:
        return {}

    fn = sim_multi_on_dfs if mode == "multi" else sim_single_on_dfs
    return fn(dfs, use_sl=use_sl, fg=fg,
              use_triple_st=use_triple_st, use_sma_macd=use_sma_macd, tf=tf)


# ---------------------------------------------------------------------------
# Stats communes
# ---------------------------------------------------------------------------
def _stats(trades: list, capital: float, equity: list,
           durations: list | None = None, unclosed: int = 0,
           tf: str | None = None) -> dict:
    if not trades:
        return {"trades": 0, "return_%": 0.0, "win_%": 0.0,
                "profit_factor": 0.0, "drawdown_%": 0.0,
                "avg_duration_j": 0.0, "unclosed": 0}
    wins   = [p for p in trades if p > 0]
    losses = [p for p in trades if p <= 0]
    pf     = sum(wins) / abs(sum(losses)) if losses else float("inf")
    ret    = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    peak   = equity[0]
    max_dd = 0.0
    for v in equity:
        peak   = max(peak, v)
        dd     = (peak - v) / peak * 100
        max_dd = max(max_dd, dd)
    if durations and tf and tf in TF_HOURS:
        avg_dur_j = round(sum(durations) / len(durations) * TF_HOURS[tf] / 24, 1)
    else:
        avg_dur_j = 0.0
    return {
        "trades":          len(trades),
        "win_%":           round(len(wins) / len(trades) * 100, 1),
        "return_%":        round(ret, 2),
        "profit_factor":   round(pf, 2),
        "drawdown_%":      round(max_dd, 2),
        "avg_duration_j":  avg_dur_j,
        "unclosed":        unclosed,
    }


# ---------------------------------------------------------------------------
# Affichage — tableau par période
# ---------------------------------------------------------------------------
def print_period_table(all_results: dict, bah: dict, years: int,
                       prev_6h: float | None = None):
    label = PERIODS[years]
    print(f"\n{Fore.YELLOW}{'='*110}")
    print(f"  PÉRIODE : {label.upper()} | Frais 0.1% | Capital {INITIAL_CAPITAL:.0f} USDT")
    print(f"{'='*110}{Style.RESET_ALL}")

    # Configs dans l'ordre
    configs = [
        ("top20", "single", True),  ("top20", "multi", True),  ("top20", "single", False), ("top20", "multi", False),
        ("top10", "single", True),  ("top10", "multi", True),  ("top10", "single", False), ("top10", "multi", False),
        ("top5",  "single", True),  ("top5",  "multi", True),  ("top5",  "single", False), ("top5",  "multi", False),
        ("btceth","single", True),  ("btceth","multi", True),  ("btceth","single", False), ("btceth","multi", False),
        ("btc",   "single", True),  ("btc",   "multi", True),  ("btc",   "single", False), ("btc",   "multi", False),
        ("eth",   "single", True),  ("eth",   "multi", True),  ("eth",   "single", False), ("eth",   "multi", False),
    ]

    mode_label = {"single": "1pos", "multi": "multi"}
    sl_label   = {True: "avecSL", False: "sansSL"}

    headers = ["Portfolio", "Mode", "SL"] + ALL_TIMEFRAMES + ["B&H"]
    rows    = []

    for (port, mode, use_sl) in configs:
        row = [PORTFOLIO_LABELS[port], mode_label[mode], sl_label[use_sl]]
        for tf in ALL_TIMEFRAMES:
            r   = all_results.get((port, mode, use_sl, tf, years), {})
            ret = r.get("return_%")
            if ret is None:
                row.append("—")
            else:
                color = Fore.GREEN if ret > 0 else (Fore.RED if ret < 0 else "")
                row.append(f"{color}{ret:+.1f}%{Style.RESET_ALL}")
        # Buy-and-hold (même pour tous les TF de la même config/période)
        bah_val = bah.get((port, years), 0.0)
        row.append(f"{bah_val:+.1f}%")
        rows.append(row)

    # Ligne référence v3 (6h seulement)
    if prev_6h is not None:
        rows.append(["— v3 top20 6h —", "", "avecSL"] +
                    ["" if tf != "6h" else f"{prev_6h:+.1f}%" for tf in ALL_TIMEFRAMES] +
                    [""])

    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*80}")
    print("  MULTI-SIM — 24 configurations × 7 TF × 4 périodes")
    print(f"{'='*80}{Style.RESET_ALL}")

    # Tous les symboles nécessaires
    all_symbols = list(dict.fromkeys(
        s for syms in PORTFOLIOS.values() for s in syms
    ))

    # Pré-chargement des données (depuis cache disque)
    print(f"\n{Fore.CYAN}Chargement des données depuis le cache...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, ALL_TIMEFRAMES, verbose=True)

    # Calcul indicateurs (cache mémoire)
    print(f"{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    total_pairs = len(all_symbols) * len(ALL_TIMEFRAMES)
    done = 0
    for symbol in all_symbols:
        for tf in ALL_TIMEFRAMES:
            get_df(symbol, tf)
            done += 1
            print(f"\r  {done}/{total_pairs}", end="", flush=True)
    print()

    # Fear & Greed historique
    print(f"{Fore.CYAN}Chargement Fear & Greed Index...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=True)
    print(f"  {len(fg_data)} jours de F&G disponibles (veto BUY si F&G > {fear_greed.FG_GREED_VETO})")

    # Buy-and-hold
    print(f"{Fore.CYAN}Calcul buy-and-hold...{Style.RESET_ALL}")
    bah: dict = {}
    for port, syms in PORTFOLIOS.items():
        for years in PERIODS:
            bah[(port, years)] = buy_and_hold(syms, "6h", years)

    # Simulations
    print(f"{Fore.CYAN}Simulations (672 backtests)...{Style.RESET_ALL}")
    all_results: dict = {}
    sim_fns = {"single": sim_single, "multi": sim_multi}
    total = len(PORTFOLIOS) * 2 * 2 * len(ALL_TIMEFRAMES) * len(PERIODS)
    done  = 0

    for port, syms in PORTFOLIOS.items():
        for mode in ("single", "multi"):
            for use_sl in (True, False):
                for tf in ALL_TIMEFRAMES:
                    for years in PERIODS:
                        try:
                            r = sim_fns[mode](syms, tf, years, use_sl, fg=fg_data)
                        except Exception as e:
                            r = {}
                        all_results[(port, mode, use_sl, tf, years)] = r
                        done += 1
                    print(f"\r  {done}/{total}", end="", flush=True)
    print()

    # Résultats précédents v3 (6h, top20, 1pos, avec SL) pour référence
    PREV_V3 = {1: 1.6, 2: 4.8, 3: 32.1, 4: 43.4}

    # Affichage par période
    for years in PERIODS:
        print_period_table(all_results, bah, years, prev_6h=PREV_V3.get(years))

    # Résumé : meilleure config par période
    print(f"\n{Fore.CYAN}{'='*80}")
    print("  RÉSUMÉ — Meilleure config par période (return % sur 6h)")
    print(f"{'='*80}{Style.RESET_ALL}")

    summary_headers = ["Période", "Meilleure config", "Return", "Mode", "SL", "B&H",
                       "Trades", "Win%", "Dur.moy", "Non-ferm."]
    summary_rows    = []
    for years in PERIODS:
        best_ret    = None
        best_config = None
        for port in PORTFOLIOS:
            for mode in ("single", "multi"):
                for use_sl in (True, False):
                    r   = all_results.get((port, mode, use_sl, "6h", years), {})
                    ret = r.get("return_%")
                    if ret is not None and (best_ret is None or ret > best_ret):
                        best_ret    = ret
                        best_config = (port, mode, use_sl)
        if best_config:
            port, mode, use_sl = best_config
            bah_val = bah.get((port, years), 0.0)
            r = all_results.get((port, mode, use_sl, "6h", years), {})
            summary_rows.append([
                PERIODS[years],
                PORTFOLIO_LABELS[port],
                f"{best_ret:+.2f}%",
                {"single": "1pos", "multi": "multi"}[mode],
                "avecSL" if use_sl else "sansSL",
                f"{bah_val:+.1f}%",
                r.get("trades", "—"),
                f"{r.get('win_%', 0):.1f}%",
                f"{r.get('avg_duration', 0):.1f}b",
                r.get("unclosed", "—"),
            ])
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.GREEN}Note : frais {FEE_RATE*100:.1f}% achat + {FEE_RATE*100:.1f}% vente inclus{Style.RESET_ALL}")
