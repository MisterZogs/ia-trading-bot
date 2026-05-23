"""
Calcul de tous les indicateurs techniques.
Logique 100% état (state-based) : chaque condition est vraie tant que
l'indicateur est dans sa zone, pas seulement au moment du croisement.
Cela génère plus de signaux et reflète mieux la réalité du marché.
"""

import pandas as pd
import pandas_ta as ta
import config


def compute_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule tous les indicateurs sur le DataFrame OHLCV.
    Retourne le DataFrame enrichi avec les colonnes d'indicateurs.
    """
    df = df.copy()

    # RSI
    df["rsi"] = ta.rsi(df["close"], length=config.RSI_PERIOD)

    # SMA
    df["sma_fast"] = ta.sma(df["close"], length=config.SMA_FAST)
    df["sma_slow"] = ta.sma(df["close"], length=config.SMA_SLOW)

    # MACD
    macd = ta.macd(
        df["close"],
        fast=config.MACD_FAST,
        slow=config.MACD_SLOW,
        signal=config.MACD_SIGNAL,
    )
    df["macd"] = macd[f"MACD_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]
    df["macd_signal"] = macd[f"MACDs_{config.MACD_FAST}_{config.MACD_SLOW}_{config.MACD_SIGNAL}"]

    # Stochastique
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=config.STOCH_K, d=config.STOCH_D)
    df["stoch_k"] = stoch[f"STOCHk_{config.STOCH_K}_{config.STOCH_D}_3"]
    df["stoch_d"] = stoch[f"STOCHd_{config.STOCH_K}_{config.STOCH_D}_3"]

    # Bollinger Bands (noms de colonnes dynamiques selon la version de pandas-ta)
    bb = ta.bbands(df["close"], length=config.BOLLINGER_PERIOD, std=config.BOLLINGER_STD)
    bb_cols = bb.columns.tolist()
    df["bb_lower"] = bb[next(c for c in bb_cols if c.startswith("BBL"))]
    df["bb_upper"] = bb[next(c for c in bb_cols if c.startswith("BBU"))]
    df["bb_mid"] = bb[next(c for c in bb_cols if c.startswith("BBM"))]

    # SMA200 — filtre de tendance long terme
    # Si prix < SMA200 : marché en downtrend → on bloque les BUY
    df["sma200"] = ta.sma(df["close"], length=config.SMA_TREND)

    # OBV (On-Balance Volume) — confirmation par le volume
    # OBV monte quand le volume accompagne la hausse de prix → tendance saine
    df["obv"] = ta.obv(df["close"], df["volume"])
    # SMA(20) de l'OBV pour lisser et comparer la tendance du volume
    df["obv_sma"] = ta.sma(df["obv"], length=20)

    # ATR(14) — volatilité réelle pour le volatility targeting et le trailing stop
    df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=14)

    # ADX(14) — force de tendance (>25 = tendance forte, <20 = ranging)
    # Utilisé par le filtre régime : bloque les BUY en marché directionnel
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=14)
    df["adx"] = adx_df["ADX_14"] if "ADX_14" in adx_df.columns else adx_df.iloc[:, 0]

    # Volume SMA(20) — filtre de confirmation par le volume
    df["volume_sma"] = ta.sma(df["volume"], length=20)

    # Triple SuperTrend — filtre de tendance multi-période
    # ST fast (7,3) : capture le momentum court terme
    # ST medium (14,2) : tendance intermédiaire
    # ST slow (21,1) : tendance longue
    for length, mult in [(7, 3.0), (14, 2.0), (21, 1.0)]:
        st = ta.supertrend(df["high"], df["low"], df["close"], length=length, multiplier=mult)
        # direction : 1 = haussier (prix > ST line), -1 = baissier
        dir_col = f"SUPERTd_{length}_{mult}"
        if dir_col in st.columns:
            df[f"st_dir_{length}"] = st[dir_col]
        else:
            # fallback si la colonne a un nom légèrement différent
            dcols = [c for c in st.columns if c.startswith("SUPERTd")]
            df[f"st_dir_{length}"] = st[dcols[0]] if dcols else float("nan")

    return df


# ---------------------------------------------------------------------------
# Conditions BUY — logique état : vraies TANT QUE l'indicateur est dans sa zone
# ---------------------------------------------------------------------------

def cond_price_drop_buy(df: pd.DataFrame) -> bool:
    """Condition 1 BUY : Le prix a baissé de plus de X% sur la dernière bougie."""
    if len(df) < 2:
        return False
    prev_close = df["close"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    change_pct = (curr_close - prev_close) / prev_close * 100
    return change_pct < -config.PRICE_CHANGE_THRESHOLD_PCT


def cond_rsi_buy(df: pd.DataFrame) -> bool:
    """Condition 2 BUY : RSI < 30 — zone de survente active."""
    rsi = df["rsi"].iloc[-1]
    return pd.notna(rsi) and rsi < config.RSI_OVERSOLD


def cond_sma_buy(df: pd.DataFrame) -> bool:
    """
    Condition 3 BUY : SMA20 < SMA50 (tendance baissière en cours).
    On cherche à acheter dans une zone de faiblesse pour profiter du rebond.
    """
    fast = df["sma_fast"].iloc[-1]
    slow = df["sma_slow"].iloc[-1]
    if any(pd.isna(v) for v in [fast, slow]):
        return False
    return fast < slow


def cond_macd_buy(df: pd.DataFrame) -> bool:
    """
    Condition 4 BUY : MACD < Signal (momentum baissier = pression vendeuse forte).
    Combiné aux autres indicateurs, indique une survente.
    """
    macd = df["macd"].iloc[-1]
    sig = df["macd_signal"].iloc[-1]
    if any(pd.isna(v) for v in [macd, sig]):
        return False
    return macd < sig


def cond_stoch_buy(df: pd.DataFrame) -> bool:
    """Condition 5 BUY : Stochastique K < 20 — zone de survente active."""
    k = df["stoch_k"].iloc[-1]
    return pd.notna(k) and k < config.STOCH_OVERSOLD


def cond_bollinger_buy(df: pd.DataFrame) -> bool:
    """
    Condition 6 BUY : Le prix est sous la bande inférieure de Bollinger.
    Indique que le prix est statistiquement bas (>2 écarts-types sous la moyenne).
    """
    curr_close = df["close"].iloc[-1]
    curr_lower = df["bb_lower"].iloc[-1]
    if pd.isna(curr_lower):
        return False
    return curr_close < curr_lower


def cond_volume_buy(df: pd.DataFrame) -> bool:
    """
    Condition 7 BUY : Volume > SMA(20) du volume.
    Une chute de prix sur volume élevé = panique vendeuse = bon point d'entrée mean-reversion.
    Confirme que le signal n'est pas un faux mouvement sans conviction.
    """
    vol     = df["volume"].iloc[-1]
    vol_sma = df["volume_sma"].iloc[-1]
    if pd.isna(vol_sma) or vol_sma == 0:
        return False
    return vol > vol_sma


def cond_volume_sell(df: pd.DataFrame) -> bool:
    """
    Condition 7 SELL : Volume > SMA(20) du volume.
    Une hausse de prix sur volume élevé = euphorie acheteuse = bon point de sortie.
    """
    vol     = df["volume"].iloc[-1]
    vol_sma = df["volume_sma"].iloc[-1]
    if pd.isna(vol_sma) or vol_sma == 0:
        return False
    return vol > vol_sma


def cond_obv_buy(df: pd.DataFrame) -> bool:
    """
    Condition 7 BUY : OBV au-dessus de sa SMA(20).
    Indique que le volume accompagne une accumulation → la pression acheteuse
    est structurellement présente même si le prix est en survente temporaire.
    """
    obv = df["obv"].iloc[-1]
    obv_sma = df["obv_sma"].iloc[-1]
    if any(pd.isna(v) for v in [obv, obv_sma]):
        return False
    return obv > obv_sma


def cond_triple_st_buy(df: pd.DataFrame) -> bool:
    """
    Condition 8 BUY : Triple SuperTrend — signal mean-reversion.
    ST fast (7) baissier (dip en cours) ET ST slow (21) haussier (uptrend de fond).
    Capture exactement les dips dans un marché globalement haussier.
    """
    fast = df["st_dir_7"].iloc[-1]
    slow = df["st_dir_21"].iloc[-1]
    if any(pd.isna(v) for v in [fast, slow]):
        return False
    return fast == -1 and slow == 1  # dip temporaire dans uptrend long


def cond_triple_st_sell(df: pd.DataFrame) -> bool:
    """
    Condition 8 SELL : ST fast (7) redevenu haussier = fin du rebond court terme.
    Le momentum court terme a retourné → bon moment de sortir.
    """
    fast = df["st_dir_7"].iloc[-1]
    if pd.isna(fast):
        return False
    return fast == 1


# ---------------------------------------------------------------------------
# Conditions SELL — logique état : vraies TANT QUE l'indicateur est dans sa zone
# ---------------------------------------------------------------------------

def cond_price_rise_sell(df: pd.DataFrame) -> bool:
    """Condition 1 SELL : Le prix a monté de plus de X% sur la dernière bougie."""
    if len(df) < 2:
        return False
    prev_close = df["close"].iloc[-2]
    curr_close = df["close"].iloc[-1]
    change_pct = (curr_close - prev_close) / prev_close * 100
    return change_pct > config.PRICE_CHANGE_THRESHOLD_PCT


def cond_rsi_sell(df: pd.DataFrame) -> bool:
    """Condition 2 SELL : RSI > 70 — zone de surachat active."""
    rsi = df["rsi"].iloc[-1]
    return pd.notna(rsi) and rsi > config.RSI_OVERBOUGHT


def cond_sma_sell(df: pd.DataFrame) -> bool:
    """
    Condition 3 SELL : SMA20 > SMA50 (tendance haussière en cours).
    On cherche à vendre dans une zone de force pour sécuriser les gains.
    """
    fast = df["sma_fast"].iloc[-1]
    slow = df["sma_slow"].iloc[-1]
    if any(pd.isna(v) for v in [fast, slow]):
        return False
    return fast > slow


def cond_macd_sell(df: pd.DataFrame) -> bool:
    """
    Condition 4 SELL : MACD > Signal (momentum haussier fort = surachat potentiel).
    """
    macd = df["macd"].iloc[-1]
    sig = df["macd_signal"].iloc[-1]
    if any(pd.isna(v) for v in [macd, sig]):
        return False
    return macd > sig


def cond_stoch_sell(df: pd.DataFrame) -> bool:
    """Condition 5 SELL : Stochastique K > 80 — zone de surachat active."""
    k = df["stoch_k"].iloc[-1]
    return pd.notna(k) and k > config.STOCH_OVERBOUGHT


def cond_bollinger_sell(df: pd.DataFrame) -> bool:
    """
    Condition 6 SELL : Le prix est au-dessus de la bande supérieure de Bollinger.
    Indique que le prix est statistiquement haut (>2 écarts-types au-dessus de la moyenne).
    """
    curr_close = df["close"].iloc[-1]
    curr_upper = df["bb_upper"].iloc[-1]
    if pd.isna(curr_upper):
        return False
    return curr_close > curr_upper


def cond_obv_sell(df: pd.DataFrame) -> bool:
    """
    Condition 7 SELL : OBV sous sa SMA(20).
    Indique que le volume accompagne une distribution → la pression vendeuse
    est structurellement présente même si le prix est encore haut.
    """
    obv = df["obv"].iloc[-1]
    obv_sma = df["obv_sma"].iloc[-1]
    if any(pd.isna(v) for v in [obv, obv_sma]):
        return False
    return obv < obv_sma


# ---------------------------------------------------------------------------
# Scoring global
# ---------------------------------------------------------------------------

BUY_CONDITIONS = [
    ("Prix en baisse",     cond_price_drop_buy),
    ("RSI < 30",           cond_rsi_buy),
    ("SMA20 < SMA50",      cond_sma_buy),
    ("MACD < Signal",      cond_macd_buy),
    ("Stoch < 20",         cond_stoch_buy),
    ("Prix < BB basse",    cond_bollinger_buy),
    ("Volume > Vol_SMA",   cond_volume_buy),
    ("Triple ST dip",      cond_triple_st_buy),
]

SELL_CONDITIONS = [
    ("Prix en hausse",     cond_price_rise_sell),
    ("RSI > 70",           cond_rsi_sell),
    ("SMA20 > SMA50",      cond_sma_sell),
    ("MACD > Signal",      cond_macd_sell),
    ("Stoch > 80",         cond_stoch_sell),
    ("Prix > BB haute",    cond_bollinger_sell),
    ("Volume > Vol_SMA",   cond_volume_sell),
    ("Triple ST rebond",   cond_triple_st_sell),
]


def vectorized_signals(df: pd.DataFrame, use_triple_st: bool = True,
                       use_sma_macd: bool = True,
                       use_regime_filter: bool = False) -> pd.Series:
    """
    Calcule le signal (BUY / SELL / HOLD) pour chaque bougie du DataFrame en une passe vectorisée.
    Beaucoup plus rapide que d'appeler score_signal() dans une boucle.
    Retourne une Series de strings indexée comme df.
    """
    import numpy as np

    close  = df["close"]
    prev   = close.shift(1)
    chg    = (close - prev) / prev * 100

    # --- BUY conditions (vectorisées) ---
    b1 = chg < -config.PRICE_CHANGE_THRESHOLD_PCT                        # prix en baisse
    b2 = df["rsi"].notna() & (df["rsi"] < config.RSI_OVERSOLD)           # RSI < 30
    if use_sma_macd:
        b3 = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] < df["sma_slow"])
        b4 = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] < df["macd_signal"])
    else:
        b3 = pd.Series(False, index=df.index)
        b4 = pd.Series(False, index=df.index)
    b5 = df["stoch_k"].notna() & (df["stoch_k"] < config.STOCH_OVERSOLD) # Stoch < 20
    b6 = df["bb_lower"].notna() & (close < df["bb_lower"])               # prix < BB basse
    b7 = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])  # volume spike
    # Triple ST : fast ST baissier (dip) + slow ST haussier (uptrend de fond)
    if use_triple_st:
        b8 = (df["st_dir_7"].fillna(0) == -1) & (df["st_dir_21"].fillna(0) == 1)
    else:
        b8 = pd.Series(False, index=df.index)

    buy_score = b1.astype(int) + b2.astype(int) + b3.astype(int) + b4.astype(int) + \
                b5.astype(int) + b6.astype(int) + b7.astype(int) + b8.astype(int)

    # --- SELL conditions (vectorisées) ---
    s1 = chg > config.PRICE_CHANGE_THRESHOLD_PCT
    s2 = df["rsi"].notna() & (df["rsi"] > config.RSI_OVERBOUGHT)
    if use_sma_macd:
        s3 = df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] > df["sma_slow"])
        s4 = df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] > df["macd_signal"])
    else:
        s3 = pd.Series(False, index=df.index)
        s4 = pd.Series(False, index=df.index)
    s5 = df["stoch_k"].notna() & (df["stoch_k"] > config.STOCH_OVERBOUGHT)
    s6 = df["bb_upper"].notna() & (close > df["bb_upper"])
    s7 = df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"])
    # Triple ST SELL : fast ST redevenu haussier = fin du rebond
    if use_triple_st:
        s8 = (df["st_dir_7"].fillna(0) == 1)
    else:
        s8 = pd.Series(False, index=df.index)

    sell_score = s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int) + \
                 s5.astype(int) + s6.astype(int) + s7.astype(int) + s8.astype(int)

    # Filtre de tendance SMA200
    in_uptrend = df["sma200"].isna() | (close > df["sma200"])

    min_score = config.MIN_SCORE_TO_TRADE
    is_buy  = in_uptrend & (buy_score >= min_score) & (buy_score > sell_score)
    is_sell = (sell_score >= min_score) & (sell_score > buy_score)

    # Filtre régime ADX : bloque les BUY quand ADX > 25 (marché directionnel)
    # La mean-reversion est moins efficace en tendance forte
    if use_regime_filter and "adx" in df.columns:
        in_ranging = df["adx"].isna() | (df["adx"] <= 25)
        is_buy = is_buy & in_ranging

    signals = pd.Series("HOLD", index=df.index, dtype=object)
    signals[is_buy]  = "BUY"
    signals[is_sell] = "SELL"
    # BUY prend priorité si les deux sont vrais
    signals[is_buy & is_sell] = "BUY"
    return signals


def score_signal(df: pd.DataFrame,
                 use_sma_macd: bool = True,
                 use_triple_st: bool = True) -> dict:
    """
    Calcule le score BUY et SELL sur la dernière bougie.
    Filtre de tendance SMA200 : bloque tout BUY si prix < SMA200 (downtrend).

    use_sma_macd  : inclure b3 (SMA20<SMA50) et b4 (MACD<Signal)
    use_triple_st : inclure b8 (Triple SuperTrend)
    """
    skip_buy  = set()
    skip_sell = set()
    if not use_sma_macd:
        skip_buy.update({"SMA20 < SMA50", "MACD < Signal"})
        skip_sell.update({"SMA20 > SMA50", "MACD > Signal"})
    if not use_triple_st:
        skip_buy.add("Triple ST dip")
        skip_sell.add("Triple ST rebond")

    buy_details  = {name: fn(df) for name, fn in BUY_CONDITIONS  if name not in skip_buy}
    sell_details = {name: fn(df) for name, fn in SELL_CONDITIONS if name not in skip_sell}

    buy_score = sum(buy_details.values())
    sell_score = sum(sell_details.values())

    # Filtre de tendance long terme — veto dur sur les BUY en downtrend
    price   = df["close"].iloc[-1]
    sma200  = df["sma200"].iloc[-1]
    in_uptrend = pd.isna(sma200) or price > sma200  # isna = pas encore calculé, on laisse passer

    signal = "HOLD"
    if in_uptrend and buy_score >= config.MIN_SCORE_TO_TRADE and buy_score > sell_score:
        signal = "BUY"
    elif sell_score >= config.MIN_SCORE_TO_TRADE and sell_score > buy_score:
        signal = "SELL"

    return {
        "signal": signal,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "buy_details": buy_details,
        "sell_details": sell_details,
        "price": price,
        "rsi": df["rsi"].iloc[-1],
        "in_uptrend": in_uptrend,
        "sma200": sma200,
    }
