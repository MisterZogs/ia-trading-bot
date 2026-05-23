"""
Configuration centrale du bot de trading.
Modifie ces valeurs pour adapter la stratégie.
"""

# --- Actifs tradés — 20 cryptos × 5% = 100% du capital potentiellement déployé ---
SYMBOLS = [
    "BTC/USDT",   # Bitcoin
    "ETH/USDT",   # Ethereum
    "BNB/USDT",   # BNB
    "SOL/USDT",   # Solana
    "XRP/USDT",   # Ripple
    "ADA/USDT",   # Cardano
    "AVAX/USDT",  # Avalanche
    "DOT/USDT",   # Polkadot
    "LINK/USDT",  # Chainlink
    "MATIC/USDT", # Polygon
    "UNI/USDT",   # Uniswap
    "ATOM/USDT",  # Cosmos
    "NEAR/USDT",  # NEAR Protocol
    "LTC/USDT",   # Litecoin
    "DOGE/USDT",  # Dogecoin
    "TRX/USDT",   # Tron
    "ALGO/USDT",  # Algorand
    "AAVE/USDT",  # Aave
    "ARB/USDT",   # Arbitrum
    "OP/USDT",    # Optimism
]

# --- Timeframe par défaut ---
TIMEFRAME = "6h"

# Timeframe par symbole — optimisé sur BTC/ETH, appliqué aux autres par défaut
SYMBOL_TIMEFRAMES = {
    "BTC/USDT": "12h",
    "ETH/USDT": "6h",
    # Les 18 autres utilisent TIMEFRAME (6h) par défaut
}

# Stop-Loss et Take-Profit par symbole — optimisés sur 4 ans avec simulation high/low + SMA200
SYMBOL_RISK = {
    "BTC/USDT":  {"sl": 0.05, "tp": 0.07},
    "ETH/USDT":  {"sl": 0.20, "tp": 0.25},
    "BNB/USDT":  {"sl": 0.10, "tp": 0.20},
    "SOL/USDT":  {"sl": 0.05, "tp": 0.25},
    "XRP/USDT":  {"sl": 0.15, "tp": 0.20},
    "ADA/USDT":  {"sl": 0.07, "tp": 0.25},
    "AVAX/USDT": {"sl": 0.07, "tp": 0.30},
    "DOT/USDT":  {"sl": 0.10, "tp": 0.25},
    "LINK/USDT": {"sl": 0.05, "tp": 0.30},
    "MATIC/USDT":{"sl": 0.07, "tp": 0.30},
    "UNI/USDT":  {"sl": 0.07, "tp": 0.30},
    "ATOM/USDT": {"sl": 0.05, "tp": 0.25},
    "NEAR/USDT": {"sl": 0.07, "tp": 0.20},
    "LTC/USDT":  {"sl": 0.20, "tp": 0.25},
    "DOGE/USDT": {"sl": 0.07, "tp": 0.10},
    "TRX/USDT":  {"sl": 0.05, "tp": 0.07},
    "ALGO/USDT": {"sl": 0.05, "tp": 0.15},
    "AAVE/USDT": {"sl": 0.15, "tp": 0.20},
    "ARB/USDT":  {"sl": 0.10, "tp": 0.25},
    "OP/USDT":   {"sl": 0.10, "tp": 0.30},
}

# --- Stratégie : seuil de scoring ---
# 3/6 conditions minimum pour déclencher un signal (validé par backtest)
MIN_SCORE_TO_TRADE = 3

# --- Gestion du capital ---
# 5% du capital par trade (validé par backtest)
POSITION_SIZE_PCT = 0.05

# Exposition maximale par crypto (% du capital total)
# Max 1 position par symbole, 20 symboles × 5% = 100% max déployé
MAX_EXPOSURE_PER_SYMBOL = 0.05

# --- Risk Management (valeurs par défaut si symbole absent de SYMBOL_RISK) ---
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.10

# --- Paramètres des indicateurs ---
# SMA de tendance long terme — filtre anti-krach
# BUY bloqué si prix < SMA_TREND (marché en downtrend)
SMA_TREND = 200

RSI_PERIOD = 14
RSI_OVERSOLD = 30         # En dessous = survente (signal BUY)
RSI_OVERBOUGHT = 70       # Au dessus = surachat (signal SELL)

SMA_FAST = 20
SMA_SLOW = 50

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

STOCH_K = 14
STOCH_D = 3
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80

BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0

# Variation de prix minimum pour déclencher le filtre de momentum (algo 1)
PRICE_CHANGE_THRESHOLD_PCT = 2.0

# --- ATR : sizing adaptatif à la volatilité ---
# La taille de position est scalée selon ATR/prix vs une référence.
# Marché calme (ATR faible) → position plus grande.
# Marché volatile (ATR élevé) → position réduite.
ATR_SIZING_REF_PCT  = 0.020   # ATR/prix de référence (2% = typique 12h BTC)
ATR_SIZING_MIN_MULT = 0.40    # plancher : 40% de la taille de base (forte volatilité)
ATR_SIZING_MAX_MULT = 2.00    # plafond  : 200% de la taille de base (très calme)

# --- ATR : take-profit dynamique ---
# TP = prix_entrée + ATR_TP_MULT × ATR  (s'adapte à la volatilité courante)
# Évite un TP trop proche en marché volatile (sorties prématurées).
ATR_TP_MULT    = 2.5    # TP = entrée + 2.5 × ATR
ATR_TP_MIN_PCT = 0.03   # TP minimum garanti = +3% (si ATR anormalement faible)

# --- Boucle principale ---
# Intervalle de vérification en secondes (doit correspondre au timeframe)
LOOP_INTERVAL_SECONDS = 60  # Vérifie chaque minute si une nouvelle bougie est fermée

# Nombre de bougies historiques à charger pour les indicateurs
CANDLES_LIMIT = 200
