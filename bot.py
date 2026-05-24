"""
Bot de trading live — corrigé et aligné avec le backtest.

Corrections vs version précédente :
  1. Bougie en cours de formation exclue (df.iloc[:-1])
  2. Config épurée respectée (use_sma_macd=False, use_triple_st=False)
  3. Multi-position supporté (comme sim_multi du backtest)
  4. Fear & Greed veto actif (comme dans le backtest)
  5. Fees inclus dans le PnL (0.1% achat + 0.1% vente)
  6. SansSL par défaut (meilleure config validée)
  7. Retry automatique sur erreurs réseau
  8. Shutdown propre sur CTRL+C

Usage:
    python main.py           # Paper trading
    python main.py --live    # Live (nécessite .env avec API_KEY / API_SECRET)
"""

import os
import json
import time
import signal
import ccxt
import requests
import pandas as pd
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from colorama import Fore, Style, init
from dotenv import load_dotenv

import config
import indicators
import data_cache
import fear_greed as fg_module
import multi_sim as ms
from risk_manager import RiskManager

init(autoreset=True)
load_dotenv()

# ---------------------------------------------------------------------------
# Stratégie active : Top 20 / multi / sansSL / 30m / +TripleST / pondéré-strict
# Backtest 2023 : +54% | Win 31% | DD 29% | Excel row 1623
# ---------------------------------------------------------------------------
USE_SMA_MACD  = True    # +TripleST : filtre SMA/MACD actif
USE_TRIPLE_ST = True    # +TripleST : filtre Triple SuperTrend actif
USE_SL        = False   # sansSL : sortie sur signal SELL ou TP uniquement

LIVE_SYMBOLS  = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOT/USDT", "LINK/USDT", "MATIC/USDT",
    "UNI/USDT", "ATOM/USDT", "NEAR/USDT", "LTC/USDT", "DOGE/USDT",
    "TRX/USDT", "ALGO/USDT", "AAVE/USDT", "ARB/USDT", "OP/USDT",
]
LIVE_TIMEFRAME = "30m"
LIVE_POS_PCT  = 0.10    # fallback si poids non dispo
LIVE_MAX_POS  = 10      # 10 positions simultanées max

PARIS_TZ           = ZoneInfo("Europe/Paris")
DAILY_SUMMARY_HOUR = 9   # heure française (CET/CEST)

MAX_RETRIES   = 3       # tentatives sur erreur réseau
RETRY_DELAY   = 5       # secondes entre tentatives
TRADES_LOG    = "trades_log.jsonl"  # une ligne JSON par ouverture/fermeture
BOT_LOG       = "bot.log"

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


def _compute_weights(symbols, tf, use_triple_st, use_sma_macd, fg):
    """Calcule les poids de régularité pondéré-strict (floor=0) pour chaque symbole."""
    SCORE_YEARS = list(range(2018, 2025))
    import data_cache as _dc
    min_y = 2 if tf in ("30m", "1h") else 4
    scores = {}
    for sym in symbols:
        rets = []
        for year in SCORE_YEARS:
            df_y = ms.get_df_for_year(sym, tf, year)
            if df_y is None or len(df_y) < 10:
                continue
            r = ms.sim_multi_on_dfs(
                {sym: df_y}, use_sl=False, fg=fg,
                use_triple_st=use_triple_st, use_sma_macd=use_sma_macd, tf=tf
            )
            ret = r.get("return_%")
            if ret is not None:
                rets.append(ret)
        if len(rets) >= min_y:
            moy = sum(rets) / len(rets)
            pct_pos = sum(1 for v in rets if v > 0) / len(rets)
            scores[sym] = moy * (pct_pos ** 2)
        else:
            scores[sym] = 0.0
    scores_pos = {s: max(scores.get(s, 0.0), 0.0) for s in symbols}
    total = sum(scores_pos.values())
    if total == 0:
        return {s: 1.0 / len(symbols) for s in symbols}
    return {s: scores_pos[s] / total for s in symbols}


class TradingBot:
    def __init__(self, initial_capital: float = 1000.0):
        self.exchange      = self._init_exchange()
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.risk          = RiskManager(initial_capital, use_sl=USE_SL,
                                        pos_pct=LIVE_POS_PCT, max_positions=LIVE_MAX_POS)
        self.trade_log:  list[dict] = []
        self._last_candle: dict[str, int] = {}
        self._running      = True
        self._fg_cache:    dict = {}          # {date -> int}
        self._last_daily_summary: date | None = None
        self._price_cache: dict[str, float] = {}  # pour unrealized PnL
        self._symbol_weights: dict[str, float] = {}  # poids pondéré-strict

        # Chargement du Fear & Greed au démarrage
        try:
            self._fg_cache = fg_module.load(verbose=False)
        except Exception as e:
            self._log(f"Fear & Greed non chargé ({e}) — veto désactivé", Fore.YELLOW)

        # Calcul des poids de régularité (pondéré-strict)
        try:
            self._log("Calcul des poids pondéré-strict...", Fore.CYAN)
            data_cache.prefetch_all(LIVE_SYMBOLS, [LIVE_TIMEFRAME], verbose=False)
            for sym in LIVE_SYMBOLS:
                ms.get_df(sym, LIVE_TIMEFRAME)
            self._symbol_weights = _compute_weights(
                LIVE_SYMBOLS, LIVE_TIMEFRAME,
                use_triple_st=USE_TRIPLE_ST, use_sma_macd=USE_SMA_MACD,
                fg=self._fg_cache,
            )
            self._log(
                "Poids calculés : " + " | ".join(
                    f"{s.replace('/USDT','')}: {w*100:.1f}%"
                    for s, w in sorted(self._symbol_weights.items(),
                                       key=lambda x: -x[1])[:5]
                ) + " ...",
                Fore.CYAN
            )
        except Exception as e:
            self._log(f"Poids non calculés ({e}) — fallback 10%/pos", Fore.YELLOW)

        # Shutdown propre sur CTRL+C
        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        mode = "PAPER TRADING" if self.paper_trading else "LIVE TRADING"
        self._log(f"Bot démarré — {mode} | Capital: {initial_capital:.2f} USDT | "
                  f"Stratégie: Top20 / +TripleST / {LIVE_TIMEFRAME} / "
                  f"{'sansSL' if not USE_SL else 'avecSL'} / pondéré-strict",
                  Fore.CYAN)
        self._log(f"Symboles : {LIVE_SYMBOLS} | "
                  f"Fear&Greed veto > {fg_module.FG_GREED_VETO}", Fore.CYAN)
        self._telegram(
            f"[{mode}] Bot démarré\n"
            f"Capital: {initial_capital:.0f} USDT\n"
            f"Stratégie: Top20 / +TripleST / {LIVE_TIMEFRAME} / sansSL / pondéré-strict\n"
            f"Symboles: {', '.join(s.replace('/USDT','') for s in LIVE_SYMBOLS)}"
        )

    # ------------------------------------------------------------------ #
    # Initialisation exchange
    # ------------------------------------------------------------------ #
    def _init_exchange(self) -> ccxt.Exchange:
        exchange_id    = os.getenv("EXCHANGE", "binance")
        exchange_class = getattr(ccxt, exchange_id)
        return exchange_class({
            "apiKey":          os.getenv("API_KEY", ""),
            "secret":          os.getenv("API_SECRET", ""),
            "enableRateLimit": True,
            "options":         {"defaultType": "spot"},
        })

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    def _log(self, msg: str, color: str = Style.RESET_ALL):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(f"{color}{line}{Style.RESET_ALL}", flush=True)
        with open(BOT_LOG, "a", buffering=1) as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ #
    # Shutdown propre
    # ------------------------------------------------------------------ #
    def _handle_shutdown(self, *_):
        self._log("Arrêt demandé — fin de la boucle en cours...", Fore.YELLOW)
        self._running = False

    # ------------------------------------------------------------------ #
    # Export position vers fichier JSONL
    # ------------------------------------------------------------------ #
    def _export(self, event: dict):
        import numpy as np
        event["ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(TRADES_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o) + "\n")

    # ------------------------------------------------------------------ #
    # Notifications Telegram
    # ------------------------------------------------------------------ #
    def _telegram(self, msg: str):
        """Envoie un message texte sur le channel Telegram configuré."""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            return
        self._log(f"[TELEGRAM] {msg[:60].replace(chr(10),' | ')}", Fore.MAGENTA)
        try:
            requests.get(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                params={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
                timeout=5,
            )
        except Exception:
            pass  # ne pas planter le bot si Telegram est indisponible

    # ------------------------------------------------------------------ #
    # Fetch OHLCV avec retry
    # ------------------------------------------------------------------ #
    def fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        tf = LIVE_TIMEFRAME
        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                raw = self.exchange.fetch_ohlcv(symbol, tf, limit=config.CANDLES_LIMIT + 1)
                df  = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                return df
            except Exception as e:
                last_exc = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
        raise last_exc

    def _is_new_candle(self, symbol: str, df: pd.DataFrame) -> bool:
        """True si la dernière bougie FERMÉE est nouvelle depuis le dernier check."""
        last_ts = int(df.iloc[-1]["timestamp"].timestamp())
        if self._last_candle.get(symbol) == last_ts:
            return False
        self._last_candle[symbol] = last_ts
        return True

    # ------------------------------------------------------------------ #
    # Fear & Greed
    # ------------------------------------------------------------------ #
    def _fg_today(self) -> int | None:
        return self._fg_cache.get(date.today())

    # ------------------------------------------------------------------ #
    # Exécution des ordres
    # ------------------------------------------------------------------ #
    def _execute_buy(self, symbol: str, price: float, score: dict):
        sym_pct = self._symbol_weights.get(symbol) or None
        pos = self.risk.open_position(symbol, price, pos_pct=sym_pct)
        if pos is None:
            self._log(f"[{symbol}] BUY refusé — capital insuffisant "
                      f"(disponible: {self.risk._available_capital():.2f} USDT)", Fore.YELLOW)
            return

        if not self.paper_trading:
            try:
                self.exchange.create_market_buy_order(symbol, pos.size)
            except Exception as e:
                # Annuler la position en mémoire si l'ordre échoue
                self.risk.close_position(symbol, pos, price)
                self._log(f"[{symbol}] Erreur ordre BUY: {e}", Fore.RED)
                return

        n_open = len(self.risk.positions.get(symbol, []))
        tp_str = f"TP: {pos.take_profit:.4f}"
        sl_str = f" | SL: {pos.stop_loss:.4f}" if pos.stop_loss > 0 else ""
        self._log(
            f"[{symbol}] BUY #{n_open} | Prix: {price:.4f} | "
            f"Taille: {pos.size:.6f} | Score: {score['buy_score']} | "
            f"{tp_str}{sl_str}",
            Fore.GREEN,
        )
        self._export({
            "event":       "OPEN",
            "symbol":      symbol,
            "side":        "BUY",
            "price":       round(price, 8),
            "size":        round(pos.size, 8),
            "cost_usdt":   round(pos.cost_usdt, 4),
            "take_profit": round(pos.take_profit, 8),
            "stop_loss":   round(pos.stop_loss, 8),
            "buy_score":   score.get("buy_score"),
            "fg_val":      self._fg_today(),
            "paper":       self.paper_trading,
        })
        mode_tag = "[PAPER]" if self.paper_trading else "[LIVE]"
        self._telegram(
            f"{mode_tag} BUY {symbol}\n"
            f"Prix: {price:.4f} | Taille: {pos.size:.6f}\n"
            f"Cout: {pos.cost_usdt:.2f} USDT | TP: {pos.take_profit:.4f}\n"
            f"Score: {score.get('buy_score')} | F&G: {self._fg_today()}"
        )

    def _execute_sell(self, symbol: str, pos, price: float, reason: str):
        if not self.paper_trading:
            try:
                self.exchange.create_market_sell_order(symbol, pos.size)
            except Exception as e:
                self._log(f"[{symbol}] Erreur ordre SELL: {e}", Fore.RED)
                return

        result = self.risk.close_position(symbol, pos, price)
        self.trade_log.append(result)

        color = Fore.GREEN if result["pnl_usdt"] >= 0 else Fore.RED
        self._log(
            f"[{symbol}] SELL ({reason}) | Prix: {price:.4f} | "
            f"PnL: {result['pnl_usdt']:+.4f} USDT ({result['pnl_pct']:+.2f}%) | "
            f"Capital: {self.risk.total_capital:.2f} USDT",
            color,
        )
        duration_min = round((datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60, 1)
        self._export({
            "event":        "CLOSE",
            "symbol":       symbol,
            "side":         "SELL",
            "reason":       reason,
            "price":        round(price, 8),
            "size":         round(pos.size, 8),
            "pnl_usdt":     result["pnl_usdt"],
            "pnl_pct":      result["pnl_pct"],
            "duration_min": duration_min,
            "paper":        self.paper_trading,
        })
        mode_tag = "[PAPER]" if self.paper_trading else "[LIVE]"
        pnl_sign  = "+" if result["pnl_usdt"] >= 0 else ""
        self._telegram(
            f"{mode_tag} SELL {symbol} ({reason})\n"
            f"Prix: {price:.4f} | Duree: {duration_min:.0f}min\n"
            f"PnL: {pnl_sign}{result['pnl_usdt']:.4f} USDT ({pnl_sign}{result['pnl_pct']:.2f}%)\n"
            f"Capital: {self.risk.total_capital:.2f} USDT"
        )

    def _execute_sell_all(self, symbol: str, price: float, reason: str):
        """Ferme toutes les positions d'un symbole sur signal SELL."""
        if symbol not in self.risk.positions:
            return
        if not self.paper_trading:
            total_size = sum(p.size for p in self.risk.positions[symbol])
            try:
                self.exchange.create_market_sell_order(symbol, total_size)
            except Exception as e:
                self._log(f"[{symbol}] Erreur ordre SELL all: {e}", Fore.RED)
                return

        positions_before = list(self.risk.positions.get(symbol, []))
        results = self.risk.close_all_positions(symbol, price)
        for pos, result in zip(positions_before, results):
            self.trade_log.append(result)
            color = Fore.GREEN if result["pnl_usdt"] >= 0 else Fore.RED
            self._log(
                f"[{symbol}] SELL ({reason}) | Prix: {price:.4f} | "
                f"PnL: {result['pnl_usdt']:+.4f} USDT ({result['pnl_pct']:+.2f}%) | "
                f"Capital: {self.risk.total_capital:.2f} USDT",
                color,
            )
            duration_min = round((datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 60, 1)
            self._export({
                "event":        "CLOSE",
                "symbol":       symbol,
                "side":         "SELL",
                "reason":       reason,
                "price":        round(price, 8),
                "size":         round(result["size"], 8),
                "pnl_usdt":     result["pnl_usdt"],
                "pnl_pct":      result["pnl_pct"],
                "duration_min": duration_min,
                "paper":        self.paper_trading,
            })

    # ------------------------------------------------------------------ #
    # Traitement d'un symbole
    # ------------------------------------------------------------------ #
    def process_symbol(self, symbol: str):
        """Fetch → indicateurs → signal → action."""
        try:
            df_raw = self.fetch_ohlcv(symbol)
        except Exception as e:
            self._log(f"[{symbol}] Erreur fetch OHLCV: {e}", Fore.RED)
            return

        # --- BUG FIX 1 : exclure la bougie en cours de formation ---
        # La dernière bougie retournée par l'API est incomplète (en cours).
        # On travaille toujours sur la dernière bougie FERMÉE.
        df_closed = df_raw.iloc[:-1].reset_index(drop=True)

        if not self._is_new_candle(symbol, df_closed):
            return   # pas de nouvelle bougie fermée depuis le dernier check

        # --- Calcul des indicateurs ---
        df_ind = indicators.compute_all(df_closed)
        df_ind = df_ind.dropna().reset_index(drop=True)
        if len(df_ind) < 5:
            return

        # --- BUG FIX 2 : config épurée (use_sma_macd=False, use_triple_st=False) ---
        score = indicators.score_signal(
            df_ind,
            use_sma_macd=USE_SMA_MACD,
            use_triple_st=USE_TRIPLE_ST,
        )
        price = score["price"]
        self._price_cache[symbol] = price

        # --- BUG FIX 3 : vérification SL/TP multi-positions ---
        exits = self.risk.check_exits(symbol, price)
        for pos, reason in exits:
            self._execute_sell(symbol, pos, price, reason)

        # --- BUG FIX 4 : Fear & Greed veto ---
        fg_val = self._fg_today()
        if score["signal"] == "BUY" and fg_val is not None:
            if fg_val > fg_module.FG_GREED_VETO:
                self._log(
                    f"[{symbol}] BUY bloqué — Fear&Greed={fg_val} "
                    f"(veto > {fg_module.FG_GREED_VETO})",
                    Fore.YELLOW,
                )
                return

        # --- Signaux ---
        if score["signal"] == "BUY":
            self._log(
                f"[{symbol}] Signal BUY | Score {score['buy_score']} | "
                f"RSI: {score['rsi']:.1f} | Prix: {price:.4f} | "
                f"SMA200: {'OK' if score['in_uptrend'] else 'NON'}",
                Fore.GREEN,
            )
            self._execute_buy(symbol, price, score)

        elif score["signal"] == "SELL" and symbol in self.risk.positions:
            self._log(
                f"[{symbol}] Signal SELL | Score {score['sell_score']} | "
                f"RSI: {score['rsi']:.1f} | Prix: {price:.4f}",
                Fore.RED,
            )
            self._execute_sell_all(symbol, price, reason="signal")

    # ------------------------------------------------------------------ #
    # Affichage du statut
    # ------------------------------------------------------------------ #
    def print_status(self):
        summary     = self.risk.summary()
        unreal_pnl  = self.risk.unrealized_pnl(self._price_cache)
        n_trades    = len(self.trade_log)
        total_pnl   = sum(t["pnl_usdt"] for t in self.trade_log)
        win_trades  = sum(1 for t in self.trade_log if t["pnl_usdt"] > 0)
        win_rate    = win_trades / n_trades * 100 if n_trades else 0

        self._log(
            f"── STATUS ── Capital: {summary['capital']:.2f} USDT | "
            f"Déployé: {summary['deployed']:.2f} | "
            f"Positions ouvertes: {summary['open_positions']} | "
            f"PnL non réalisé: {unreal_pnl:+.2f} USDT",
            Fore.CYAN,
        )
        if n_trades:
            self._log(
                f"── TRADES ── {n_trades} trades | "
                f"PnL réalisé: {total_pnl:+.2f} USDT | "
                f"Win rate: {win_rate:.1f}%",
                Fore.CYAN,
            )

        # Détail des positions ouvertes
        now = datetime.now(timezone.utc)
        for symbol, plist in summary["positions"].items():
            curr_price = self._price_cache.get(symbol)
            for pos in plist:
                unreal    = pos.unrealized_pnl(curr_price) if curr_price else 0
                color     = Fore.GREEN if unreal >= 0 else Fore.RED
                dur_secs  = (now - pos.entry_time).total_seconds()
                dur_h     = int(dur_secs // 3600)
                dur_m     = int((dur_secs % 3600) // 60)
                self._log(
                    f"   {symbol} | Entry: {pos.entry_price:.4f} | "
                    f"TP: {pos.take_profit:.4f} | "
                    f"Durée: {dur_h}h{dur_m:02d}m | "
                    f"PnL: {unreal:+.2f} USDT",
                    color,
                )

    # ------------------------------------------------------------------ #
    # Mode debug : simulation d'un signal entrant
    # ------------------------------------------------------------------ #
    def debug_signal(self, symbol: str, side: str):
        """Injecte un signal BUY ou SELL sur un symbole sans attendre une bougie."""
        side = side.upper()
        if side not in ("BUY", "SELL"):
            self._log(f"Side invalide : {side} (BUY ou SELL attendu)", Fore.RED)
            return

        self._log(f"[DEBUG] Fetch prix actuel pour {symbol}...", Fore.YELLOW)
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            price  = ticker["last"]
        except Exception as e:
            self._log(f"[DEBUG] Impossible de récupérer le prix : {e}", Fore.RED)
            return

        self._log(f"[DEBUG] Signal {side} simulé sur {symbol} @ {price:.6f}", Fore.YELLOW)

        if side == "BUY":
            fake_score = {"buy_score": 0, "rsi": 0.0, "in_uptrend": False, "price": price}
            self._execute_buy(symbol, price, fake_score)
        else:
            if symbol not in self.risk.positions:
                self._log(f"[DEBUG] Aucune position ouverte sur {symbol} à vendre", Fore.YELLOW)
            else:
                self._execute_sell_all(symbol, price, reason="debug")

        self.print_status()

    # ------------------------------------------------------------------ #
    # Status Telegram (toutes les 10 itérations ~10 min)
    # ------------------------------------------------------------------ #
    def _send_status_telegram(self):
        summary    = self.risk.summary()
        unreal_pnl = self.risk.unrealized_pnl(self._price_cache)
        n_trades   = len(self.trade_log)
        total_pnl  = sum(t["pnl_usdt"] for t in self.trade_log)
        win_rate   = (sum(1 for t in self.trade_log if t["pnl_usdt"] > 0)
                      / n_trades * 100) if n_trades else 0
        mode_tag   = "[PAPER]" if self.paper_trading else "[LIVE]"

        lines = [
            f"{mode_tag} STATUS",
            f"Capital: {summary['capital']:.2f} USDT | Deploye: {summary['deployed']:.2f}",
            f"Positions ouvertes: {summary['open_positions']}",
            f"PnL non realise: {unreal_pnl:+.2f} USDT",
        ]
        if n_trades:
            lines.append(f"Trades: {n_trades} | PnL realise: {total_pnl:+.2f} USDT | WR: {win_rate:.0f}%")

        self._telegram("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Résumé quotidien Telegram (9h heure française)
    # ------------------------------------------------------------------ #
    def _send_daily_summary_telegram(self):
        summary    = self.risk.summary()
        unreal_pnl = self.risk.unrealized_pnl(self._price_cache)
        n_trades   = len(self.trade_log)
        total_pnl  = sum(t["pnl_usdt"] for t in self.trade_log)
        win_trades = sum(1 for t in self.trade_log if t["pnl_usdt"] > 0)
        win_rate   = win_trades / n_trades * 100 if n_trades else 0
        mode_tag   = "[PAPER]" if self.paper_trading else "[LIVE]"
        today_str  = datetime.now(PARIS_TZ).strftime("%d/%m/%Y")

        lines = [
            f"{mode_tag} RESUME DU JOUR — {today_str}",
            f"",
            f"Capital: {summary['capital']:.2f} USDT",
            f"Deploye: {summary['deployed']:.2f} USDT",
            f"Positions ouvertes: {summary['open_positions']}",
            f"PnL non realise: {unreal_pnl:+.2f} USDT",
            f"",
            f"Depuis le demarrage:",
            f"Trades: {n_trades} | PnL realise: {total_pnl:+.2f} USDT | WR: {win_rate:.0f}%",
        ]

        for symbol, plist in summary["positions"].items():
            curr_price = self._price_cache.get(symbol)
            for pos in plist:
                unreal = pos.unrealized_pnl(curr_price) if curr_price else 0
                lines.append(f"  {symbol} entry={pos.entry_price:.4f} PnL={unreal:+.2f} USDT")

        self._telegram("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Boucle principale
    # ------------------------------------------------------------------ #
    def run(self):
        """Boucle principale : tourne jusqu'à CTRL+C."""
        self._log(
            f"Boucle démarrée | {len(LIVE_SYMBOLS)} symboles | "
            f"Intervalle: {config.LOOP_INTERVAL_SECONDS}s",
            Fore.CYAN,
        )
        iteration = 0
        while self._running:
            iteration += 1
            for symbol in LIVE_SYMBOLS:
                if not self._running:
                    break
                self.process_symbol(symbol)

            if iteration % 10 == 0:
                self.print_status()

            # Résumé quotidien à 9h heure française
            now_paris = datetime.now(PARIS_TZ)
            if (now_paris.hour == DAILY_SUMMARY_HOUR
                    and self._last_daily_summary != now_paris.date()):
                self._last_daily_summary = now_paris.date()
                daily_summary = self.risk.summary()
                if daily_summary["open_positions"] > 0 or len(self.trade_log) > 0:
                    self._send_daily_summary_telegram()

            # Attente courte avec vérification du flag d'arrêt
            for _ in range(config.LOOP_INTERVAL_SECONDS):
                if not self._running:
                    break
                time.sleep(1)

        self._log("Bot arrêté proprement.", Fore.YELLOW)
        self.print_status()
