"""
Gestion du risque : taille de position, stop-loss, take-profit, exposition.

Mode multi-position : plusieurs positions simultanées par symbole (comme le backtest).
Fees inclus dans le PnL (0.1% achat + 0.1% vente).
"""

from datetime import datetime, timezone

import config

FEE_RATE = 0.001   # 0.1% par côté (identique au backtest)


class Position:
    """Représente une position ouverte sur un actif."""

    def __init__(self, symbol: str, entry_price: float, size: float,
                 use_sl: bool = False, use_tp: bool = True):
        self.symbol      = symbol
        self.entry_price = entry_price
        self.size        = size
        self.entry_fee   = entry_price * size * FEE_RATE

        risk = config.SYMBOL_RISK.get(symbol, {
            "sl": config.STOP_LOSS_PCT,
            "tp": config.TAKE_PROFIT_PCT,
        })
        # SL : désactivé si use_sl=False (meilleure config validée = sansSL)
        self.stop_loss   = entry_price * (1 - risk["sl"]) if use_sl else 0.0
        self.take_profit = entry_price * (1 + risk["tp"])
        self.entry_time  = datetime.now(timezone.utc)

    @property
    def cost_usdt(self) -> float:
        """Capital immobilisé (valeur d'entrée)."""
        return self.entry_price * self.size

    def pnl(self, exit_price: float) -> float:
        """PnL net en USDT après fees d'entrée et de sortie."""
        gross    = (exit_price - self.entry_price) * self.size
        exit_fee = exit_price * self.size * FEE_RATE
        return gross - self.entry_fee - exit_fee

    def pnl_pct(self, exit_price: float) -> float:
        return self.pnl(exit_price) / self.cost_usdt * 100

    def should_stop_loss(self, price: float) -> bool:
        return self.stop_loss > 0 and price <= self.stop_loss

    def should_take_profit(self, price: float) -> bool:
        return price >= self.take_profit

    def unrealized_pnl(self, current_price: float) -> float:
        """PnL non réalisé (sans fee de sortie, pour affichage)."""
        return (current_price - self.entry_price) * self.size - self.entry_fee

    def __repr__(self):
        sl_str = f"SL: {self.stop_loss:.2f} | " if self.stop_loss > 0 else ""
        return (
            f"Position({self.symbol} | Entry: {self.entry_price:.4f} | "
            f"Size: {self.size:.6f} | {sl_str}TP: {self.take_profit:.4f})"
        )


class RiskManager:
    """
    Gère le risque global du portefeuille.
    Multi-position : plusieurs positions par symbole simultanément.
    """

    def __init__(self, total_capital: float,
                 use_sl: bool = False,
                 use_tp: bool = True,
                 pos_pct: float | None = None,
                 max_positions: int = 9999):
        self.total_capital = total_capital
        self.use_sl        = use_sl
        self.use_tp        = use_tp
        self.pos_pct       = pos_pct if pos_pct is not None else config.POSITION_SIZE_PCT
        self.max_positions = max_positions
        # dict {symbol: [Position, Position, ...]}
        self.positions: dict[str, list[Position]] = {}

    # ------------------------------------------------------------------ #
    # Capital et exposition
    # ------------------------------------------------------------------ #
    def _deployed_capital(self) -> float:
        """Capital total engagé dans des positions ouvertes."""
        return sum(
            p.cost_usdt
            for plist in self.positions.values()
            for p in plist
        )

    def _available_capital(self) -> float:
        return self.total_capital - self._deployed_capital()

    def _position_size_usdt(self) -> float:
        return self.total_capital * self.pos_pct

    def can_buy(self) -> bool:
        """True si capital suffisant ET limite de positions non atteinte."""
        if self.n_positions() >= self.max_positions:
            return False
        return self._available_capital() >= self._position_size_usdt()

    # ------------------------------------------------------------------ #
    # Ouverture de position
    # ------------------------------------------------------------------ #
    def open_position(self, symbol: str, price: float,
                      pos_pct: float | None = None) -> Position | None:
        """
        Ouvre une nouvelle position sur le symbole si le capital le permet.
        pos_pct : fraction du capital à allouer (override de self.pos_pct si fourni).
        Retourne la Position créée, ou None si refusé.
        """
        if not self.can_buy():
            return None

        pct      = pos_pct if pos_pct is not None else self.pos_pct
        pos_usdt = self.total_capital * pct
        size     = pos_usdt / price
        pos      = Position(symbol, price, size,
                            use_sl=self.use_sl, use_tp=self.use_tp)

        # Déduire les fees d'entrée du capital disponible
        self.total_capital -= pos.entry_fee

        if symbol not in self.positions:
            self.positions[symbol] = []
        self.positions[symbol].append(pos)
        return pos

    # ------------------------------------------------------------------ #
    # Vérification des exits (SL / TP)
    # ------------------------------------------------------------------ #
    def check_exits(self, symbol: str, price: float) -> list[tuple]:
        """
        Vérifie SL et TP pour toutes les positions du symbole.
        Retourne une liste de (Position, raison) pour les positions à fermer.
        """
        if symbol not in self.positions:
            return []
        to_close = []
        for pos in self.positions[symbol]:
            if pos.should_stop_loss(price):
                to_close.append((pos, "stop_loss"))
            elif pos.should_take_profit(price):
                to_close.append((pos, "take_profit"))
        return to_close

    # ------------------------------------------------------------------ #
    # Fermeture de position
    # ------------------------------------------------------------------ #
    def close_position(self, symbol: str, pos: Position,
                       exit_price: float) -> dict:
        """Ferme une position spécifique et met à jour le capital."""
        if symbol in self.positions:
            try:
                self.positions[symbol].remove(pos)
            except ValueError:
                pass
            if not self.positions[symbol]:
                del self.positions[symbol]

        pnl_usdt = pos.pnl(exit_price)
        self.total_capital += pos.cost_usdt + pnl_usdt

        return {
            "symbol":   symbol,
            "entry":    pos.entry_price,
            "exit":     exit_price,
            "size":     pos.size,
            "pnl_usdt": round(pnl_usdt, 4),
            "pnl_pct":  round(pos.pnl_pct(exit_price), 2),
        }

    def close_all_positions(self, symbol: str, exit_price: float) -> list[dict]:
        """Ferme toutes les positions d'un symbole (signal SELL)."""
        if symbol not in self.positions:
            return []
        results = []
        for pos in list(self.positions[symbol]):
            results.append(self.close_position(symbol, pos, exit_price))
        return results

    # ------------------------------------------------------------------ #
    # Résumé
    # ------------------------------------------------------------------ #
    def unrealized_pnl(self, prices: dict[str, float]) -> float:
        """PnL non réalisé total sur les positions ouvertes."""
        total = 0.0
        for symbol, plist in self.positions.items():
            price = prices.get(symbol)
            if price:
                total += sum(p.unrealized_pnl(price) for p in plist)
        return total

    def n_positions(self) -> int:
        return sum(len(plist) for plist in self.positions.values())

    def summary(self) -> dict:
        return {
            "capital":        round(self.total_capital, 2),
            "deployed":       round(self._deployed_capital(), 2),
            "available":      round(self._available_capital(), 2),
            "open_positions": self.n_positions(),
            "positions":      {s: plist for s, plist in self.positions.items()},
        }
