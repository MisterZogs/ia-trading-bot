"""
Point d'entrée du bot de trading.

Usage:
    python main.py                            # Paper trading
    python main.py --live                     # Live (nécessite .env avec clés API)
    python main.py --backtest                 # Backtest historique
    python main.py --debug-signal BTC/USDT BUY   # Simule un signal BUY
    python main.py --debug-signal BTC/USDT SELL  # Simule un signal SELL
"""

import sys
from bot import TradingBot


def main():
    args = sys.argv[1:]

    if "--backtest" in args:
        from backtester import run_backtest, print_results
        import config
        results = []
        for symbol in config.SYMBOLS:
            for tf in ["2h", "4h", "6h", "1d"]:
                try:
                    stats = run_backtest(symbol, tf, initial_capital=1000.0, candles=500)
                    results.append(stats)
                except Exception as e:
                    print(f"Erreur {symbol}/{tf}: {e}")
        print_results(results)
        return

    if "--debug-signal" in args:
        idx = args.index("--debug-signal")
        if len(args) < idx + 3:
            print("Usage : python main.py --debug-signal SYMBOL SIDE (ex: BTC/USDT BUY)")
            return
        symbol = args[idx + 1]
        side   = args[idx + 2]
        bot = TradingBot(initial_capital=1000.0)
        bot.debug_signal(symbol, side)
        return

    if "--live" in args:
        import os
        os.environ["PAPER_TRADING"] = "false"
        print("ATTENTION : Mode LIVE activé. Les ordres seront réels.")
        confirm = input("Taper 'OUI' pour confirmer : ")
        if confirm != "OUI":
            print("Annulé.")
            return

    bot = TradingBot(initial_capital=1000.0)
    bot.run()


if __name__ == "__main__":
    main()
