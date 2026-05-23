"""
Étape 14 — Impact de la concentration du capital

Teste 4 niveaux de concentration sur la meilleure config connue :
  Top 20 / 12h / épurée / multi / sansSL

Schemes testés :
  A — 10% par trade, max 10 trades simultanés
  B — 20% par trade, max  5 trades simultanés
  C — 50% par trade, max  2 trades simultanés
  D — 100% par trade, max 1 trade  simultané

Lecture : chaque scheme alloue 100% du capital si tous les slots sont pleins.
          La différence est la diversification instantanée.
"""

import math
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import fear_greed
import multi_sim as ms
from multi_sim import sim_concentration

init(autoreset=True)

# ---------------------------------------------------------------------------
# Config de référence
# ---------------------------------------------------------------------------
SYMBOLS   = config.SYMBOLS          # Top 20
TIMEFRAME = "12h"
FEE_RATE  = ms.FEE_RATE
INITIAL_CAPITAL = ms.INITIAL_CAPITAL

SCHEMES = [
    {"label": "5% / illimité (réf)", "pos_pct": 0.05, "max_trades": 9999},
    {"label": "10% / max10",         "pos_pct": 0.10, "max_trades":   10},
    {"label": "20% / max5",          "pos_pct": 0.20, "max_trades":    5},
    {"label": "50% / max2",          "pos_pct": 0.50, "max_trades":    2},
    {"label": "100% / max1",         "pos_pct": 1.00, "max_trades":    1},
]

# Épurée = sans TripleST, sans SMA/MACD
USE_TRIPLE_ST = False
USE_SMA_MACD  = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_dfs(symbols, tf, years):
    """Charge et prépare les DFs pour la période (derniers `years` ans)."""
    n   = ms.CANDLES_PER_YEAR[tf] * years
    dfs = {}
    for sym in symbols:
        df = ms.get_df(sym, tf)
        if df is not None and len(df) >= 10:
            dfs[sym] = df.tail(n).reset_index(drop=True)
    return dfs


def load_dfs_year(symbols, tf, year):
    """Charge les DFs filtrés sur une année calendaire."""
    dfs = {}
    for sym in symbols:
        df = ms.get_df_for_year(sym, tf, year)
        if df is not None:
            dfs[sym] = df
    return dfs


def fmt_ret(v):
    if v is None:
        return "—"
    color = Fore.GREEN if v > 0 else (Fore.RED if v < 0 else "")
    return f"{color}{v:+.1f}%{Style.RESET_ALL}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n{Fore.CYAN}{'='*80}")
    print("  ÉTAPE 14 — Concentration du capital")
    print(f"  Config : Top 20 / {TIMEFRAME} / épurée / multi / sansSL")
    print(f"{'='*80}{Style.RESET_ALL}")

    # Chargement données
    print(f"\n{Fore.CYAN}Chargement données et calcul indicateurs...{Style.RESET_ALL}")
    data_cache.prefetch_all(SYMBOLS, [TIMEFRAME], verbose=False)
    for sym in SYMBOLS:
        ms.get_df(sym, TIMEFRAME)
        ms.get_df_for_year(sym, TIMEFRAME, 2024)  # préchauffe le cache 10y si besoin

    fg_data = fear_greed.load(verbose=False)
    print(f"  Fear & Greed : {len(fg_data)} jours")

    # -------------------------------------------------------------------------
    # 1. Tableau par période (1/2/3/4 ans)
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*80}")
    print("  RÉSULTATS PAR PÉRIODE")
    print(f"{'='*80}{Style.RESET_ALL}")

    period_headers = ["Scheme", "1 an", "2 ans", "3 ans", "4 ans",
                      "Trades(4a)", "Win%(4a)", "DD%(4a)"]
    period_rows    = []

    for scheme in SCHEMES:
        row = [scheme["label"]]
        r4  = {}
        for years in [1, 2, 3, 4]:
            dfs = load_dfs(SYMBOLS, TIMEFRAME, years)
            r   = sim_concentration(dfs, scheme["pos_pct"], scheme["max_trades"],
                                    fg=fg_data, tf=TIMEFRAME,
                                    use_triple_st=USE_TRIPLE_ST,
                                    use_sma_macd=USE_SMA_MACD)
            row.append(fmt_ret(r.get("return_%")))
            if years == 4:
                r4 = r
        row.append(r4.get("trades", "—"))
        row.append(f"{r4.get('win_%', 0):.1f}%")
        row.append(f"{r4.get('drawdown_%', 0):.1f}%")
        period_rows.append(row)

    print(tabulate(period_rows, headers=period_headers, tablefmt="rounded_outline"))

    # -------------------------------------------------------------------------
    # 2. Tableau par année calendaire (2018–2025)
    # -------------------------------------------------------------------------
    import datetime
    current_year = datetime.date.today().year
    years_range  = list(range(2018, current_year + 1))

    print(f"\n{Fore.YELLOW}{'='*80}")
    print("  RÉSULTATS PAR ANNÉE CALENDAIRE")
    print(f"{'='*80}{Style.RESET_ALL}")

    year_headers = ["Scheme"] + [str(y) for y in years_range] + ["Pct+", "Moy/an"]
    year_rows    = []

    for scheme in SCHEMES:
        row      = [scheme["label"]]
        returns  = []
        for year in years_range:
            dfs = load_dfs_year(SYMBOLS, TIMEFRAME, year)
            if not dfs:
                row.append("—")
                continue
            r   = sim_concentration(dfs, scheme["pos_pct"], scheme["max_trades"],
                                    fg=fg_data, tf=TIMEFRAME,
                                    use_triple_st=USE_TRIPLE_ST,
                                    use_sma_macd=USE_SMA_MACD)
            ret = r.get("return_%")
            if ret is None:
                row.append("—")
            else:
                returns.append(ret)
                row.append(fmt_ret(ret))

        pct_pos = f"{len([r for r in returns if r > 0]) / len(returns) * 100:.0f}%" if returns else "—"
        moy     = f"{sum(returns)/len(returns):+.1f}%" if returns else "—"
        row.extend([pct_pos, moy])
        year_rows.append(row)

    print(tabulate(year_rows, headers=year_headers, tablefmt="rounded_outline"))

    # -------------------------------------------------------------------------
    # 3. Résumé stats détaillées (4 ans)
    # -------------------------------------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*80}")
    print("  STATS DÉTAILLÉES (4 ans)")
    print(f"{'='*80}{Style.RESET_ALL}")

    sum_headers = ["Scheme", "Return 4a", "Trades", "Win%", "Drawdown", "Durée moy (j)"]
    sum_rows    = []
    for scheme in SCHEMES:
        dfs = load_dfs(SYMBOLS, TIMEFRAME, 4)
        r   = sim_concentration(dfs, scheme["pos_pct"], scheme["max_trades"],
                                fg=fg_data, tf=TIMEFRAME)
        ret = r.get("return_%", 0)
        color = Fore.GREEN if ret > 0 else Fore.RED
        sum_rows.append([
            scheme["label"],
            f"{color}{ret:+.2f}%{Style.RESET_ALL}",
            r.get("trades", "—"),
            f"{r.get('win_%', 0):.1f}%",
            f"{r.get('drawdown_%', 0):.1f}%",
            f"{r.get('avg_duration_j', 0):.1f}j",
        ])
    print(tabulate(sum_rows, headers=sum_headers, tablefmt="rounded_outline"))
    print()
