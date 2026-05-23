# Quickstart — IA Trading Bot

> Version courte. Pour les détails complets, voir `README.md`.

## Ce que fait ce bot

Mean-reversion sur cryptos Binance : achète en survente, vend en surachat.  
Signal = score ≥ 3 parmi 6 indicateurs (RSI, Bollinger, Stoch, Volume, SMA200, ATR).

## Meilleure configuration validée

**Top 20 / 12h / multi / sansSL / épurée** (voir `config.py`)
- Return récent 4 ans : **+81%** | Return période ancienne : **+6%** | Drawdown max : 18%
- Monte Carlo (500 simulations) : **P5 = +59%** — robuste au timing, 100% de runs profitables
- Walk-forward (7 fenêtres d'1 an) : **+15.8% par an en moyenne**, 5/7 années positives

## Fichiers essentiels

| Fichier | Rôle |
|---------|------|
| `config.py` | Symboles, SL/TP, MIN_SCORE — **seul fichier à modifier** |
| `indicators.py` | Calcul indicateurs + signaux BUY/SELL/HOLD |
| `data_cache.py` | Cache disque OHLCV (7 jours) |
| `multi_sim.py` | Moteur de backtest (`sim_multi`, `sim_single`, `sim_multi_on_dfs`) |
| `full_ranking.py` | Lance 2 688 backtests → `full_ranking_results.csv` |
| `main.py` | Point d'entrée du bot live |

## Installation

```bash
pip install ccxt pandas pandas-ta numpy scikit-learn xgboost tabulate colorama openpyxl
brew install libomp  # macOS uniquement
```

## Lancer un backtest

```bash
python3 full_ranking.py        # ~10 min, génère full_ranking_results.csv + .xlsx
python3 etape7_epure.py        # test rapide de la stratégie épurée
```

## Scripts de validation (lancer dans l'ordre)

```bash
python3 etape8_walkforward.py  # walk-forward : 7 fenêtres d'1 an — détecte le surapprentissage
python3 etape9_regime.py       # filtre ADX — testé et rejeté (trop restrictif)
python3 etape10_atr.py         # sizing/TP adaptatifs ATR — testé, rejeté sur la plupart des configs
python3 etape11_montecarlo.py  # Monte Carlo 500 runs — valide la robustesse au timing
python3 etape12_multitf.py     # confirmation multi-TF 12h+4h — testé et rejeté
python3 etape13_ml.py          # filtre ML XGBoost — testé, overfitting confirmé
```

## Résultats de la recherche d'améliorations

| Amélioration testée | Résultat | Verdict |
|---------------------|----------|---------|
| Sizing adaptatif ATR | −34% return | ❌ Rejeté |
| TP dynamique ATR | −2% return (neutre) | ❌ Non retenu |
| Filtre régime ADX > 25 | −30% return (ADX > 25 = 50% du temps) | ❌ Rejeté |
| Confirmation multi-TF 12h+4h | −79% return (11% des BUY gardés) | ❌ Rejeté |
| Filtre ML XGBoost | −7% return (overfitting acc_train=0.98) | ❌ Rejeté |
| **Walk-forward validation** | 5/7 années positives, pas de surapprentissage | ✅ Validé |
| **Monte Carlo stress test** | P5=+59%, 100% runs profitables | ✅ Validé |

**Conclusion : la stratégie baseline est déjà optimale. Les tentatives d'amélioration dégradent toutes le return.**

## Fichier Excel généré (`full_ranking_results.xlsx`)

Colonnes colorées :
- **Vert** : métriques récentes (`return_%`, `drawdown_%`, `win_%`, `trades`, `pf`, `bah_%`, `alpha_%`)
- **Bleu** : métriques anciennes (`ret_old_%`, `dd_old_%`, `bah_old_%`, `alpha_old_%`)

Les **10 meilleures configurations** sont en **gras** (score composite : return récent + return ancien + drawdown + PF, toutes positives sur les deux périodes) :

| Rang | Portfolio | TF | Stratégie | Récent | Ancien | DD |
|------|-----------|----|-----------|--------|--------|----|
| 2 | Top 20 | 12h | épurée | +81% | +6% | 18% |
| 5 | Top 5 | 12h | baseline | +66% | +22% | 10% |
| 8 | Top 5 | 6h | baseline | +58% | +43% | 15% |
| 11 | Top 20 | 2h | épurée+ST | +54% | +65% | 15% |
| 12 | BTC+ETH | 12h | +TripleST | +53% | +19% | 6% |
| 14 | Top 5 | 6h | baseline | +53% | +57% | 17% |
| 21 | BTC+ETH | 12h | baseline | +47% | +42% | 4% |
| 22 | BTC+ETH | 6h | +TripleST | +47% | +38% | 5% |
| 29 | ETH | 6h | +TripleST | +43% | +34% | 3% |
| 46 | ETH | 6h | +TripleST | +36% | +28% | 1% |

Requiert : `pip install openpyxl`

## Lancer le bot live

```bash
python3 main.py          # paper trading (simulation, sans ordres réels)
python3 main.py --live   # live trading (nécessite .env avec API_KEY / API_SECRET)
```

**Le bot live est aligné avec la meilleure config backtest :**
- Config épurée (use_sma_macd=False, use_triple_st=False)
- Multi-position par symbole
- SansSL (sortie sur signal SELL ou TP)
- Bougie en cours de formation exclue
- Fear & Greed veto actif (BUY bloqué si F&G > 85)
- Fees 0.1% inclus dans le PnL
- Retry automatique sur erreurs réseau
- Shutdown propre sur CTRL+C

## Lexique rapide

| Terme | Signification |
|-------|--------------|
| **baseline** | 8 conditions, sans TripleST |
| **+TripleST** | baseline + filtre Triple SuperTrend (b8) |
| **épurée** | 6 conditions : sans b3 (SMA) et b4 (MACD) — meilleure sur Top20 |
| **sansSL** | Pas de stop-loss fixe ; sortie sur signal SELL ou TP |
| **multi** | Plusieurs positions simultanées par paire |
| **alpha_%** | Surperformance vs Buy & Hold |
| **pf** | Profit Factor = gains / pertes (> 2 = très bon) |
| **delta_%** | Return récent − Return ancien (robustesse) |
| **P5 Monte Carlo** | Dans 95% des scénarios de timing, le return dépasse ce seuil |
| **ADX** | Average Directional Index — mesure la force de tendance (>25 = tendance forte) |
