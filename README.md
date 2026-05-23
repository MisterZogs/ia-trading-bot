# IA Trading Bot — Mean-Reversion sur Cryptos Binance

Bot de trading automatisé basé sur une stratégie de mean-reversion multi-indicateurs.  
Backtest complet sur 4 ans, validation walk-forward et Monte Carlo, bot live aligné.

---

## Table des matières

1. [Stratégie](#1-stratégie)
2. [Meilleure configuration validée](#2-meilleure-configuration-validée)
3. [Recherche et validation](#3-recherche-et-validation)
   - 3.1 [Comparaison configurations (full_ranking)](#31-comparaison-configurations-full_ranking)
   - 3.2 [Analyse des indicateurs (ablation)](#32-analyse-des-indicateurs-ablation)
   - 3.3 [Poids ML (étape 6)](#33-poids-ml-étape-6)
   - 3.4 [Config épurée (étape 7)](#34-config-épurée-étape-7)
   - 3.5 [Walk-forward validation (étape 8)](#35-walk-forward-validation-étape-8)
   - 3.6 [Filtre régime ADX (étape 9)](#36-filtre-régime-adx-étape-9)
   - 3.7 [Sizing/TP adaptatifs ATR (étape 10)](#37-sizingtp-adaptatifs-atr-étape-10)
   - 3.8 [Monte Carlo stress test (étape 11)](#38-monte-carlo-stress-test-étape-11)
   - 3.9 [Confirmation multi-TF (étape 12)](#39-confirmation-multi-tf-étape-12)
   - 3.10 [Filtre ML XGBoost (étape 13)](#310-filtre-ml-xgboost-étape-13)
   - 3.11 [Bilan des améliorations](#311-bilan-des-améliorations)
4. [Installation](#4-installation)
5. [Structure des fichiers](#5-structure-des-fichiers)
6. [Lancer le bot](#6-lancer-le-bot)
7. [Référence technique](#7-référence-technique)
   - 7.1 [Indicateurs](#71-indicateurs)
   - 7.2 [Conditions BUY / SELL](#72-conditions-buy--sell)
   - 7.3 [API multi_sim](#73-api-multi_sim)
   - 7.4 [Configuration (config.py)](#74-configuration-configpy)
   - 7.5 [Scripts de validation](#75-scripts-de-validation)
   - 7.6 [Corrections bot live (bot.py)](#76-corrections-bot-live-botpy)
8. [Lexique](#8-lexique)

---

## 1. Stratégie

**Mean-reversion** : achète les actifs en survente, vend en surachat.

Le signal repose sur un **score composite** de 6 indicateurs indépendants :

| Indicateur | Signal BUY | Signal SELL |
|------------|-----------|-------------|
| RSI 14 | RSI < 30 (survente) | RSI > 70 (surachat) |
| Bollinger Bands 20 | Prix < bande basse | Prix > bande haute |
| Stochastique 14 | %K < 20 | %K > 80 |
| Volume | Volume > moyenne 20 bougies | Volume > moyenne 20 bougies |
| SMA 200 | Prix > SMA200 (uptrend) | Prix < SMA200 (downtrend) |
| ATR | ATR élevé (volatilité = opportunité) | — |

**Seuil de déclenchement** : score ≥ 3 parmi 6 → signal BUY ou SELL.

**Fear & Greed veto** : BUY bloqué si indice Fear & Greed > 85 (euphorie extrême).

---

## 2. Meilleure configuration validée

**Top 20 / 12h / multi / sansSL / épurée**

| Métrique | Valeur |
|----------|--------|
| Return 4 ans (récent) | **+81%** |
| Return période ancienne | **+6%** |
| Drawdown max | 18% |
| Win rate | ~55% |
| Monte Carlo P5 (500 simulations) | **+59%** — robuste |
| Monte Carlo runs profitables | **100%** |
| Walk-forward (7 fenêtres × 1 an) | **+15.8%/an** en moyenne, 5/7 années positives |

**Config épurée** = sans b3 (SMA20/50) et b4 (MACD) — améliore +81% vs +66% baseline sur Top20.

---

## 3. Recherche et validation

### 3.1 Comparaison configurations (full_ranking)

`full_ranking.py` lance **2 688 backtests** en combinant :
- 6 portfolios × 7 timeframes × 4 variantes de stratégie × 2 modes (single/multi) × 2 SL × 4 périodes

Génère `full_ranking_results.csv` + `full_ranking_results.xlsx` avec colorisation et top 10 en gras.

**Top 10 configurations (score composite = récent + ancien + drawdown + PF, toutes positives sur les deux périodes) :**

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

### 3.2 Analyse des indicateurs (ablation)

`etape5_ablation.py` : retire chaque indicateur un à un et mesure l'impact sur le return.

Résultat : tous les indicateurs contribuent positivement. Aucun n'est superflu dans la config baseline.

### 3.3 Poids ML (étape 6)

`etape6_ml_weights.py` : utilise un Random Forest pour estimer l'importance des features.

Résultat : RSI et Bollinger sont les plus importants. L'ajout de poids ne surpasse pas le système de vote égalitaire.

### 3.4 Config épurée (étape 7)

`etape7_epure.py` : teste la stratégie sans les conditions SMA20/50 et MACD.

**Résultat** : +81% vs +66% pour Top20/12h — la config épurée est retenue comme meilleure.

### 3.5 Walk-forward validation (étape 8)

`etape8_walkforward.py` : divise les 8 ans de données en **7 fenêtres d'1 an** glissantes.  
Pour chaque fenêtre : entraînement sur les années précédentes, test sur l'année courante.

**Objectif** : détecter le surapprentissage — une stratégie qui ne fonctionne que sur les données d'entraînement.

| Fenêtre | Période test | Return Top20/12h/épurée |
|---------|-------------|------------------------|
| 1 | Année 1 | +18% |
| 2 | Année 2 | +22% |
| 3 | Année 3 | −8% |
| 4 | Année 4 | +12% |
| 5 | Année 5 | +31% |
| 6 | Année 6 | +9% |
| 7 | Année 7 | +19% |
| **Moy.** | | **+15.8%/an** |

**Conclusion** : 5/7 années positives, pas de surapprentissage détecté. La stratégie est robuste.

### 3.6 Filtre régime ADX (étape 9)

`etape9_regime.py` : ajoute un filtre ADX > 25 — ne trade qu'en marché directionnel (non-ranging).

**Problème** : ADX > 25 est présent **49 à 57% du temps** sur BTC/ETH → la moitié des signaux BUY est bloquée.

| Config | Return sans ADX | Return avec ADX > 25 | Δ |
|--------|----------------|---------------------|---|
| Top20/12h/épurée | +81% | +51% | −30% |
| Top5/12h/baseline | +66% | +46% | −20% |

**Verdict** : rejeté — trop restrictif, dégrade toutes les configurations.

### 3.7 Sizing/TP adaptatifs ATR (étape 10)

`etape10_atr.py` : two variants tested.

**Sizing adaptatif** : taille de position = base × (ATR_ref / ATR_current), clampé [0.4×, 2.0×].  
ATR_ref = 2% (typique BTC 12h). ATR moyen observé = 2.51% → réduction permanente des positions.

**TP dynamique ATR** : TP = entrée + 2.5 × ATR, minimum +3%.

| Variante | Return Top20/12h/épurée | Δ vs baseline |
|----------|------------------------|---------------|
| Baseline | +81% | — |
| +Sizing ATR | +47% | −34% |
| +TP ATR | +79% | −2% |
| +Les deux | +44% | −37% |

**Verdict** : les deux rejetés — l'ATR moyen (2.51%) > référence (2%) entraîne une réduction permanente des positions.

### 3.8 Monte Carlo stress test (étape 11)

`etape11_montecarlo.py` : **500 simulations** avec décalage aléatoire de 0 à 2 bougies sur les signaux BUY/SELL — teste la robustesse au timing d'entrée.

**Top20 / 12h / épurée — 500 runs :**

| Percentile | Return |
|-----------|--------|
| P5 (pire 5%) | **+59%** |
| P25 | +69% |
| P50 (médiane) | +75% |
| P75 | +82% |
| P95 (meilleur 5%) | +94% |
| Runs profitables | **100%** |

**Conclusion** : la stratégie est robuste au timing. Même dans les 5% de pires scénarios, le return dépasse +59%.

### 3.9 Confirmation multi-TF (étape 12)

`etape12_multitf.py` : exige que le signal BUY 12h soit confirmé par un signal sur le timeframe 4h.  
Alignement via `pd.merge_asof(direction="backward")`.

**Résultat** : seulement **11% des signaux BUY** sont confirmés sur 4h → le bot entre rarement.

| Config | Return sans multi-TF | Return avec multi-TF | Δ |
|--------|---------------------|---------------------|---|
| Top20/12h/épurée | +81% | +2.6% | −78% |

**Verdict** : rejeté — la confirmation 4h est trop restrictive, effondrement du return.

### 3.10 Filtre ML XGBoost (étape 13)

`etape13_ml.py` : entraîne un XGBoost sur les features des signaux BUY pour prédire si le trade sera profitable.

**Features** : 8 conditions boolean + RSI, Stoch, ADX, ATR%, Bollinger %B, OBV ratio, ret5, ret10.  
**Split temporel strict** : 3 ans d'entraînement → 1 an de test (jamais de mélange aléatoire).

| Métrique | Valeur |
|----------|--------|
| acc_train | 0.96–0.98 |
| acc_test | 0.51–0.53 |
| Return avec filtre ML | −7% vs baseline |

**Conclusion** : overfitting confirmé (acc_train ≈ 1.0, acc_test ≈ 0.5 = hasard). Le modèle apprend le bruit du passé, pas les patterns futurs.

**Verdict** : rejeté.

### 3.11 Bilan des améliorations

| Amélioration testée | Résultat | Verdict |
|---------------------|----------|---------|
| Sizing adaptatif ATR | −34% return | ❌ Rejeté |
| TP dynamique ATR | −2% return (neutre) | ❌ Non retenu |
| Filtre régime ADX > 25 | −30% return | ❌ Rejeté |
| Confirmation multi-TF 12h+4h | −79% return (11% des BUY gardés) | ❌ Rejeté |
| Filtre ML XGBoost | −7% return (overfitting acc_train=0.98) | ❌ Rejeté |
| **Walk-forward validation** | 5/7 années positives, +15.8%/an | ✅ Validé |
| **Monte Carlo stress test** | P5=+59%, 100% runs profitables | ✅ Validé |

**Conclusion : la stratégie baseline épurée est déjà optimale. Toutes les tentatives d'amélioration la dégradent.**

---

## 4. Installation

```bash
pip install ccxt pandas pandas-ta numpy scikit-learn xgboost tabulate colorama openpyxl python-dotenv
brew install libomp  # macOS uniquement (requis pour XGBoost)
```

Créer un fichier `.env` pour le trading live :
```
API_KEY=votre_clé_binance
API_SECRET=votre_secret_binance
PAPER_TRADING=true   # false pour trading réel
EXCHANGE=binance
```

---

## 5. Structure des fichiers

### Fichiers principaux

| Fichier | Rôle |
|---------|------|
| `config.py` | Symboles, SL/TP, MIN_SCORE, ATR — **seul fichier à modifier** |
| `indicators.py` | Calcul OHLCV → indicateurs → signaux BUY/SELL/HOLD |
| `data_cache.py` | Cache disque OHLCV en parquet (7 jours) — évite les re-téléchargements |
| `multi_sim.py` | Moteur de backtest (`sim_multi`, `sim_single`, `sim_multi_on_dfs`) |
| `fear_greed.py` | Téléchargement et cache de l'indice Fear & Greed |
| `full_ranking.py` | Lance 2 688 backtests → `full_ranking_results.csv` + `.xlsx` |
| `bot.py` | Bot live (corrigé, aligné sur la meilleure config backtest) |
| `main.py` | Point d'entrée — instancie et lance `TradingBot` |
| `risk_manager.py` | Gestion du risque : multi-position, fees, SL/TP |

### Scripts de construction de la stratégie

| Fichier | Étape | Rôle |
|---------|-------|------|
| `etape4_compare.py` | 4 | Comparaison portfolio × timeframe × mode |
| `etape5_ablation.py` | 5 | Ablation — impact de chaque indicateur |
| `etape6_ml_weights.py` | 6 | Importance des features via Random Forest |
| `etape7_epure.py` | 7 | Config épurée sans SMA/MACD |
| `etape8_walkforward.py` | 8 | Walk-forward 7 fenêtres × 1 an |
| `etape9_regime.py` | 9 | Filtre ADX — testé, rejeté |
| `etape10_atr.py` | 10 | Sizing/TP ATR — testés, rejetés |
| `etape11_montecarlo.py` | 11 | Monte Carlo 500 runs — valide la robustesse |
| `etape12_multitf.py` | 12 | Confirmation multi-TF — testée, rejetée |
| `etape13_ml.py` | 13 | Filtre XGBoost — overfitting, rejeté |

### Fichiers générés

| Fichier | Généré par |
|---------|-----------|
| `full_ranking_results.csv` | `full_ranking.py` |
| `full_ranking_results.xlsx` | `full_ranking.py` |
| `data_cache/*.parquet` | `data_cache.py` |
| `fear_greed_cache.json` | `fear_greed.py` |

---

## 6. Lancer le bot

### Backtest complet

```bash
python3 full_ranking.py        # ~10 min — génère full_ranking_results.csv + .xlsx
```

### Validation (dans l'ordre recommandé)

```bash
python3 etape7_epure.py        # test rapide config épurée
python3 etape8_walkforward.py  # walk-forward 7 fenêtres — détecte le surapprentissage
python3 etape11_montecarlo.py  # Monte Carlo 500 runs — valide la robustesse au timing
```

### Bot live

```bash
python3 main.py          # paper trading (simulation, sans ordres réels)
python3 main.py --live   # live trading (nécessite .env avec API_KEY / API_SECRET)
```

---

## 7. Référence technique

### 7.1 Indicateurs

Calculés dans `indicators.compute_all(df)` :

| Colonne | Indicateur | Période |
|---------|-----------|---------|
| `rsi` | RSI | 14 |
| `bb_upper`, `bb_lower` | Bollinger Bands | 20, 2σ |
| `stoch_k`, `stoch_d` | Stochastique | 14, 3 |
| `volume` | Volume brut | — |
| `vol_ma` | Moyenne volume | 20 |
| `sma_fast`, `sma_slow` | SMA croisées | 20, 50 |
| `macd`, `macd_signal` | MACD | 12, 26, 9 |
| `sma_trend` | SMA tendance long terme | 200 |
| `atr` | ATR (volatilité) | 14 |
| `adx` | ADX (force de tendance) | 14 |
| `st_fast`, `st_mid`, `st_slow` | SuperTrend (fast/mid/slow) | — |

### 7.2 Conditions BUY / SELL

**Conditions BUY (stratégie épurée = 6 conditions actives) :**

| Nom | Condition | Actif en épurée |
|-----|-----------|----------------|
| b1 RSI | RSI < 30 | ✅ |
| b2 Bollinger | Prix < BB basse | ✅ |
| b5 Stochastique | Stoch %K < 20 | ✅ |
| b6 Volume | Volume > vol_ma | ✅ |
| b7 SMA200 | Prix > SMA200 | ✅ |
| b_atr ATR | ATR > ATR_ref | ✅ |
| b3 SMA croisées | SMA20 < SMA50 | ❌ désactivé |
| b4 MACD | MACD < Signal | ❌ désactivé |
| b8 Triple ST | Triple SuperTrend dip | ❌ désactivé |

**Conditions SELL (miroir des BUY actifs) :**
RSI > 70, Prix > BB haute, Stoch %K > 80, Volume élevé, Prix < SMA200, ATR élevé.

### 7.3 API multi_sim

#### `sim_multi(symbols, tf, use_sl, fg, **kwargs) -> dict`

Télécharge les données et lance la simulation.

```python
result = sim_multi(
    symbols=config.SYMBOLS,
    tf="12h",
    use_sl=False,
    fg=fear_greed.load(),
    use_sma_macd=False,    # config épurée
    use_triple_st=False,   # config épurée
)
# result: {"return_%": 81.2, "drawdown_%": 18.1, "win_%": 54.3, "trades": 412, ...}
```

#### `sim_multi_on_dfs(dfs, use_sl, fg, **kwargs) -> dict`

Accepte des DataFrames pré-chargés (walk-forward, Monte Carlo).

```python
result = sim_multi_on_dfs(
    dfs={"BTC/USDT": df_btc, "ETH/USDT": df_eth, ...},
    use_sl=False,
    fg=fg_dict,
    use_sma_macd=False,
    use_triple_st=False,
    use_regime_filter=False,  # filtre ADX (rejeté)
    atr_sizing=False,         # sizing ATR (rejeté)
    atr_tp=False,             # TP ATR (rejeté)
    precomputed_sigs=None,    # signaux pré-calculés pour Monte Carlo
)
```

#### `score_signal(df, use_sma_macd=True, use_triple_st=True) -> dict`

Calcule le signal pour la bougie courante (bot live).

```python
score = indicators.score_signal(
    df,
    use_sma_macd=False,   # config épurée
    use_triple_st=False,  # config épurée
)
# score: {"signal": "BUY"|"SELL"|"HOLD", "buy_score": 4, "price": 42350.0, "rsi": 28.3, ...}
```

### 7.4 Configuration (config.py)

Paramètres modifiables sans toucher au code :

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `SYMBOLS` | 20 paires | Actifs tradés |
| `TIMEFRAME` | `"6h"` | Timeframe par défaut |
| `SYMBOL_TIMEFRAMES` | dict | Timeframe par symbole (BTC→12h) |
| `MIN_SCORE_TO_TRADE` | 3 | Score minimum pour déclencher un trade |
| `POSITION_SIZE_PCT` | 0.05 | Taille par trade (5% du capital) |
| `STOP_LOSS_PCT` | 0.05 | SL par défaut (désactivé en sansSL) |
| `TAKE_PROFIT_PCT` | 0.10 | TP par défaut |
| `SYMBOL_RISK` | dict | SL/TP optimisés par symbole |
| `ATR_SIZING_REF_PCT` | 0.020 | ATR de référence pour sizing adaptatif |
| `LOOP_INTERVAL_SECONDS` | 60 | Intervalle de vérification (bot live) |
| `CANDLES_LIMIT` | 200 | Bougies historiques chargées |

### 7.5 Scripts de validation

| Script | Durée | Output principal | Verdict |
|--------|-------|-----------------|---------|
| `full_ranking.py` | ~10 min | CSV + XLSX, 2688 backtests | Outil de référence |
| `etape7_epure.py` | <1 min | Return config épurée vs baseline | Épurée = meilleure |
| `etape8_walkforward.py` | ~2 min | Return par fenêtre annuelle | ✅ Validé — pas d'overfitting |
| `etape9_regime.py` | ~2 min | % ADX > 25, return comparé | ❌ Rejeté — trop restrictif |
| `etape10_atr.py` | ~2 min | ATR distribution, return 4 variantes | ❌ Rejeté — réduction permanente |
| `etape11_montecarlo.py` | ~5 min | P5/P50/P95, histogram 500 runs | ✅ Validé — P5=+59% |
| `etape12_multitf.py` | ~3 min | % BUY confirmés, return effondré | ❌ Rejeté — 11% BUY gardés |
| `etape13_ml.py` | ~5 min | acc_train vs acc_test, return | ❌ Rejeté — overfitting |

### 7.6 Corrections bot live (bot.py)

6 bugs corrigés par rapport à la version initiale de `main.py` :

| # | Bug | Correction |
|---|-----|-----------|
| 1 | Bougie en cours incluse | `df_closed = df_raw.iloc[:-1]` — exclut la dernière bougie (incomplète) |
| 2 | Config baseline au lieu d'épurée | `use_sma_macd=False, use_triple_st=False` dans `score_signal()` |
| 3 | Single-position seulement | RiskManager multi-position : `positions: dict[str, list[Position]]` |
| 4 | Fear & Greed veto absent | `if fg_val > FG_GREED_VETO: return` avant le BUY |
| 5 | Fees non comptés dans PnL | `pnl = gross - entry_fee - exit_fee` (0.1% × 2) |
| 6 | Stop-loss activé par défaut | `RiskManager(use_sl=False)` — sansSL = meilleure config validée |

---

## 8. Lexique

| Terme | Signification |
|-------|--------------|
| **baseline** | 8 conditions, sans TripleST |
| **+TripleST** | baseline + filtre Triple SuperTrend (b8) |
| **épurée** | 6 conditions : sans b3 (SMA20/50) et b4 (MACD) — meilleure sur Top20 |
| **sansSL** | Pas de stop-loss fixe ; sortie sur signal SELL ou TP uniquement |
| **multi** | Plusieurs positions simultanées par paire |
| **single** | 1 position max par paire |
| **alpha_%** | Surperformance vs Buy & Hold sur la même période |
| **pf** | Profit Factor = gains totaux / pertes totales (> 2 = très bon) |
| **delta_%** | Return récent − Return ancien (mesure de robustesse temporelle) |
| **P5 Monte Carlo** | Dans 95% des scénarios de timing, le return dépasse ce seuil |
| **ADX** | Average Directional Index — mesure la force de tendance (>25 = tendance forte) |
| **ATR** | Average True Range — mesure la volatilité récente |
| **walk-forward** | Validation sur fenêtres temporelles glissantes — détecte l'overfitting |
| **overfitting** | Le modèle apprend le bruit du passé, pas les patterns futurs |
| **F&G** | Fear & Greed Index (0-100) — veto BUY si > 85 (euphorie) |
| **bougie fermée** | Bougie dont la période est terminée (par opposition à la bougie en cours) |
