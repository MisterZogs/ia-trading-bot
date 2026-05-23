"""
ML sur les signaux BUY — XGBoost / RandomForest
================================================
Entraîne un classifieur binaire pour prédire si un BUY signal
sera profitable ou non, en utilisant le contexte des indicateurs.

Features par trade :
  - Les 8 conditions BUY (booléens)
  - ADX, RSI, Stoch K, ATR%, OBV/OBV_SMA ratio
  - Position relative au Bollinger (bb_pct)
  - Contexte de marché : rendement des 5 dernières bougies

Split temporel strict : train sur les 3 premières années, test sur la 4e.
(Jamais de validation croisée aléatoire — look-ahead bias interdit)

Usage :
    python3 etape13_ml.py
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import indicators
import multi_sim
import fear_greed

init(autoreset=True)

TRAIN_YEARS = 3   # années d'entraînement
TEST_YEARS  = 1   # année de test (la plus récente)
HOLD_PERIODS = [1, 2, 3, 5]   # horizons pour calculer la rentabilité

CONFIGS = [
    {
        "label":         "Top20/12h/épurée",
        "symbols":       multi_sim.PORTFOLIOS["top20"],
        "timeframe":     "12h",
        "use_triple_st": False,
        "use_sma_macd":  False,
    },
    {
        "label":         "Top5/12h/baseline",
        "symbols":       multi_sim.PORTFOLIOS["top5"],
        "timeframe":     "12h",
        "use_triple_st": False,
        "use_sma_macd":  True,
    },
]


# ---------------------------------------------------------------------------
# Extraction des features et labels pour chaque signal BUY
# ---------------------------------------------------------------------------
def extract_features(df: pd.DataFrame, sigs: np.ndarray,
                     hold_candles: int = 3) -> tuple[pd.DataFrame, pd.Series]:
    """
    Pour chaque BUY signal dans sigs, extrait les features au moment de l'entrée
    et calcule le label (1 = profitable si on tient hold_candles bougies).
    """
    close = df["close"].values
    n     = len(df)
    rows  = []

    for i, sig in enumerate(sigs):
        if sig != "BUY":
            continue
        if i + hold_candles >= n:
            continue   # pas assez de bougies futures pour calculer le label

        entry_price = close[i]
        exit_price  = close[i + hold_candles]
        profitable  = 1 if exit_price > entry_price * (1 + 2 * multi_sim.FEE_RATE) else 0

        row = df.iloc[i]

        # Features indicateurs
        atr_pct   = row["atr"] / entry_price if entry_price > 0 else 0
        bb_range  = row["bb_upper"] - row["bb_lower"]
        bb_pct    = (entry_price - row["bb_lower"]) / bb_range if bb_range > 0 else 0.5
        obv_ratio = row["obv"] / row["obv_sma"] if row["obv_sma"] and row["obv_sma"] != 0 else 1.0

        # Rendements récents (contexte de momentum)
        ret5  = (entry_price - close[max(0, i-5)])  / close[max(0, i-5)]  if i >= 5  else 0
        ret10 = (entry_price - close[max(0, i-10)]) / close[max(0, i-10)] if i >= 10 else 0

        # Conditions BUY booléennes (0/1)
        chg    = (close[i] - close[i-1]) / close[i-1] if i >= 1 else 0
        b1 = int(chg < -config.PRICE_CHANGE_THRESHOLD_PCT / 100)
        b2 = int(pd.notna(row["rsi"])     and row["rsi"]     < config.RSI_OVERSOLD)
        b3 = int(pd.notna(row["sma_fast"]) and pd.notna(row["sma_slow"]) and row["sma_fast"] < row["sma_slow"])
        b4 = int(pd.notna(row["macd"])    and pd.notna(row["macd_signal"]) and row["macd"] < row["macd_signal"])
        b5 = int(pd.notna(row["stoch_k"]) and row["stoch_k"] < config.STOCH_OVERSOLD)
        b6 = int(pd.notna(row["bb_lower"]) and entry_price   < row["bb_lower"])
        b7 = int(pd.notna(row["volume_sma"]) and row["volume_sma"] > 0
                 and row["volume"] > row["volume_sma"])
        b8 = int(row.get("st_dir_7", 0) == -1 and row.get("st_dir_21", 0) == 1)

        rows.append({
            "b1": b1, "b2": b2, "b3": b3, "b4": b4,
            "b5": b5, "b6": b6, "b7": b7, "b8": b8,
            "rsi":      row["rsi"]     if pd.notna(row["rsi"])     else 50,
            "stoch_k":  row["stoch_k"] if pd.notna(row["stoch_k"]) else 50,
            "adx":      row["adx"]     if pd.notna(row.get("adx")) else 20,
            "atr_pct":  round(atr_pct, 4),
            "bb_pct":   round(bb_pct, 4),
            "obv_ratio":round(obv_ratio, 4),
            "ret5":     round(ret5, 4),
            "ret10":    round(ret10, 4),
            "label":    profitable,
        })

    if not rows:
        return pd.DataFrame(), pd.Series(dtype=int)

    df_out = pd.DataFrame(rows)
    return df_out.drop("label", axis=1), df_out["label"]


# ---------------------------------------------------------------------------
# Simulation avec filtre ML
# ---------------------------------------------------------------------------
def sim_with_ml_filter(dfs: dict, base_sigs: dict,
                       models: dict, fg_data: dict,
                       use_triple_st: bool, use_sma_macd: bool) -> dict:
    """
    Filtre les BUY selon la prédiction du modèle ML avant de passer au sim.
    models : dict {symbol: trained_model}
    """
    filtered_sigs = {}
    stats = {"total_buy": 0, "ml_approved": 0}

    for s in dfs:
        df   = dfs[s]
        sigs = base_sigs[s].copy()
        model = models.get(s)

        if model is None:
            filtered_sigs[s] = sigs
            continue

        close = df["close"].values
        n     = len(df)

        for i in range(n):
            if sigs[i] != "BUY":
                continue
            stats["total_buy"] += 1

            row    = df.iloc[i]
            entry  = close[i]
            chg    = (close[i] - close[i-1]) / close[i-1] if i >= 1 else 0
            atr_pct   = row["atr"] / entry if entry > 0 else 0
            bb_range  = row["bb_upper"] - row["bb_lower"]
            bb_pct    = (entry - row["bb_lower"]) / bb_range if bb_range > 0 else 0.5
            obv_ratio = row["obv"] / row["obv_sma"] if row["obv_sma"] and row["obv_sma"] != 0 else 1.0
            ret5  = (entry - close[max(0, i-5)])  / close[max(0, i-5)]  if i >= 5  else 0
            ret10 = (entry - close[max(0, i-10)]) / close[max(0, i-10)] if i >= 10 else 0

            feat = pd.DataFrame([{
                "b1": int(chg < -config.PRICE_CHANGE_THRESHOLD_PCT / 100),
                "b2": int(pd.notna(row["rsi"])      and row["rsi"]      < config.RSI_OVERSOLD),
                "b3": int(pd.notna(row["sma_fast"])  and pd.notna(row["sma_slow"])  and row["sma_fast"] < row["sma_slow"]),
                "b4": int(pd.notna(row["macd"])     and pd.notna(row["macd_signal"]) and row["macd"] < row["macd_signal"]),
                "b5": int(pd.notna(row["stoch_k"])  and row["stoch_k"]  < config.STOCH_OVERSOLD),
                "b6": int(pd.notna(row["bb_lower"]) and entry            < row["bb_lower"]),
                "b7": int(pd.notna(row["volume_sma"]) and row["volume_sma"] > 0 and row["volume"] > row["volume_sma"]),
                "b8": int(row.get("st_dir_7", 0) == -1 and row.get("st_dir_21", 0) == 1),
                "rsi":      row["rsi"]     if pd.notna(row["rsi"])     else 50,
                "stoch_k":  row["stoch_k"] if pd.notna(row["stoch_k"]) else 50,
                "adx":      row["adx"]     if pd.notna(row.get("adx")) else 20,
                "atr_pct":  round(atr_pct, 4),
                "bb_pct":   round(bb_pct, 4),
                "obv_ratio":round(obv_ratio, 4),
                "ret5":     round(ret5, 4),
                "ret10":    round(ret10, 4),
            }])

            pred = model.predict(feat)[0]
            if pred == 1:
                stats["ml_approved"] += 1
            else:
                sigs[i] = "HOLD"

        filtered_sigs[s] = sigs

    return multi_sim.sim_multi_on_dfs(
        dfs, use_sl=False, fg=fg_data,
        precomputed_sigs=filtered_sigs,
    ), stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    try:
        from xgboost import XGBClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score
        HAS_ML = True
    except ImportError:
        print(f"{Fore.RED}XGBoost ou scikit-learn non installé.{Style.RESET_ALL}")
        print("  pip install xgboost scikit-learn")
        return

    print(f"\n{Fore.CYAN}{'='*80}")
    print("  ML SUR LES SIGNAUX BUY — XGBoost + RandomForest")
    print(f"{'='*80}{Style.RESET_ALL}")
    print(f"  Train : {TRAIN_YEARS} ans  |  Test : {TEST_YEARS} an  |  Hold : 3 bougies")
    print(f"  Split temporel strict (pas de look-ahead)")

    all_symbols = list(dict.fromkeys(s for cfg in CONFIGS for s in cfg["symbols"]))
    all_tfs     = list(dict.fromkeys(cfg["timeframe"] for cfg in CONFIGS))

    print(f"\n{Fore.CYAN}Chargement des données...{Style.RESET_ALL}")
    data_cache.prefetch_all(all_symbols, all_tfs, verbose=True)

    print(f"{Fore.CYAN}Calcul des indicateurs...{Style.RESET_ALL}")
    done, total = 0, len(all_symbols) * len(all_tfs)
    for sym in all_symbols:
        for tf in all_tfs:
            multi_sim.get_df(sym, tf)
            done += 1
            print(f"\r  {done}/{total}", end="", flush=True)
    print()

    fg_data = fear_greed.load(verbose=False)

    summary_headers = [
        "Config / Modèle",
        "Return", "DD", "PF", "Win%", "Trades",
        "Δ Return", "Δ DD", "BUY gardés",
    ]
    summary_rows = []

    for cfg in CONFIGS:
        tf   = cfg["timeframe"]
        n    = multi_sim.CANDLES_PER_YEAR[tf]
        n_tr = n * TRAIN_YEARS
        n_te = n * TEST_YEARS

        print(f"\n{Fore.YELLOW}{'─'*60}")
        print(f"  {cfg['label']}")
        print(f"{'─'*60}{Style.RESET_ALL}")

        # DataFrames complets (train+test) — utilise les 8 ans pour avoir assez de marge
        dfs_full: dict = {}
        for sym in cfg["symbols"]:
            df = multi_sim.get_df_8y(sym, tf)
            if df is None:
                df = multi_sim.get_df(sym, tf)   # fallback
            if df is not None and len(df) >= n_tr + n_te:
                dfs_full[sym] = df.tail(n_tr + n_te).reset_index(drop=True)

        if not dfs_full:
            print("  Pas assez de données.")
            continue

        # Split train / test
        dfs_train = {s: dfs_full[s].iloc[:n_tr].reset_index(drop=True) for s in dfs_full}
        dfs_test  = {s: dfs_full[s].iloc[n_tr:].reset_index(drop=True) for s in dfs_full}

        # Signaux sur train et test
        sigs_train = {
            s: indicators.vectorized_signals(
                dfs_train[s], use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"]).values
            for s in dfs_train
        }
        sigs_test = {
            s: indicators.vectorized_signals(
                dfs_test[s], use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"]).values
            for s in dfs_test
        }

        # Construire le dataset d'entraînement (tous symboles combinés)
        X_parts, y_parts = [], []
        for s in dfs_train:
            X, y = extract_features(dfs_train[s], sigs_train[s], hold_candles=3)
            if len(X) > 0:
                X_parts.append(X)
                y_parts.append(y)

        if not X_parts:
            print("  Aucun signal BUY dans la période d'entraînement.")
            continue

        X_train = pd.concat(X_parts, ignore_index=True)
        y_train = pd.concat(y_parts, ignore_index=True)

        print(f"  Train : {len(X_train)} signaux BUY "
              f"({y_train.sum()} profitables = {y_train.mean()*100:.0f}%)")

        # Baseline test (sans ML)
        r_base = multi_sim.sim_multi_on_dfs(
            dfs_test, use_sl=False, fg=fg_data,
            precomputed_sigs=sigs_test,
        )

        def fmt(v):
            c = Fore.GREEN if v > 0 else Fore.RED
            return f"{c}{v:+.1f}%{Style.RESET_ALL}"

        def fmtd(v, good_pos=True):
            if abs(v) < 0.5:
                return f"{v:+.1f}"
            c = (Fore.GREEN if v > 0 else Fore.RED) if good_pos \
                else (Fore.GREEN if v < 0 else Fore.RED)
            return f"{c}{v:+.1f}{Style.RESET_ALL}"

        summary_rows.append([
            f"{Fore.YELLOW}{cfg['label']}{Style.RESET_ALL}",
            "", "", "", "", "", "", "", "",
        ])
        summary_rows.append([
            "  Baseline (sans ML)",
            fmt(r_base.get("return_%", 0)),
            f"{r_base.get('drawdown_%', 0):.1f}%",
            f"{r_base.get('profit_factor', 0):.2f}",
            f"{r_base.get('win_%', 0):.1f}%",
            str(r_base.get("trades", 0)),
            "—", "—", "—",
        ])
        ret0, dd0 = r_base.get("return_%", 0), r_base.get("drawdown_%", 0)
        t0 = r_base.get("trades", 0)

        # Entraîner et tester XGBoost + RandomForest
        models_to_test = [
            ("XGBoost", XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, verbosity=0,
            )),
            ("RandomForest", RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, n_jobs=-1,
            )),
        ]

        for model_name, model in models_to_test:
            model.fit(X_train, y_train)

            # Évaluation sur train (pour détecter overfitting)
            acc_tr = accuracy_score(y_train, model.predict(X_train))

            # Entraîner un modèle par symbole (même poids pour tous)
            # On utilise le même modèle global pour tous les symboles
            models_dict = {s: model for s in dfs_test}

            r_ml, stats_ml = sim_with_ml_filter(
                dfs_test, sigs_test, models_dict, fg_data,
                use_triple_st=cfg["use_triple_st"],
                use_sma_macd=cfg["use_sma_macd"],
            )

            total_b = stats_ml["total_buy"]
            approv  = stats_ml["ml_approved"]
            pct_k   = approv / total_b * 100 if total_b else 0

            ret  = r_ml.get("return_%", 0)
            dd   = r_ml.get("drawdown_%", 0)
            t    = r_ml.get("trades", 0)

            print(f"  {model_name} : acc_train={acc_tr:.2f} | "
                  f"BUY gardés {approv}/{total_b} ({pct_k:.0f}%) | "
                  f"Return {ret:+.1f}%")

            summary_rows.append([
                f"  {model_name} (acc_tr={acc_tr:.2f})",
                fmt(ret),
                f"{dd:.1f}%",
                f"{r_ml.get('profit_factor', 0):.2f}",
                f"{r_ml.get('win_%', 0):.1f}%",
                str(t),
                fmtd(ret - ret0, good_pos=True),
                fmtd(dd  - dd0,  good_pos=False),
                f"{approv}/{total_b} ({pct_k:.0f}%)",
            ])

        # Feature importance XGBoost
        xgb = models_to_test[0][1]
        feat_names = X_train.columns.tolist()
        importances = sorted(
            zip(feat_names, xgb.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        print(f"\n  Feature importance XGBoost ({cfg['label']}) :")
        for name, imp in importances[:8]:
            bar = "█" * int(imp * 200)
            print(f"    {name:<12} {imp:.3f}  {bar}")

    # Tableau de synthèse
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  SYNTHÈSE ML")
    print(f"{'='*90}{Style.RESET_ALL}")
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.CYAN}Interprétation :{Style.RESET_ALL}")
    print("  • acc_train ≈ 0.65-0.70 → apprentissage réel")
    print("  • acc_train ≈ 1.00      → overfitting (mémorise les données)")
    print("  • acc_train ≈ 0.50      → pas mieux que random (dataset trop petit)")
    print("  • Δ Return > 0          → le filtre ML améliore la sélection")
    print("  • BUY gardés < 20%      → trop restrictif, le modèle rejette trop")
    print()
    print("  Dataset crypto 12h : ~300-500 BUY sur 3 ans → petit dataset")
    print("  → Résultats à interpréter avec prudence (variance élevée)")


if __name__ == "__main__":
    main()
