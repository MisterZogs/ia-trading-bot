"""
Étape 6 — ML pour apprendre le poids optimal de chaque condition BUY

Approche :
  1. Calcule les 8 features binaires (b1-b8) sur toutes les bougies
  2. Construit le label : le prix a-t-il atteint le TP dans les H bougies suivantes ?
  3. Entraîne 3 modèles : Logistic Regression, Random Forest, XGBoost
  4. Affiche les poids/importances de chaque condition
  5. Backtest avec le score pondéré ML vs score fixe MIN_SCORE=3

Split temporel : 75% train / 25% test (jamais de fuite du futur)
"""

import numpy as np
import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style, init

import config
import data_cache
import fear_greed
import multi_sim as ms

init(autoreset=True)

# ---------------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------------
FOCUS_SYMBOLS = config.SYMBOLS          # Top20
FOCUS_TF      = "12h"                   # meilleur TF identifié
HORIZON       = 20                      # bougies pour atteindre le TP
TP_DEFAULT    = 0.10                    # TP par défaut si absent de SYMBOL_RISK
PROBA_THRESH  = 0.55                    # seuil de probabilité pour déclencher BUY (LR/XGB)
MIN_SCORE_REF = 3                       # seuil de référence actuel

CONDITION_NAMES = [
    "Prix baisse >2%", "RSI<30", "SMA20<SMA50", "MACD<Sig",
    "Stoch<20", "Prix<BB_low", "Vol>VolSMA", "TripleST",
]


# ---------------------------------------------------------------------------
# Calcul des 8 features binaires sur un DataFrame
# ---------------------------------------------------------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"]
    prev  = close.shift(1)
    chg   = (close - prev) / prev * 100
    zero  = pd.Series(False, index=df.index)

    feats = pd.DataFrame({
        "b1": chg < -config.PRICE_CHANGE_THRESHOLD_PCT,
        "b2": df["rsi"].notna() & (df["rsi"] < config.RSI_OVERSOLD),
        "b3": df["sma_fast"].notna() & df["sma_slow"].notna() & (df["sma_fast"] < df["sma_slow"]),
        "b4": df["macd"].notna() & df["macd_signal"].notna() & (df["macd"] < df["macd_signal"]),
        "b5": df["stoch_k"].notna() & (df["stoch_k"] < config.STOCH_OVERSOLD),
        "b6": df["bb_lower"].notna() & (close < df["bb_lower"]),
        "b7": df["volume_sma"].notna() & (df["volume_sma"] > 0) & (df["volume"] > df["volume_sma"]),
        "b8": (df["st_dir_7"].fillna(0) == -1) & (df["st_dir_21"].fillna(0) == 1),
    }).astype(int)

    # Filtre tendance SMA200 (veto BUY si downtrend)
    feats["in_uptrend"] = (df["sma200"].isna() | (close > df["sma200"])).astype(int)

    return feats


def compute_label(df: pd.DataFrame, symbol: str, horizon: int) -> pd.Series:
    """
    Label = 1 si le high max sur les H bougies suivantes >= close * (1 + TP)
    """
    tp_pct = config.SYMBOL_RISK.get(symbol, {}).get("tp", TP_DEFAULT)
    target = df["close"] * (1 + tp_pct)
    # rolling max des highs futurs
    future_high = df["high"].shift(-1).rolling(window=horizon, min_periods=1).max().shift(-(horizon-1))
    label = (future_high >= target).astype(int)
    return label


# ---------------------------------------------------------------------------
# Construction du dataset
# ---------------------------------------------------------------------------
def build_dataset():
    rows = []
    for sym in FOCUS_SYMBOLS:
        df = ms.get_df(sym, FOCUS_TF)
        if df is None or len(df) < 100:
            continue
        feats = compute_features(df)
        label = compute_label(df, sym, HORIZON)
        feats["label"]  = label
        feats["symbol"] = sym
        feats["date"]   = df["timestamp"].values if "timestamp" in df.columns else np.arange(len(df))
        # Retirer les dernières H bougies (label invalide)
        feats = feats.iloc[:-HORIZON]
        # Garder seulement les bougies en uptrend (les seules où on peut BUY)
        feats = feats[feats["in_uptrend"] == 1]
        rows.append(feats)

    data = pd.concat(rows, ignore_index=True)
    return data


# ---------------------------------------------------------------------------
# Simulation avec score pondéré ML
# ---------------------------------------------------------------------------
def sim_ml_weighted(syms, tf, years, weights, threshold, fg):
    """
    BUY quand le score pondéré >= threshold.
    weights : array de 8 floats (un par condition b1-b8)
    """
    import math
    n = ms.CANDLES_PER_YEAR[tf] * years
    dfs  = {}
    sigs = {}

    for sym in syms:
        df = ms.get_df(sym, tf)
        if df is None or len(df) < 10:
            continue
        sliced = df.tail(n).reset_index(drop=True)
        feats  = compute_features(sliced)

        close = sliced["close"]
        prev  = close.shift(1)
        chg   = (close - prev) / prev * 100

        b = np.column_stack([feats[f"b{i+1}"].values for i in range(8)])
        score_buy  = b @ np.array(weights)

        # Sell : conditions miroir (mêmes poids, côté opposé)
        s1 = (chg > config.PRICE_CHANGE_THRESHOLD_PCT).astype(int).values
        s2 = (sliced["rsi"].notna() & (sliced["rsi"] > config.RSI_OVERBOUGHT)).astype(int).values
        s3 = (sliced["sma_fast"].notna() & sliced["sma_slow"].notna() & (sliced["sma_fast"] > sliced["sma_slow"])).astype(int).values
        s4 = (sliced["macd"].notna() & sliced["macd_signal"].notna() & (sliced["macd"] > sliced["macd_signal"])).astype(int).values
        s5 = (sliced["stoch_k"].notna() & (sliced["stoch_k"] > config.STOCH_OVERBOUGHT)).astype(int).values
        s6 = (sliced["bb_upper"].notna() & (close > sliced["bb_upper"])).astype(int).values
        s7 = (sliced["volume_sma"].notna() & (sliced["volume_sma"] > 0) & (sliced["volume"] > sliced["volume_sma"])).astype(int).values
        s8 = (sliced["st_dir_7"].fillna(0) == 1).astype(int).values
        s  = np.column_stack([s1,s2,s3,s4,s5,s6,s7,s8])
        score_sell = s @ np.array(weights)

        in_uptrend = (sliced["sma200"].isna() | (close > sliced["sma200"])).values

        sig_arr = np.full(len(sliced), "HOLD", dtype=object)
        is_buy  = in_uptrend & (score_buy >= threshold) & (score_buy > score_sell)
        is_sell = (score_sell >= threshold) & (score_sell > score_buy)
        sig_arr[is_buy]  = "BUY"
        sig_arr[is_sell] = "SELL"
        sig_arr[is_buy & is_sell] = "BUY"

        dfs[sym]  = sliced
        sigs[sym] = sig_arr

    if not dfs:
        return {}

    arr_low   = {s: dfs[s]["low"].values   for s in dfs}
    arr_high  = {s: dfs[s]["high"].values  for s in dfs}
    arr_close = {s: dfs[s]["close"].values for s in dfs}
    arr_len   = {s: len(dfs[s]) for s in dfs}
    risks     = {s: config.SYMBOL_RISK.get(s, {"sl": config.STOP_LOSS_PCT, "tp": config.TAKE_PROFIT_PCT}) for s in dfs}
    min_slot  = ms.INITIAL_CAPITAL * ms.POSITION_SIZE_PCT
    ref_sym   = next(iter(dfs))
    arr_dates = dfs[ref_sym]["timestamp"].dt.date.values if "timestamp" in dfs[ref_sym].columns else None

    capital   = ms.INITIAL_CAPITAL
    positions: dict = {s: [] for s in dfs}
    trades    = []
    equity    = [capital]
    max_len   = max(arr_len.values())

    for i in range(2, max_len):
        fg_val = None
        if fg is not None and arr_dates is not None and i < len(arr_dates):
            fg_val = fg.get(arr_dates[i])

        for sym in dfs:
            if i >= arr_len[sym]:
                continue
            high  = arr_high[sym][i]
            close = arr_close[sym][i]

            still_open = []
            for pos in positions[sym]:
                if high >= pos["tp"]:
                    fee_exit = pos["tp"] * pos["size"] * ms.FEE_RATE
                    pnl = (pos["tp"] - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                else:
                    still_open.append(pos)
            positions[sym] = still_open

            sig = sigs[sym][i]
            if sig == "BUY" and fg_val is not None and fg_val > fear_greed.FG_GREED_VETO:
                sig = "HOLD"

            deployed  = sum(p["entry"] * p["size"] for plist in positions.values() for p in plist)
            available = capital - deployed

            if sig == "BUY" and available >= min_slot:
                risk = risks[sym]
                capital -= min_slot * ms.FEE_RATE
                size = min_slot / close
                positions[sym].append({"entry": close, "size": size, "sl": 0.0,
                                        "tp": close * (1 + risk["tp"])})
            elif sig == "SELL" and positions[sym]:
                for pos in positions[sym]:
                    fee_exit = close * pos["size"] * ms.FEE_RATE
                    pnl = (close - pos["entry"]) * pos["size"] - fee_exit
                    capital += pnl
                    trades.append(pnl)
                positions[sym] = []

        equity.append(capital)

    for sym, plist in positions.items():
        last = arr_close[sym][-1]
        for pos in plist:
            fee = last * pos["size"] * ms.FEE_RATE
            capital += (last - pos["entry"]) * pos["size"] - fee
            trades.append((last - pos["entry"]) * pos["size"] - fee)

    return ms._stats(trades, capital, equity)


def bah(syms, tf, years):
    n = ms.CANDLES_PER_YEAR[tf] * years
    returns = []
    for sym in syms:
        df = ms.get_df(sym, tf)
        if df is None or len(df) < 10:
            continue
        dp = df.tail(n)
        s, e = dp["close"].iloc[0], dp["close"].iloc[-1]
        if s > 0:
            returns.append((e - s) / s * 100)
    return round(sum(returns) / len(returns), 1) if returns else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import classification_report, roc_auc_score
        from sklearn.pipeline import Pipeline
    except ImportError:
        print("Installer scikit-learn : pip install scikit-learn")
        exit(1)

    try:
        import xgboost as xgb
        HAS_XGB = True
    except ImportError:
        print("XGBoost non installé (pip install xgboost) — on utilisera GradientBoosting")
        HAS_XGB = False

    print(f"\n{Fore.CYAN}{'='*90}")
    print("  ÉTAPE 6 — ML Weights : poids optimal de chaque condition BUY")
    print(f"{'='*90}{Style.RESET_ALL}")

    # ---- Chargement --------------------------------------------------------
    print(f"\n{Fore.CYAN}Chargement données ({FOCUS_TF})...{Style.RESET_ALL}")
    data_cache.prefetch_all(FOCUS_SYMBOLS, [FOCUS_TF], verbose=False)
    for sym in FOCUS_SYMBOLS:
        ms.get_df(sym, FOCUS_TF)
    print(f"  {len(FOCUS_SYMBOLS)} symboles chargés")

    print(f"{Fore.CYAN}Chargement Fear & Greed...{Style.RESET_ALL}")
    fg_data = fear_greed.load(verbose=False)

    # ---- Dataset -----------------------------------------------------------
    print(f"{Fore.CYAN}Construction du dataset (horizon={HORIZON} bougies)...{Style.RESET_ALL}")
    data = build_dataset()
    print(f"  {len(data)} observations | {data['label'].mean()*100:.1f}% positives (TP atteint)")

    feature_cols = [f"b{i+1}" for i in range(8)]
    X = data[feature_cols].values
    y = data["label"].values

    # Split temporel 75/25 — on prend les dernières bougies pour le test
    split = int(len(X) * 0.75)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Taux positifs — train: {y_train.mean()*100:.1f}% | test: {y_test.mean()*100:.1f}%")

    # ---- Modèle 1 : Logistic Regression ------------------------------------
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  MODÈLE 1 — Logistic Regression (coefficients = poids directs)")
    print(f"{'='*90}{Style.RESET_ALL}")

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict_proba(X_test)[:, 1]
    auc_lr = roc_auc_score(y_test, y_pred_lr)

    coefs = lr.coef_[0]
    coefs_norm = coefs / np.abs(coefs).max()  # normalisé entre -1 et 1

    rows_lr = []
    for i, (name, coef, norm) in enumerate(zip(CONDITION_NAMES, coefs, coefs_norm)):
        bar_len = int(abs(norm) * 20)
        bar = ("█" * bar_len) if coef > 0 else ("░" * bar_len)
        color = Fore.GREEN if coef > 0 else Fore.RED
        rows_lr.append([
            f"b{i+1} — {name}",
            f"{color}{coef:+.3f}{Style.RESET_ALL}",
            f"{color}{norm:+.2f}{Style.RESET_ALL}",
            f"{color}{bar}{Style.RESET_ALL}",
        ])
    rows_lr.sort(key=lambda r: float(r[1].replace(Fore.GREEN,"").replace(Fore.RED,"")
                 .replace(Style.RESET_ALL,"").strip()), reverse=True)

    print(tabulate(rows_lr,
                   headers=["Condition", "Coef brut", "Coef normalisé", "Impact"],
                   tablefmt="rounded_outline"))
    print(f"  AUC ROC sur test : {auc_lr:.3f}  (0.5 = aléatoire, 1.0 = parfait)")

    # ---- Modèle 2 : Random Forest ------------------------------------------
    print(f"\n{Fore.YELLOW}{'='*90}")
    print("  MODÈLE 2 — Random Forest (feature importance)")
    print(f"{'='*90}{Style.RESET_ALL}")

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    auc_rf = roc_auc_score(y_test, y_pred_rf)
    imp_rf = rf.feature_importances_
    imp_rf_norm = imp_rf / imp_rf.max()

    rows_rf = []
    for i, (name, imp, norm) in enumerate(zip(CONDITION_NAMES, imp_rf, imp_rf_norm)):
        bar = "█" * int(norm * 20)
        rows_rf.append([f"b{i+1} — {name}", f"{imp:.4f}", f"{norm:.2f}", bar])
    rows_rf.sort(key=lambda r: -float(r[1]))

    print(tabulate(rows_rf,
                   headers=["Condition", "Importance", "Normalisée", "Impact"],
                   tablefmt="rounded_outline"))
    print(f"  AUC ROC sur test : {auc_rf:.3f}")

    # ---- Modèle 3 : XGBoost / GradientBoosting -----------------------------
    print(f"\n{Fore.YELLOW}{'='*90}")
    model_name = "XGBoost" if HAS_XGB else "GradientBoosting"
    print(f"  MODÈLE 3 — {model_name} (feature importance)")
    print(f"{'='*90}{Style.RESET_ALL}")

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    if HAS_XGB:
        gb = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                scale_pos_weight=scale_pos, random_state=42,
                                eval_metric="auc", verbosity=0)
    else:
        gb = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict_proba(X_test)[:, 1]
    auc_gb = roc_auc_score(y_test, y_pred_gb)
    imp_gb = gb.feature_importances_
    imp_gb_norm = imp_gb / imp_gb.max()

    rows_gb = []
    for i, (name, imp, norm) in enumerate(zip(CONDITION_NAMES, imp_gb, imp_gb_norm)):
        bar = "█" * int(norm * 20)
        rows_gb.append([f"b{i+1} — {name}", f"{imp:.4f}", f"{norm:.2f}", bar])
    rows_gb.sort(key=lambda r: -float(r[1]))

    print(tabulate(rows_gb,
                   headers=["Condition", "Importance", "Normalisée", "Impact"],
                   tablefmt="rounded_outline"))
    print(f"  AUC ROC sur test : {auc_gb:.3f}")

    # ---- Résumé des poids --------------------------------------------------
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  RÉSUMÉ — Classement des conditions par les 3 modèles")
    print(f"{'='*90}{Style.RESET_ALL}")

    # Rang moyen (1 = plus important)
    def ranks(arr):
        order = np.argsort(arr)[::-1]
        r = np.zeros(8, int)
        for rank, idx in enumerate(order):
            r[idx] = rank + 1
        return r

    r_lr = ranks(np.abs(coefs))
    r_rf = ranks(imp_rf)
    r_gb = ranks(imp_gb)
    avg_rank = (r_lr + r_rf + r_gb) / 3

    summary_rows = []
    for i, name in enumerate(CONDITION_NAMES):
        coef_sign = "+" if coefs[i] > 0 else "-"
        coef_color = Fore.GREEN if coefs[i] > 0 else Fore.RED
        summary_rows.append([
            f"b{i+1} — {name}",
            f"{r_lr[i]}",
            f"{r_rf[i]}",
            f"{r_gb[i]}",
            f"{avg_rank[i]:.1f}",
            f"{coef_color}{coef_sign}{Style.RESET_ALL}",
        ])
    summary_rows.sort(key=lambda r: float(r[4]))

    print(tabulate(summary_rows,
                   headers=["Condition", "Rang LR", "Rang RF", f"Rang {model_name}", "Rang moy.", "LR sign"],
                   tablefmt="rounded_outline"))

    # ---- Backtest : score pondéré par LR -----------------------------------
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  BACKTEST — Comparaison : score fixe vs score pondéré ML")
    print(f"{'='*90}{Style.RESET_ALL}")

    CONFIGS_BT = {
        "Top20 / 12h / 4ans": (config.SYMBOLS, "12h", 4),
        "Top5  / 12h / 4ans": (["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","XRP/USDT"], "12h", 4),
        "BTC+ETH / 12h / 4ans": (["BTC/USDT","ETH/USDT"], "12h", 4),
    }

    # Poids LR normalisés en [0, 1] (pour remplacer le score 0/1 par poids)
    # On clip les négatifs à 0 (conditions nuisibles → poids nul)
    weights_lr_pos = np.clip(coefs / np.abs(coefs[coefs > 0]).min(), 0, None)
    # Seuil = équivalent MIN_SCORE=3 dans l'espace pondéré
    threshold_lr = weights_lr_pos[coefs > 0].mean() * MIN_SCORE_REF

    print(f"\n  Poids LR utilisés (négatifs mis à 0) :")
    for i, (name, w) in enumerate(zip(CONDITION_NAMES, weights_lr_pos)):
        bar = "█" * int(w / weights_lr_pos.max() * 15)
        print(f"    b{i+1} {name:<20} {w:.2f}  {bar}")
    print(f"  Seuil d'activation : {threshold_lr:.2f}\n")

    bt_headers = ["Config", "Stratégie", "Return%", "Max DD%", "Win%", "Trades", "PF", "B&H%", "Alpha"]
    bt_rows = []

    for cfg_name, (syms, tf, years) in CONFIGS_BT.items():
        bah_val = bah(syms, tf, years)

        # Référence : score fixe MIN_SCORE=3, toutes conditions
        ref = ms.sim_multi(syms, tf, years, use_sl=False, fg=fg_data, use_triple_st=True)
        ref_ret = ref.get("return_%", 0)
        bt_rows.append([cfg_name, f"Référence (MIN_SCORE={MIN_SCORE_REF})",
                        f"{ref_ret:+.1f}%", f"{ref.get('drawdown_%',0):+.1f}%",
                        f"{ref.get('win_%',0):.0f}%", ref.get("trades",0),
                        f"{ref.get('profit_factor',0):.2f}", f"{bah_val:+.1f}%",
                        f"{round(ref_ret-bah_val,1):+.1f}%"])

        # Score pondéré LR
        ml = sim_ml_weighted(syms, tf, years, weights_lr_pos.tolist(), threshold_lr, fg_data)
        ml_ret = ml.get("return_%", 0)
        color = Fore.GREEN if ml_ret > ref_ret else Fore.RED
        bt_rows.append([cfg_name, "Score pondéré LR",
                        f"{color}{ml_ret:+.1f}%{Style.RESET_ALL}",
                        f"{ml.get('drawdown_%',0):+.1f}%",
                        f"{ml.get('win_%',0):.0f}%", ml.get("trades",0),
                        f"{ml.get('profit_factor',0):.2f}", f"{bah_val:+.1f}%",
                        f"{round(ml_ret-bah_val,1):+.1f}%"])
        bt_rows.append(["", "", "", "", "", "", "", "", ""])

    print(tabulate(bt_rows, headers=bt_headers, tablefmt="rounded_outline"))

    # ---- Conclusion --------------------------------------------------------
    print(f"\n{Fore.CYAN}{'='*90}")
    print("  CONCLUSION")
    print(f"{'='*90}{Style.RESET_ALL}")
    print(f"""
  AUC ROC des modèles (prédiction de la profitabilité d'un BUY) :
    Logistic Regression : {auc_lr:.3f}
    Random Forest       : {auc_rf:.3f}
    {model_name:<20}: {auc_gb:.3f}

  Un AUC > 0.55 indique que les conditions ont un pouvoir prédictif réel.
  Un AUC ≈ 0.50 signifie que les conditions sont proches d'un signal aléatoire.
  Plus l'AUC est élevé, plus le modèle ML peut améliorer la sélection des entrées.
""")
