"""
solution3_multihot_full.py
--------------------------
Pipeline completo com multi-hot encoding em tudo:
    Classificador 1 (SVM + multihot): biomass_positive
    Classificador 2 (SVM + multihot): product_positive (apenas viaveis)
    Regressor (ANN + multihot):       biomass + product (apenas ambos positivos)

Metricas calculadas separadamente para cada componente:
    - Classificador 1: accuracy, precision, recall, f1, confusion matrix
    - Classificador 2: accuracy, precision, recall, f1, confusion matrix
    - ANN regressora: MAE, R² por output (apenas nas amostras onde foi usada)
    - Pipeline completo: MAE, R² por output (em todo o conjunto de teste)

Logica de predicao:
    if biomass_positive == 0 -> biomass = 0, product = 0
    elif product_positive == 0 -> biomass = ANN, product = 0
    else -> biomass = ANN, product = ANN
"""

import os
import json
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, r2_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Configuracao 
DATA_XLSX  = "dataset_multihot.xlsx"
DATA_SHEET = 0

RAW_GLUCOSE_COL = "glucose_uptake(mmol/gDW/h)"
RAW_O2_COL      = "o2_uptake(mmol/gDW/h)"
RAW_NK_COL      = "num_knockouts"
RAW_BIOMASS_COL = "biomass(h\u207b\u00b9)"
RAW_PRODUCT_COL = "product_flux(mmol/gDW/h)"

GLUCOSE_COL = "Glucose"
O2_COL      = "O2"
NK_COL      = "NK"
BIOMASS_COL = "Biomass"
PRODUCT_COL = "Product"

NUMERIC_COLS      = [GLUCOSE_COL, O2_COL, NK_COL]
TARGET_OBJECTIVES = [BIOMASS_COL, PRODUCT_COL]

N_SPLITS     = 5
RANDOM_STATE = 24
MAX_EVALS    = 200
PATIENCE     = 25
VERBOSE_FIT  = 0
TEST_SIZE    = 0.15

VIABILITY_THRESHOLD = 0.5

OUT_DIR         = "solution3_multihot_full"
CLF1_PATH       = os.path.join(OUT_DIR, "clf1_biomass.joblib")
CLF2_PATH       = os.path.join(OUT_DIR, "clf2_product.joblib")
CLF_SCALER_PATH = os.path.join(OUT_DIR, "clf_scaler.joblib")
MODEL_PATH      = os.path.join(OUT_DIR, "regressor_ann.keras")
XSCALER_PATH    = os.path.join(OUT_DIR, "x_numeric_scaler.joblib")
YSCALER_PATH    = os.path.join(OUT_DIR, "y_scaler.joblib")
METADATA_JSON   = os.path.join(OUT_DIR, "model_metadata.json")
REPORT_XLSX     = os.path.join(OUT_DIR, "training_report.xlsx")


def set_seeds(seed: int = 24) -> None:
    """Garante reproducibilidade fixando todas as seeds aleatorias."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seeds(RANDOM_STATE)
tf.config.run_functions_eagerly(False)


def load_multihot(excel_path: str, sheet) -> Tuple:
    """Carrega o dataset multi-hot e descobre automaticamente as colunas dos genes entre NK e Biomass."""
    df = pd.read_excel(excel_path, sheet_name=sheet).drop_duplicates().reset_index(drop=True)

    rename = {
        RAW_GLUCOSE_COL: GLUCOSE_COL,
        RAW_O2_COL:      O2_COL,
        RAW_NK_COL:      NK_COL,
        RAW_BIOMASS_COL: BIOMASS_COL,
        RAW_PRODUCT_COL: PRODUCT_COL,
    }
    df = df.rename(columns=rename)

    all_cols  = df.columns.tolist()
    idx_nk    = all_cols.index(NK_COL)
    idx_bio   = all_cols.index(BIOMASS_COL)
    gene_cols = all_cols[idx_nk + 1: idx_bio]

    print(f"  Multi-hot: {len(gene_cols)} colunas de genes")
    print(f"  Primeiro gene: {gene_cols[0]} | Ultimo gene: {gene_cols[-1]}")

    input_cols = NUMERIC_COLS + gene_cols
    for c in input_cols + TARGET_OBJECTIVES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=input_cols + TARGET_OBJECTIVES).reset_index(drop=True)
    df[NK_COL] = df[NK_COL].round().astype(int)

    X             = df[input_cols].to_numpy(dtype=float)
    y             = df[TARGET_OBJECTIVES].to_numpy(dtype=float)
    biomass_label = (df[BIOMASS_COL].to_numpy() > 0).astype(int)
    product_label = (df[PRODUCT_COL].to_numpy() > 0).astype(int)
    strata        = df[NK_COL].to_numpy(dtype=int)

    return X, y, biomass_label, product_label, strata, gene_cols, df


def compute_clf_metrics(y_true, y_pred, y_prob, name: str) -> Dict:
    """Calcula e imprime metricas detalhadas de um classificador binario: accuracy, precision, recall, F1 e confusion matrix."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print(f"\n── {name} ──")
    print(f"  Accuracy:           {acc:.4f}")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall:             {rec:.4f}")
    print(f"  F1-score:           {f1:.4f}")
    print(f"  False Positive Rate:{fpr:.4f}  (inviavel previsto como viavel)")
    print(f"  False Negative Rate:{fnr:.4f}  (viavel previsto como inviavel)")
    print(f"  Confusion Matrix:")
    print(f"    TN={tn}  FP={fp}")
    print(f"    FN={fn}  TP={tp}")

    return {
        "name": name,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "false_positive_rate": fpr, "false_negative_rate": fnr,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def compute_ann_metrics(y_true, y_pred, name: str) -> Dict:
    """Calcula e imprime MAE e R² por output para a ANN regressora."""
    mae_bio  = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
    mae_prod = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
    r2_bio   = r2_score(y_true[:, 0], y_pred[:, 0])
    r2_prod  = r2_score(y_true[:, 1], y_pred[:, 1])
    mae_avg  = mean_absolute_error(y_true, y_pred)

    print(f"\n── {name} ──")
    print(f"  N amostras avaliadas: {len(y_true)}")
    print(f"  Biomass  — R²: {r2_bio:.4f} | MAE: {mae_bio:.6f}")
    print(f"  Product  — R²: {r2_prod:.4f} | MAE: {mae_prod:.6f}")
    print(f"  MAE medio:     {mae_avg:.6f}")

    return {
        "name": name,
        "n_samples": len(y_true),
        "r2_biomass": r2_bio, "mae_biomass": mae_bio,
        "r2_product": r2_prod, "mae_product": mae_prod,
        "mae_avg": mae_avg,
    }


def train_svm_classifiers(X_train, biomass_label_train, product_label_train):
    """Treina dois classificadores SVM com multi-hot encoding: clf1 para biomass_positive e clf2 para product_positive apenas nas amostras viaveis."""
    clf_scaler = StandardScaler().fit(X_train)
    X_train_s  = clf_scaler.transform(X_train)

    print("A treinar Classificador 1 (biomass_positive)...")
    clf1 = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    cv1  = cross_val_score(
        clf1, X_train_s, biomass_label_train,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )
    print(f"  CV Accuracy clf1: {cv1.mean():.4f} +/- {cv1.std():.4f}")
    clf1.fit(X_train_s, biomass_label_train)

    viable_mask          = biomass_label_train == 1
    X_train_s_viable     = X_train_s[viable_mask]
    product_label_viable = product_label_train[viable_mask]

    print("A treinar Classificador 2 (product_positive, apenas viaveis)")
    clf2 = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")
    cv2  = cross_val_score(
        clf2, X_train_s_viable, product_label_viable,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )
    print(f"  CV Accuracy clf2: {cv2.mean():.4f} +/- {cv2.std():.4f}")
    clf2.fit(X_train_s_viable, product_label_viable)

    return clf1, clf2, clf_scaler, cv1, cv2


def build_ann_model(input_dim: int, output_dim: int, params: Dict) -> keras.Model:
    """Constroi a ANN multi-hot com um unico input flat, camadas Dense configuradas pelos hiperparametros e output linear de 2 neuronios."""
    reg = regularizers.l2(params["l2"]) if params["l2"] > 0 else None

    inputs = layers.Input(shape=(input_dim,), name="inputs")
    x = inputs

    for i in range(int(params["num_layers"])):
        x = layers.Dense(
            int(params[f"units_{i}"]),
            activation=params[f"activation_{i}"],
            kernel_regularizer=reg,
            name=f"dense_{i + 1}",
        )(x)
        if int(params["batchnorm"]):
            x = layers.BatchNormalization(name=f"bn_{i + 1}")(x)
        if float(params[f"dropout_{i}"]) > 0:
            x = layers.Dropout(float(params[f"dropout_{i}"]), name=f"dropout_{i + 1}")(x)

    outputs = layers.Dense(output_dim, activation="linear", name="outputs")(x)
    model   = keras.Model(inputs=inputs, outputs=outputs, name="regressor_ann_multihot")

    opt_name = params["optimizer"]
    lr = float(params["learning_rate"])
    if opt_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=lr)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=lr)

    model.compile(optimizer=opt, loss="mae", metrics=["mae"])
    return model


def build_search_space(max_layers: int = 5) -> Dict:
    """Define o espaco de pesquisa de hiperparametros para o Hyperopt TPE."""
    acts  = ["relu", "tanh", "elu", "selu", "softplus"]
    space = {
        "num_layers":    hp.choice("num_layers",    [1, 2, 3, 4, 5]),
        "optimizer":     hp.choice("optimizer",     ["adam", "nadam", "rmsprop"]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(5e-2)),
        "epochs":        hp.quniform("epochs", 80, 700, 1),
        "batch_size":    hp.choice("batch_size",    [16, 32, 64, 128]),
        "l2":            hp.choice("l2",            [0.0, 1e-7, 1e-6, 1e-5, 1e-4]),
        "batchnorm":     hp.choice("batchnorm",     [0, 1]),
    }
    for i in range(max_layers):
        space[f"units_{i}"]      = hp.quniform(f"units_{i}", 16, 512, 1)
        space[f"dropout_{i}"]    = hp.uniform(f"dropout_{i}", 0.0, 0.45)
        space[f"activation_{i}"] = hp.choice(f"activation_{i}", acts)
    return space


def clean_params(best_params: Dict) -> Dict:
    """Converte os tipos dos hiperparametros devolvidos pelo Hyperopt para os tipos corretos do Python."""
    nl = int(best_params["num_layers"])
    cleaned = {
        "num_layers":    nl,
        "optimizer":     best_params["optimizer"],
        "learning_rate": float(best_params["learning_rate"]),
        "epochs":        int(best_params["epochs"]),
        "batch_size":    int(best_params["batch_size"]),
        "l2":            float(best_params["l2"]),
        "batchnorm":     int(best_params["batchnorm"]),
    }
    for i in range(nl):
        cleaned[f"units_{i}"]      = int(best_params[f"units_{i}"])
        cleaned[f"dropout_{i}"]    = float(best_params[f"dropout_{i}"])
        cleaned[f"activation_{i}"] = best_params[f"activation_{i}"]
    return cleaned


def make_ann_objective(X, y, strata):
    """Cria a funcao objetivo para o Hyperopt, avaliando cada configuracao com 5-fold cross-validation estratificada."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def _objective(params: Dict):
        params = clean_params(params)
        fold_mae = []
        per_output_mae_acc = []

        for tr_idx, va_idx in skf.split(X, strata):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            x_scaler = StandardScaler().fit(X_tr)
            X_tr_s   = x_scaler.transform(X_tr)
            X_va_s   = x_scaler.transform(X_va)

            y_scaler = MinMaxScaler().fit(y_tr)
            y_tr_s   = y_scaler.transform(y_tr)
            y_va_s   = y_scaler.transform(y_va)

            model = build_ann_model(X_tr_s.shape[1], y_tr_s.shape[1], params)
            es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True, verbose=0)
            model.fit(
                X_tr_s, y_tr_s,
                validation_data=(X_va_s, y_va_s),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[es],
                verbose=VERBOSE_FIT,
            )

            pred_s = model.predict(X_va_s, verbose=0)
            fold_mae.append(mean_absolute_error(y_va_s, pred_s))
            per_output_mae_acc.append(np.mean(np.abs(pred_s - y_va_s), axis=0))
            keras.backend.clear_session()

        avg_mae = float(np.mean(fold_mae))
        return {
            "loss":               avg_mae,
            "status":             STATUS_OK,
            "avg_mae":            avg_mae,
            "avg_mae_per_output": np.mean(np.vstack(per_output_mae_acc), axis=0).tolist(),
            "params":             params,
        }
    return _objective


def train_final_ann(X_train, y_train, X_test, y_test, best_params: Dict):
    """Treina o modelo final com os melhores hiperparametros encontrados pelo Hyperopt e calcula as metricas no conjunto de teste."""
    x_scaler  = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s  = x_scaler.transform(X_test)

    y_scaler  = MinMaxScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s  = y_scaler.transform(y_test)

    model = build_ann_model(X_train_s.shape[1], y_train_s.shape[1], best_params)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True, verbose=0)
    hist = model.fit(
        X_train_s, y_train_s,
        validation_data=(X_test_s, y_test_s),
        epochs=int(best_params["epochs"]),
        batch_size=int(best_params["batch_size"]),
        callbacks=[es],
        verbose=VERBOSE_FIT,
    )

    pred_train_s = model.predict(X_train_s, verbose=0)
    pred_test_s  = model.predict(X_test_s,  verbose=0)
    pred_train   = y_scaler.inverse_transform(pred_train_s)
    pred_test    = y_scaler.inverse_transform(pred_test_s)

    report = {
        "train_mae_scaled":  float(mean_absolute_error(y_train_s, pred_train_s)),
        "test_mae_scaled":   float(mean_absolute_error(y_test_s,  pred_test_s)),
        "test_r2_per_output": [float(r2_score(y_test[:, j], pred_test[:, j]))
                                for j in range(y_test.shape[1])],
        "history":   {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "y_test":    y_test,
        "pred_test": pred_test,
    }
    return model, x_scaler, y_scaler, report


def predict_pipeline(
    X, clf1, clf2, clf_scaler,
    ann_model, x_scaler_ann, y_scaler_ann,
    threshold: float = VIABILITY_THRESHOLD
):
    """Aplica o pipeline completo: clf1 filtra inviaveis, clf2 filtra sem produto, ANN preve os valores reais."""
    n = len(X)
    predictions = np.zeros((n, 2))

    X_s = clf_scaler.transform(X)

    prob_biomass = clf1.predict_proba(X_s)[:, 1]
    biomass_pos  = (prob_biomass >= threshold).astype(int)

    viable_idx  = np.where(biomass_pos == 1)[0]
    product_pos = np.zeros(n, dtype=int)
    if len(viable_idx) > 0:
        prob_product = clf2.predict_proba(X_s[viable_idx])[:, 1]
        product_pos[viable_idx] = (prob_product >= threshold).astype(int)

    both_pos_idx = np.where((biomass_pos == 1) & (product_pos == 1))[0]
    if len(both_pos_idx) > 0:
        X_ann_s = x_scaler_ann.transform(X[both_pos_idx])
        pred_s  = ann_model.predict(X_ann_s, verbose=0)
        pred    = y_scaler_ann.inverse_transform(pred_s)
        predictions[both_pos_idx] = pred

    biomass_only_idx = np.where((biomass_pos == 1) & (product_pos == 0))[0]
    if len(biomass_only_idx) > 0:
        X_ann_s = x_scaler_ann.transform(X[biomass_only_idx])
        pred_s  = ann_model.predict(X_ann_s, verbose=0)
        pred    = y_scaler_ann.inverse_transform(pred_s)
        predictions[biomass_only_idx, 0] = pred[:, 0]
        predictions[biomass_only_idx, 1] = 0.0

    return predictions, biomass_pos, product_pos


def save_report(
    best_params, trials, ann_report,
    clf1_metrics, clf2_metrics, ann_metrics, pipeline_metrics,
    cv1, cv2,
    y_test_full, pred_full,
    biomass_label_test, product_label_test,
    biomass_pred_test, product_pred_test
) -> None:
    """Guarda todas as metricas e predicoes num ficheiro Excel com sheets separadas por componente."""
    os.makedirs(OUT_DIR, exist_ok=True)

    clf_cv_df = pd.DataFrame({
        "classifier":       ["clf1_biomass", "clf2_product"],
        "cv_accuracy_mean": [cv1.mean(), cv2.mean()],
        "cv_accuracy_std":  [cv1.std(),  cv2.std()],
    })

    clf_test_df = pd.DataFrame([clf1_metrics, clf2_metrics])
    ann_df = pd.DataFrame([ann_metrics])
    pipeline_df_metrics = pd.DataFrame([pipeline_metrics])

    predictions_df = pd.DataFrame({
        "true_biomass":     y_test_full[:, 0],
        "true_product":     y_test_full[:, 1],
        "pred_biomass":     pred_full[:, 0],
        "pred_product":     pred_full[:, 1],
        "biomass_label":    biomass_label_test,
        "product_label":    product_label_test,
        "biomass_pred_clf": biomass_pred_test,
        "product_pred_clf": product_pred_test,
    })

    trial_rows = []
    for t in trials.trials:
        res = t.get("result", {})
        trial_rows.append({
            "loss_mae_scaled": res.get("loss"),
            "avg_mae":         res.get("avg_mae"),
            "params":          str(res.get("params", {})),
        })
    trials_df = pd.DataFrame(trial_rows).sort_values("loss_mae_scaled")

    with pd.ExcelWriter(REPORT_XLSX) as w:
        clf_cv_df.to_excel(          w, sheet_name="Classifiers_CV",        index=False)
        clf_test_df.to_excel(        w, sheet_name="Classifiers_Test",      index=False)
        ann_df.to_excel(             w, sheet_name="ANN_Metrics",           index=False)
        pipeline_df_metrics.to_excel(w, sheet_name="Pipeline_Metrics",      index=False)
        predictions_df.to_excel(     w, sheet_name="Pipeline_Predictions",  index=False)
        pd.DataFrame([best_params]).to_excel(w, sheet_name="BestParams",    index=False)
        trials_df.to_excel(          w, sheet_name="HyperoptTrials",        index=False)
        pd.DataFrame(ann_report["history"]).to_excel(w, sheet_name="Train_History", index=False)

    print(f"\nRelatorio guardado: {REPORT_XLSX}")


def main() -> None:
    """Funcao principal: carrega os dados, treina os classificadores e a ANN, avalia o pipeline e guarda os resultados."""
    print("A carregar dataset multi-hot")
    X, y, biomass_label, product_label, strata, gene_cols, df = load_multihot(DATA_XLSX, DATA_SHEET)

    print(f"N total = {len(df)}")
    print(f"Viaveis (biomass > 0): {biomass_label.sum()} ({100*biomass_label.mean():.1f}%)")
    print(f"Produto positivo (product > 0): {product_label.sum()} ({100*product_label.mean():.1f}%)")

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
    )
    X_train, X_test       = X[train_idx],             X[test_idx]
    y_train, y_test       = y[train_idx],             y[test_idx]
    bl_train, bl_test     = biomass_label[train_idx], biomass_label[test_idx]
    pl_train, pl_test     = product_label[train_idx], product_label[test_idx]
    strata_train          = strata[train_idx]

    print(f"\nTrain={len(train_idx)} | Test={len(test_idx)}")

    print(" CLASSIFICADORES SVM (multi-hot) ")
    clf1, clf2, clf_scaler, cv1, cv2 = train_svm_classifiers(X_train, bl_train, pl_train)

    X_test_s = clf_scaler.transform(X_test)

    clf1_pred = clf1.predict(X_test_s)
    clf1_prob = clf1.predict_proba(X_test_s)[:, 1]
    clf1_metrics = compute_clf_metrics(bl_test, clf1_pred, clf1_prob, "Classificador 1 — biomass_positive")

    viable_test_mask = bl_test == 1
    clf2_pred_viable = clf2.predict(X_test_s[viable_test_mask])
    clf2_prob_viable = clf2.predict_proba(X_test_s[viable_test_mask])[:, 1]
    clf2_metrics = compute_clf_metrics(
        pl_test[viable_test_mask], clf2_pred_viable, clf2_prob_viable,
        "Classificador 2 — product_positive (apenas viaveis)"
    )

    print("ANN REGRESSORA (multi-hot) ")
    reg_mask_train = (bl_train == 1) & (pl_train == 1)
    reg_mask_test  = (bl_test  == 1) & (pl_test  == 1)

    X_reg_train      = X_train[reg_mask_train]
    y_reg_train      = y_train[reg_mask_train]
    strata_reg_train = strata_train[reg_mask_train]

    X_reg_test  = X_test[reg_mask_test]
    y_reg_test  = y_test[reg_mask_test]

    print(f"Amostras para regressao (treino): {len(X_reg_train)}")
    print(f"Amostras para regressao (teste):  {len(X_reg_test)}")

    print("\nA correr Hyperopt")
    space     = build_search_space(max_layers=5)
    trials    = Trials()
    objective = make_ann_objective(X_reg_train, y_reg_train, strata_reg_train)
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest,
        max_evals=MAX_EVALS, trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE)
    )
    best_params = clean_params(space_eval(space, best))

    print("\nMelhores hiperparametros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    ann_model, x_scaler_ann, y_scaler_ann, ann_report = train_final_ann(
        X_reg_train, y_reg_train,
        X_reg_test,  y_reg_test,
        best_params
    )

    ann_metrics = compute_ann_metrics(
        y_reg_test,
        ann_report["pred_test"],
        "ANN Regressora (apenas amostras viaveis com product > 0)"
    )

    print(" PIPELINE COMPLETO")
    pred_full, biomass_pred, product_pred = predict_pipeline(
        X_test, clf1, clf2, clf_scaler,
        ann_model, x_scaler_ann, y_scaler_ann,
        threshold=VIABILITY_THRESHOLD
    )

    pipeline_metrics = compute_ann_metrics(
        y_test, pred_full,
        "Pipeline Completo (todo o conjunto de teste)"
    )

    print(f"\n  Previstos como inviavel (biomass=0):        {(biomass_pred==0).sum()}")
    print(f"  Previstos como viavel sem produto:           {((biomass_pred==1)&(product_pred==0)).sum()}")
    print(f"  Previstos como viavel com produto (ANN):     {((biomass_pred==1)&(product_pred==1)).sum()}")
    print(f"\n  Falsos positivos clf1 (inviavel->viavel):    {((biomass_pred==1)&(bl_test==0)).sum()}")
    print(f"  Falsos negativos clf1 (viavel->inviavel):    {((biomass_pred==0)&(bl_test==1)).sum()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(clf1,         CLF1_PATH)
    joblib.dump(clf2,         CLF2_PATH)
    joblib.dump(clf_scaler,   CLF_SCALER_PATH)
    ann_model.save(MODEL_PATH)
    joblib.dump(x_scaler_ann, XSCALER_PATH)
    joblib.dump(y_scaler_ann, YSCALER_PATH)

    metadata = {
        "pipeline":        ["clf1_biomass_svm_multihot", "clf2_product_svm_multihot", "ann_regressor_multihot"],
        "encoding":        "multi-hot em tudo",
        "threshold":       VIABILITY_THRESHOLD,
        "n_gene_features": len(gene_cols),
        "best_ann_params": best_params,
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    save_report(
        best_params, trials, ann_report,
        clf1_metrics, clf2_metrics, ann_metrics, pipeline_metrics,
        cv1, cv2,
        y_test, pred_full,
        bl_test, pl_test,
        biomass_pred, product_pred
    )

    print("\nGuardado em:", OUT_DIR)


if __name__ == "__main__":
    main()
