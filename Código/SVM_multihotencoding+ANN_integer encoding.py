"""
solution3_multihot_svm.py
--------------------------
Pipeline com dois classificadores SVM (multi-hot encoding) + ANN regressora
com embeddings para os genes (integer encoding).

Os classificadores usam multi-hot encoding (dataset_multihot.xlsx):
    - Cada gene tem uma coluna binaria (0/1)
    - Elimina o problema de permutacao e de integer encoding
    - O SVM consegue aprender "este gene especifico causa inviabilidade"

A ANN regressora usa integer encoding com embeddings (dataset_final_ann.xlsx):
    - Treinada apenas onde biomass > 0 AND product > 0

Estrutura:
    Classificador 1 (SVM + multihot): biomass_positive
    Classificador 2 (SVM + multihot): product_positive (apenas viaveis)
    Regressor (ANN + embeddings):     biomass + product (apenas ambos positivos)

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
    classification_report, mean_absolute_error, r2_score
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# Configuracao
# Dataset multi-hot para os classificadores
CLF_DATA_XLSX  = "dataset_multihot.xlsx"

# Dataset integer encoding para a ANN regressora
ANN_DATA_XLSX  = "dataset_final_ann.xlsx"
DATA_SHEET     = 0

RAW_GLUCOSE_COL = "glucose_uptake(mmol/gDW/h)"
RAW_O2_COL      = "o2_uptake(mmol/gDW/h)"
RAW_NK_COL      = "num_knockouts"
RAW_BIOMASS_COL = "biomass(h\u207b\u00b9)"
RAW_PRODUCT_COL = "product_flux(mmol/gDW/h)"
RAW_BIOMASS_COL_ANN = "biomass(1/h)"

GLUCOSE_COL = "Glucose"
O2_COL      = "O2"
NK_COL      = "NK"
BIOMASS_COL = "Biomass"
PRODUCT_COL = "Product"

GENE_COLS_ANN = ["g1", "g2", "g3", "g4", "g5", "g6"]
NUMERIC_COLS  = [GLUCOSE_COL, O2_COL, NK_COL]
TARGET_OBJECTIVES = [BIOMASS_COL, PRODUCT_COL]

BOUNDS_ANN = {
    GLUCOSE_COL: (0.1, 20.0),
    O2_COL:      (0.1, 20.0),
    NK_COL:      (1, 6),
    "g1": (0, 1516), "g2": (0, 1516), "g3": (0, 1516),
    "g4": (0, 1516), "g5": (0, 1516), "g6": (0, 1516),
}

N_GENE_IDS   = 1517
N_SPLITS     = 5
RANDOM_STATE = 24
MAX_EVALS    = 200
PATIENCE     = 25
VERBOSE_FIT  = 0
TEST_SIZE    = 0.15

VIABILITY_THRESHOLD = 0.5

OUT_DIR         = "solution3_multihot_svm"
CLF1_PATH       = os.path.join(OUT_DIR, "clf1_biomass.joblib")
CLF2_PATH       = os.path.join(OUT_DIR, "clf2_product.joblib")
CLF_SCALER_PATH = os.path.join(OUT_DIR, "clf_scaler.joblib")
MODEL_PATH      = os.path.join(OUT_DIR, "regressor_ann.keras")
XSCALER_PATH    = os.path.join(OUT_DIR, "x_numeric_scaler.joblib")
YSCALER_PATH    = os.path.join(OUT_DIR, "y_scaler.joblib")
METADATA_JSON   = os.path.join(OUT_DIR, "model_metadata.json")
REPORT_XLSX     = os.path.join(OUT_DIR, "training_report.xlsx")


# Seeds
def set_seeds(seed: int = 24) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seeds(RANDOM_STATE)
tf.config.run_functions_eagerly(False)


# Carregamento dataset multi-hot (classificadores)
def load_multihot(excel_path: str, sheet) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Carrega o dataset multi-hot para os classificadores SVM.
    Devolve X (numericos + genes binarios), labels e gene_cols.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet).drop_duplicates().reset_index(drop=True)

    # Renomear colunas
    rename = {
        RAW_GLUCOSE_COL:     GLUCOSE_COL,
        RAW_O2_COL:          O2_COL,
        RAW_NK_COL:          NK_COL,
        RAW_BIOMASS_COL:     BIOMASS_COL,
        RAW_PRODUCT_COL:     PRODUCT_COL,
    }
    df = df.rename(columns=rename)

    # Descobrir colunas dos genes automaticamente (entre NK e Biomass)
    all_cols  = df.columns.tolist()
    idx_nk    = all_cols.index(NK_COL)
    idx_bio   = all_cols.index(BIOMASS_COL)
    gene_cols = all_cols[idx_nk + 1: idx_bio]

    print(f"  Multi-hot: {len(gene_cols)} colunas de genes")

    # Input: numericos + genes binarios (tudo junto)
    input_cols = NUMERIC_COLS + gene_cols
    for c in input_cols + TARGET_OBJECTIVES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=input_cols + TARGET_OBJECTIVES).reset_index(drop=True)

    X             = df[input_cols].to_numpy(dtype=float)
    biomass_label = (df[BIOMASS_COL].to_numpy() > 0).astype(int)
    product_label = (df[PRODUCT_COL].to_numpy() > 0).astype(int)
    strata        = df[NK_COL].to_numpy(dtype=int)

    return X, biomass_label, product_label, strata, df, gene_cols


# Carregamento dataset ANN (integer encoding)
def load_ann_dataset(excel_path: str, sheet) -> pd.DataFrame:
    """
    Carrega o dataset integer encoding para a ANN regressora.
    """
    rename = {
        RAW_GLUCOSE_COL:     GLUCOSE_COL,
        RAW_O2_COL:          O2_COL,
        RAW_NK_COL:          NK_COL,
        RAW_BIOMASS_COL_ANN: BIOMASS_COL,
        RAW_PRODUCT_COL:     PRODUCT_COL,
    }
    decision_vars = NUMERIC_COLS + GENE_COLS_ANN

    df = pd.read_excel(excel_path, sheet_name=sheet).drop_duplicates().reset_index(drop=True)
    df = df.rename(columns=rename)

    required = decision_vars + TARGET_OBJECTIVES
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no dataset ANN: {missing}")

    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    df[NK_COL] = df[NK_COL].round().astype(int)
    for c in GENE_COLS_ANN:
        df[c] = df[c].round().astype(int)

    for c, (mn, mx) in BOUNDS_ANN.items():
        outside = ~df[c].between(mn, mx)
        if outside.any():
            raise ValueError(f"Coluna '{c}' tem {outside.sum()} valores fora dos limites [{mn}, {mx}].")

    return df


def prepare_ann_data(df: pd.DataFrame):
    X_num   = df[NUMERIC_COLS].to_numpy(dtype=float)
    X_genes = df[GENE_COLS_ANN].to_numpy(dtype=np.int32)
    y       = df[TARGET_OBJECTIVES].to_numpy(dtype=float)
    biomass_label = (df[BIOMASS_COL].to_numpy() > 0).astype(int)
    product_label = (df[PRODUCT_COL].to_numpy() > 0).astype(int)
    strata        = df[NK_COL].to_numpy(dtype=int)
    return X_num, X_genes, y, biomass_label, product_label, strata


def make_model_inputs(X_num_s: np.ndarray, X_genes: np.ndarray) -> Dict:
    inputs = {"numeric": X_num_s.astype("float32")}
    for j, col in enumerate(GENE_COLS_ANN):
        inputs[col] = X_genes[:, j].astype("int32")
    return inputs


# Classificadores SVM (multi-hot)
def train_svm_classifiers_multihot(
    X_train, biomass_label_train, product_label_train
):
    """
    Treina dois classificadores SVM com multi-hot encoding.

    Com multi-hot cada gene tem a sua propria coluna binaria (0/1).
    O SVM com kernel RBF consegue calcular distancias significativas
    entre pontos e aprender quais os genes que causam inviabilidade.
    """
    clf_scaler = StandardScaler().fit(X_train)
    X_train_s  = clf_scaler.transform(X_train)

    # -- Classificador 1: biomass_positive --
    print("A treinar Classificador 1 (biomass_positive) com SVM + multi-hot...")
    clf1 = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")

    cv_scores_clf1 = cross_val_score(
        clf1, X_train_s, biomass_label_train,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )
    print(f"  CV Accuracy clf1: {cv_scores_clf1.mean():.4f} +/- {cv_scores_clf1.std():.4f}")
    clf1.fit(X_train_s, biomass_label_train)

    # Classificador 2: product_positive (apenas viaveis) 
    viable_mask          = biomass_label_train == 1
    X_train_s_viable     = X_train_s[viable_mask]
    product_label_viable = product_label_train[viable_mask]

    print("A treinar Classificador 2 (product_positive, apenas viaveis) com SVM + multi-hot...")
    clf2 = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced")

    cv_scores_clf2 = cross_val_score(
        clf2, X_train_s_viable, product_label_viable,
        cv=StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy"
    )
    print(f"  CV Accuracy clf2: {cv_scores_clf2.mean():.4f} +/- {cv_scores_clf2.std():.4f}")
    clf2.fit(X_train_s_viable, product_label_viable)

    return clf1, clf2, clf_scaler, cv_scores_clf1, cv_scores_clf2


#  ANN Regressora (embeddings) 
def build_ann_model(output_dim: int, params: Dict) -> keras.Model:
    reg = regularizers.l2(params["l2"]) if params["l2"] > 0 else None

    numeric_input = layers.Input(shape=(len(NUMERIC_COLS),), name="numeric")
    input_layers  = [numeric_input]
    concat_parts  = [numeric_input]

    emb_dim = int(params["embedding_dim"])
    for col in GENE_COLS_ANN:
        gene_input = layers.Input(shape=(1,), dtype="int32", name=col)
        input_layers.append(gene_input)
        emb = layers.Embedding(input_dim=N_GENE_IDS, output_dim=emb_dim, name=f"emb_{col}")(gene_input)
        concat_parts.append(layers.Flatten(name=f"flat_{col}")(emb))

    x = layers.Concatenate(name="concat_numeric_genes")(concat_parts)

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
    model   = keras.Model(inputs=input_layers, outputs=outputs, name="regressor_ann_embeddings")

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
    acts  = ["relu", "tanh", "elu", "selu", "softplus"]
    space = {
        "num_layers":    hp.choice("num_layers",    [1, 2, 3, 4, 5]),
        "optimizer":     hp.choice("optimizer",     ["adam", "nadam", "rmsprop"]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(5e-2)),
        "epochs":        hp.quniform("epochs", 80, 700, 1),
        "batch_size":    hp.choice("batch_size",    [16, 32, 64, 128]),
        "l2":            hp.choice("l2",            [0.0, 1e-7, 1e-6, 1e-5, 1e-4]),
        "batchnorm":     hp.choice("batchnorm",     [0, 1]),
        "embedding_dim": hp.choice("embedding_dim", [4, 8, 12, 16, 24, 32]),
    }
    for i in range(max_layers):
        space[f"units_{i}"]      = hp.quniform(f"units_{i}", 16, 512, 1)
        space[f"dropout_{i}"]    = hp.uniform(f"dropout_{i}", 0.0, 0.45)
        space[f"activation_{i}"] = hp.choice(f"activation_{i}", acts)
    return space


def clean_params(best_params: Dict) -> Dict:
    nl = int(best_params["num_layers"])
    cleaned = {
        "num_layers":    nl,
        "optimizer":     best_params["optimizer"],
        "learning_rate": float(best_params["learning_rate"]),
        "epochs":        int(best_params["epochs"]),
        "batch_size":    int(best_params["batch_size"]),
        "l2":            float(best_params["l2"]),
        "batchnorm":     int(best_params["batchnorm"]),
        "embedding_dim": int(best_params["embedding_dim"]),
    }
    for i in range(nl):
        cleaned[f"units_{i}"]      = int(best_params[f"units_{i}"])
        cleaned[f"dropout_{i}"]    = float(best_params[f"dropout_{i}"])
        cleaned[f"activation_{i}"] = best_params[f"activation_{i}"]
    return cleaned


def make_ann_objective(X_num, X_genes, y, strata):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    def _objective(params: Dict):
        params = clean_params(params)
        fold_mae = []
        per_output_mae_acc = []

        for tr_idx, va_idx in skf.split(X_num, strata):
            Xn_tr, Xn_va = X_num[tr_idx], X_num[va_idx]
            Xg_tr, Xg_va = X_genes[tr_idx], X_genes[va_idx]
            y_tr,  y_va  = y[tr_idx], y[va_idx]

            x_scaler = StandardScaler().fit(Xn_tr)
            Xn_tr_s  = x_scaler.transform(Xn_tr)
            Xn_va_s  = x_scaler.transform(Xn_va)

            y_scaler = MinMaxScaler().fit(y_tr)
            y_tr_s   = y_scaler.transform(y_tr)
            y_va_s   = y_scaler.transform(y_va)

            model = build_ann_model(y_tr_s.shape[1], params)
            es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                               restore_best_weights=True, verbose=0)
            model.fit(
                make_model_inputs(Xn_tr_s, Xg_tr), y_tr_s,
                validation_data=(make_model_inputs(Xn_va_s, Xg_va), y_va_s),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[es],
                verbose=VERBOSE_FIT,
            )

            pred_s = model.predict(make_model_inputs(Xn_va_s, Xg_va), verbose=0)
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


def train_final_ann(X_num_train, X_genes_train, y_train,
                    X_num_test, X_genes_test, y_test, best_params: Dict):
    x_scaler   = StandardScaler().fit(X_num_train)
    Xn_train_s = x_scaler.transform(X_num_train)
    Xn_test_s  = x_scaler.transform(X_num_test)

    y_scaler   = MinMaxScaler().fit(y_train)
    y_train_s  = y_scaler.transform(y_train)
    y_test_s   = y_scaler.transform(y_test)

    model = build_ann_model(y_train_s.shape[1], best_params)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE,
                       restore_best_weights=True, verbose=0)
    hist = model.fit(
        make_model_inputs(Xn_train_s, X_genes_train), y_train_s,
        validation_data=(make_model_inputs(Xn_test_s, X_genes_test), y_test_s),
        epochs=int(best_params["epochs"]),
        batch_size=int(best_params["batch_size"]),
        callbacks=[es],
        verbose=VERBOSE_FIT,
    )

    pred_train_s = model.predict(make_model_inputs(Xn_train_s, X_genes_train), verbose=0)
    pred_test_s  = model.predict(make_model_inputs(Xn_test_s,  X_genes_test),  verbose=0)
    pred_train   = y_scaler.inverse_transform(pred_train_s)
    pred_test    = y_scaler.inverse_transform(pred_test_s)

    report = {
        "train_mae_scaled":              float(mean_absolute_error(y_train_s, pred_train_s)),
        "test_mae_scaled":               float(mean_absolute_error(y_test_s,  pred_test_s)),
        "train_mae_original_avg":        float(mean_absolute_error(y_train, pred_train)),
        "test_mae_original_avg":         float(mean_absolute_error(y_test,  pred_test)),
        "test_r2_per_output":            [float(r2_score(y_test[:, j], pred_test[:, j]))
                                          for j in range(y_test.shape[1])],
        "history":   {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "y_test":    y_test,
        "pred_test": pred_test,
    }
    return model, x_scaler, y_scaler, report


# Predicao final com pipeline completo 
def predict_pipeline(
    X_clf,          # input multi-hot para os classificadores
    X_num, X_genes, # inputs para a ANN
    clf1, clf2, clf_scaler,
    ann_model, x_scaler_ann, y_scaler_ann,
    threshold: float = VIABILITY_THRESHOLD
):
    """
    Pipeline completo:
        1. SVM clf1 (multi-hot) decide se biomass > 0
        2. SVM clf2 (multi-hot) decide se product > 0
        3. ANN (embeddings) preve os valores reais
    """
    n = len(X_num)
    predictions = np.zeros((n, 2))

    # Normalizar para o SVM
    X_clf_s = clf_scaler.transform(X_clf)

    # Classificador 1
    prob_biomass = clf1.predict_proba(X_clf_s)[:, 1]
    biomass_pos  = (prob_biomass >= threshold).astype(int)

    # Classificador 2 — apenas viaveis
    viable_idx  = np.where(biomass_pos == 1)[0]
    product_pos = np.zeros(n, dtype=int)

    if len(viable_idx) > 0:
        prob_product = clf2.predict_proba(X_clf_s[viable_idx])[:, 1]
        product_pos[viable_idx] = (prob_product >= threshold).astype(int)

    # ANN — ambos positivos
    both_pos_idx = np.where((biomass_pos == 1) & (product_pos == 1))[0]
    if len(both_pos_idx) > 0:
        Xn_s   = x_scaler_ann.transform(X_num[both_pos_idx])
        Xg     = X_genes[both_pos_idx]
        pred_s = ann_model.predict(make_model_inputs(Xn_s, Xg), verbose=0)
        pred   = y_scaler_ann.inverse_transform(pred_s)
        predictions[both_pos_idx] = pred

    # Biomass positivo mas product zero
    biomass_only_idx = np.where((biomass_pos == 1) & (product_pos == 0))[0]
    if len(biomass_only_idx) > 0:
        Xn_s   = x_scaler_ann.transform(X_num[biomass_only_idx])
        Xg     = X_genes[biomass_only_idx]
        pred_s = ann_model.predict(make_model_inputs(Xn_s, Xg), verbose=0)
        pred   = y_scaler_ann.inverse_transform(pred_s)
        predictions[biomass_only_idx, 0] = pred[:, 0]
        predictions[biomass_only_idx, 1] = 0.0

    return predictions, biomass_pos, product_pos


# Guardar relatorio 
def save_report(
    best_params, trials, ann_report,
    clf1_cv, clf2_cv,
    y_test_full, pred_full,
    biomass_label_test, product_label_test,
    biomass_pred_test, product_pred_test
) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    clf_df = pd.DataFrame({
        "classifier":       ["clf1_biomass", "clf2_product"],
        "cv_accuracy_mean": [clf1_cv.mean(), clf2_cv.mean()],
        "cv_accuracy_std":  [clf1_cv.std(),  clf2_cv.std()],
    })
    ann_df = pd.DataFrame([{
        "train_mae_scaled": ann_report["train_mae_scaled"],
        "test_mae_scaled":  ann_report["test_mae_scaled"],
        "r2_biomass":       ann_report["test_r2_per_output"][0],
        "r2_product":       ann_report["test_r2_per_output"][1],
    }])
    pipeline_df = pd.DataFrame({
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
        clf_df.to_excel(     w, sheet_name="Classifiers_CV",       index=False)
        ann_df.to_excel(     w, sheet_name="ANN_Metrics",          index=False)
        pipeline_df.to_excel(w, sheet_name="Pipeline_Predictions", index=False)
        pd.DataFrame([best_params]).to_excel(w, sheet_name="BestParams", index=False)
        trials_df.to_excel(  w, sheet_name="HyperoptTrials",       index=False)
        pd.DataFrame(ann_report["history"]).to_excel(w, sheet_name="Train_History", index=False)

    print(f"Relatorio guardado: {REPORT_XLSX}")


# Main 
def main() -> None:
    # 1. Carregar dataset multi-hot para classificadores
    print("A carregar dataset multi-hot (classificadores)...")
    X_clf, biomass_label, product_label, strata, df_clf, gene_cols = load_multihot(CLF_DATA_XLSX, DATA_SHEET)
    print(f"N total = {len(df_clf)}")
    print(f"Viaveis (biomass > 0): {biomass_label.sum()} ({100*biomass_label.mean():.1f}%)")
    print(f"Produto positivo (product > 0): {product_label.sum()} ({100*product_label.mean():.1f}%)")

    # 2. Carregar dataset integer encoding para ANN
    print("\nA carregar dataset integer encoding (ANN)...")
    df_ann = load_ann_dataset(ANN_DATA_XLSX, DATA_SHEET)
    X_num, X_genes, y, bl_ann, pl_ann, strata_ann = prepare_ann_data(df_ann)
    print(f"N ANN dataset = {len(df_ann)}")

    # 3. Split treino/teste — mesmo indice para ambos os datasets
    # Usa strata do dataset multi-hot (mesma ordem)
    idx = np.arange(len(df_clf))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
    )

    # Split classificadores (multi-hot)
    X_clf_train, X_clf_test       = X_clf[train_idx],         X_clf[test_idx]
    bl_train,    bl_test          = biomass_label[train_idx],  biomass_label[test_idx]
    pl_train,    pl_test          = product_label[train_idx],  product_label[test_idx]

    # Split ANN (integer encoding)
    X_num_train,   X_num_test     = X_num[train_idx],   X_num[test_idx]
    X_genes_train, X_genes_test   = X_genes[train_idx], X_genes[test_idx]
    y_train,       y_test         = y[train_idx],       y[test_idx]

    print(f"\nTrain={len(train_idx)} | Test={len(test_idx)}")

    # 4. Treinar classificadores SVM com multi-hot
    print("\n── Classificadores SVM (multi-hot encoding) ──")
    clf1, clf2, clf_scaler, cv1, cv2 = train_svm_classifiers_multihot(
        X_clf_train, bl_train, pl_train
    )

    # Avaliar no teste
    X_clf_test_s = clf_scaler.transform(X_clf_test)

    print("\nClassificador 1 - Test:")
    print(classification_report(bl_test, clf1.predict(X_clf_test_s),
                                 target_names=["inviavel", "viavel"]))

    viable_test_mask = bl_test == 1
    print("Classificador 2 - Test (apenas viaveis):")
    print(classification_report(
        pl_test[viable_test_mask],
        clf2.predict(X_clf_test_s[viable_test_mask]),
        target_names=["product=0", "product>0"]
    ))

    # 5. ANN regressora — apenas onde biomass > 0 AND product > 0
    print("\n── ANN Regressora (embeddings) ──")
    reg_mask_train = (bl_train == 1) & (pl_train == 1)
    reg_mask_test  = (bl_test  == 1) & (pl_test  == 1)

    X_num_reg_train   = X_num_train[reg_mask_train]
    X_genes_reg_train = X_genes_train[reg_mask_train]
    y_reg_train       = y_train[reg_mask_train]
    strata_reg_train  = strata[train_idx][reg_mask_train]

    X_num_reg_test    = X_num_test[reg_mask_test]
    X_genes_reg_test  = X_genes_test[reg_mask_test]
    y_reg_test        = y_test[reg_mask_test]

    print(f"Amostras para regressao (treino): {len(X_num_reg_train)}")
    print(f"Amostras para regressao (teste):  {len(X_num_reg_test)}")

    print("\nA correr Hyperopt para a ANN regressora...")
    space     = build_search_space(max_layers=5)
    trials    = Trials()
    objective = make_ann_objective(X_num_reg_train, X_genes_reg_train, y_reg_train, strata_reg_train)
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
        X_num_reg_train, X_genes_reg_train, y_reg_train,
        X_num_reg_test,  X_genes_reg_test,  y_reg_test,
        best_params
    )

    # 6. Pipeline completo no teste
    print("\n── Pipeline completo no teste ──")
    pred_full, biomass_pred, product_pred = predict_pipeline(
        X_clf_test,
        X_num_test, X_genes_test,
        clf1, clf2, clf_scaler,
        ann_model, x_scaler_ann, y_scaler_ann,
        threshold=VIABILITY_THRESHOLD
    )

    r2_biomass_full = r2_score(y_test[:, 0], pred_full[:, 0])
    r2_product_full = r2_score(y_test[:, 1], pred_full[:, 1])
    mae_full        = mean_absolute_error(y_test, pred_full)

    print(f"Pipeline - R2 Biomass: {r2_biomass_full:.4f}")
    print(f"Pipeline - R2 Product: {r2_product_full:.4f}")
    print(f"Pipeline - MAE total:  {mae_full:.6f}")

    # 7. Guardar modelos
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(clf1,         CLF1_PATH)
    joblib.dump(clf2,         CLF2_PATH)
    joblib.dump(clf_scaler,   CLF_SCALER_PATH)
    ann_model.save(MODEL_PATH)
    joblib.dump(x_scaler_ann, XSCALER_PATH)
    joblib.dump(y_scaler_ann, YSCALER_PATH)

    metadata = {
        "pipeline":          ["clf1_biomass_svm_multihot", "clf2_product_svm_multihot", "ann_regressor_embeddings"],
        "clf_encoding":      "multi-hot",
        "ann_encoding":      "integer (embeddings)",
        "threshold":         VIABILITY_THRESHOLD,
        "n_gene_features":   len(gene_cols),
        "best_ann_params":   best_params,
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    save_report(
        best_params, trials, ann_report,
        cv1, cv2,
        y_test, pred_full,
        bl_test, pl_test,
        biomass_pred, product_pred
    )

    print("\nGuardado em:", OUT_DIR)


if __name__ == "__main__":
    main()
