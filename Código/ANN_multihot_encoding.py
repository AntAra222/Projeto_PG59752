import os
import json
import random
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval

# ── Dataset multi-hot ──────────────────────────────────────────
DATA_XLSX  = "dataset_multihot.xlsx"
DATA_SHEET = 0

# Colunas do dataset multihot
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

COLUMN_RENAME = {
    RAW_GLUCOSE_COL: GLUCOSE_COL,
    RAW_O2_COL:      O2_COL,
    RAW_NK_COL:      NK_COL,
    RAW_BIOMASS_COL: BIOMASS_COL,
    RAW_PRODUCT_COL: PRODUCT_COL,
}

NUMERIC_COLS      = [GLUCOSE_COL, O2_COL, NK_COL]
TARGET_OBJECTIVES = [BIOMASS_COL, PRODUCT_COL]

# As colunas dos genes são todas as colunas entre NK e Biomass
# Descobertas automaticamente no load_and_preprocess

N_SPLITS     = 5
RANDOM_STATE = 24
MAX_EVALS    = 300      # Nº de combinações de hiperparâmetros
PATIENCE     = 25         # Se não melhorar em 25 epochs para
VERBOSE_FIT  = 0
TEST_SIZE    = 0.15       # Percentagem de dados para teste

OUT_DIR       = "metabolic_ann_multihot"
MODEL_PATH    = os.path.join(OUT_DIR, "surrogate_metabolic_multihot.keras")
XSCALER_PATH  = os.path.join(OUT_DIR, "x_scaler_multihot.joblib")
YSCALER_PATH  = os.path.join(OUT_DIR, "y_scaler_multihot.joblib")
METADATA_JSON = os.path.join(OUT_DIR, "model_metadata_multihot.json")
REPORT_XLSX   = os.path.join(OUT_DIR, "training_report_multihot.xlsx")


def set_seeds(seed: int = 24) -> None:
    """
    Garante que cada vez que corra o código o resultado seja sempre o mesmo.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seeds(RANDOM_STATE)
tf.config.run_functions_eagerly(False)


def load_and_preprocess(excel_path: str, sheet) -> Tuple[pd.DataFrame, list]:
    """
    Carrega o dataset multi-hot e prepara os dados.
    Devolve o dataframe e a lista de colunas dos genes.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet).drop_duplicates().reset_index(drop=True)
    df = df.rename(columns=COLUMN_RENAME)

    # Descobrir colunas dos genes automaticamente
    # São todas as colunas entre NK e Biomass
    all_cols  = df.columns.tolist()
    idx_nk    = all_cols.index(NK_COL)
    idx_bio   = all_cols.index(BIOMASS_COL)
    gene_cols = all_cols[idx_nk + 1 : idx_bio]

    print(f"  Colunas de genes encontradas: {len(gene_cols)}")
    print(f"  Primeiro gene: {gene_cols[0]} | Último gene: {gene_cols[-1]}")

    required = NUMERIC_COLS + gene_cols + TARGET_OBJECTIVES
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    df[NK_COL] = df[NK_COL].round().astype(int)

    return df, gene_cols


def build_xy(df: pd.DataFrame, gene_cols: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa os inputs (X), os outputs (y) e o strata para cross-validation.

    Com multi-hot não há separação entre numéricos e genes —
    tudo entra como um único array X.
    """
    # Inputs: numéricos + genes binários — tudo junto
    input_cols = NUMERIC_COLS + gene_cols
    X       = df[input_cols].to_numpy(dtype=float)
    y       = df[TARGET_OBJECTIVES].to_numpy(dtype=float)
    strata  = df[NK_COL].to_numpy(dtype=int)
    return X, y, strata


def build_model(input_dim: int, output_dim: int, params: Dict) -> keras.Model:
    """
    Constrói a ANN para multi-hot encoding.

    Muito mais simples que a versão com Embeddings:
    - Um único input com todas as features (1519 valores)
    - Camadas Dense normais
    - Sem Embeddings
    """
    reg = regularizers.l2(params["l2"]) if params["l2"] > 0 else None

    # Um único input com tudo junto
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
    model   = keras.Model(inputs=inputs, outputs=outputs, name="metabolic_ann_multihot")

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
    """
    Define o espaço de pesquisa de hiperparâmetros para o Hyperopt.
    Igual ao original mas sem embedding_dim (não há Embeddings).
    """
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
    """
    Limpa e converte os parâmetros devolvidos pelo Hyperopt.
    """
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


def make_objective(X, y, strata):
    """
    Função objetivo para o Hyperopt.
    Avalia cada combinação de hiperparâmetros com cross-validation.
    """
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

            model = build_model(X_tr_s.shape[1], y_tr_s.shape[1], params)
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
        avg_mae_per_output = np.mean(np.vstack(per_output_mae_acc), axis=0)
        return {
            "loss":               avg_mae,
            "status":             STATUS_OK,
            "avg_mae":            avg_mae,
            "avg_mae_pct":        float(100.0 * avg_mae),
            "avg_mae_per_output": avg_mae_per_output.tolist(),
            "params":             params,
        }
    return _objective


def train_final_model(X_train, y_train, X_test, y_test, best_params: Dict):
    """
    Treina o modelo final com os melhores hiperparâmetros.
    """
    x_scaler  = StandardScaler().fit(X_train)
    X_train_s = x_scaler.transform(X_train)
    X_test_s  = x_scaler.transform(X_test)

    y_scaler  = MinMaxScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s  = y_scaler.transform(y_test)

    model = build_model(X_train_s.shape[1], y_train_s.shape[1], best_params)
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
        "train_mae_scaled":              float(mean_absolute_error(y_train_s, pred_train_s)),
        "test_mae_scaled":               float(mean_absolute_error(y_test_s,  pred_test_s)),
        "train_mae_original_avg":        float(mean_absolute_error(y_train, pred_train)),
        "test_mae_original_avg":         float(mean_absolute_error(y_test,  pred_test)),
        "train_mae_scaled_per_output":   np.mean(np.abs(pred_train_s - y_train_s), axis=0).tolist(),
        "test_mae_scaled_per_output":    np.mean(np.abs(pred_test_s  - y_test_s),  axis=0).tolist(),
        "train_mae_original_per_output": np.mean(np.abs(pred_train - y_train), axis=0).tolist(),
        "test_mae_original_per_output":  np.mean(np.abs(pred_test  - y_test),  axis=0).tolist(),
        "test_r2_per_output":            [float(r2_score(y_test[:, j], pred_test[:, j]))
                                          for j in range(y_test.shape[1])],
        "history":   {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "y_test":    y_test,
        "pred_test": pred_test,
    }
    return model, x_scaler, y_scaler, report


def save_report(best_params: Dict, trials: Trials, final_report: Dict,
                df: pd.DataFrame, gene_cols: list) -> None:
    """
    Guarda o relatório num Excel.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    trial_rows = []
    for t in trials.trials:
        res = t.get("result", {})
        trial_rows.append({
            "loss_mae_scaled":    res.get("loss"),
            "avg_mae_scaled":     res.get("avg_mae"),
            "avg_mae_pct_scaled": res.get("avg_mae_pct"),
            "params":             str(res.get("params", {})),
        })
    trials_df = pd.DataFrame(trial_rows).sort_values("loss_mae_scaled")

    summary_df = pd.DataFrame([{
        "N":                      len(df),
        "n_gene_features":        len(gene_cols),
        "total_input_features":   len(NUMERIC_COLS) + len(gene_cols),
        "outputs_predicted":      ", ".join(TARGET_OBJECTIVES),
        "train_mae_scaled":       final_report["train_mae_scaled"],
        "test_mae_scaled":        final_report["test_mae_scaled"],
        "train_mae_original_avg": final_report["train_mae_original_avg"],
        "test_mae_original_avg":  final_report["test_mae_original_avg"],
    }])

    per_output_df = pd.DataFrame({
        "objective":          TARGET_OBJECTIVES,
        "train_mae_scaled":   final_report["train_mae_scaled_per_output"],
        "test_mae_scaled":    final_report["test_mae_scaled_per_output"],
        "train_mae_original": final_report["train_mae_original_per_output"],
        "test_mae_original":  final_report["test_mae_original_per_output"],
        "test_r2":            final_report["test_r2_per_output"],
    })

    test_pred_df = pd.DataFrame()
    for j, obj in enumerate(TARGET_OBJECTIVES):
        test_pred_df[f"true_{obj}"]      = final_report["y_test"][:, j]
        test_pred_df[f"pred_{obj}"]      = final_report["pred_test"][:, j]
        test_pred_df[f"abs_error_{obj}"] = np.abs(
            final_report["pred_test"][:, j] - final_report["y_test"][:, j]
        )

    with pd.ExcelWriter(REPORT_XLSX) as w:
        summary_df.to_excel(    w, sheet_name="Summary",            index=False)
        pd.DataFrame([best_params]).to_excel(w, sheet_name="BestParams", index=False)
        trials_df.to_excel(     w, sheet_name="HyperoptTrials",     index=False)
        per_output_df.to_excel( w, sheet_name="Metrics_per_output", index=False)
        pd.DataFrame(final_report["history"]).to_excel(w, sheet_name="Train_History", index=False)
        test_pred_df.to_excel(  w, sheet_name="Test_predictions",   index=False)
        df.describe().T.to_excel(w, sheet_name="Data_describe")
        df[NK_COL].value_counts().sort_index().rename_axis(NK_COL).reset_index(
            name="count"
        ).to_excel(w, sheet_name="NK_distribution", index=False)


def save_metadata(best_params: Dict, gene_cols: list) -> None:
    """
    Guarda informação do modelo para usar no futuro.
    """
    metadata = {
        "gene_encoding":   "multi-hot (um 0/1 por gene)",
        "numeric_cols":    NUMERIC_COLS,
        "gene_cols":       gene_cols,
        "n_gene_features": len(gene_cols),
        "total_inputs":    len(NUMERIC_COLS) + len(gene_cols),
        "target_objectives_predicted_by_ann": TARGET_OBJECTIVES,
        "optimization_objectives_later": {
            "maximize": [PRODUCT_COL, BIOMASS_COL],
            "minimize": [NK_COL, GLUCOSE_COL],
        },
        "best_params": best_params,
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    # 1. Carregar dados
    print("A carregar dados...")
    df, gene_cols = load_and_preprocess(DATA_XLSX, DATA_SHEET)
    X, y, strata  = build_xy(df, gene_cols)

    print(f"N={len(df)} | X_dim={X.shape[1]} | y_dim={y.shape[1]}")
    print(f"Inputs: {len(NUMERIC_COLS)} numéricos + {len(gene_cols)} genes = {X.shape[1]} total")
    print(f"Outputs: {TARGET_OBJECTIVES}")
    print("Distribuição NK:")
    print(df[NK_COL].value_counts().sort_index())

    # 2. Split treino/teste
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata
    )
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    strata_train    = strata[train_idx]
    print(f"\nTrain={len(train_idx)} | Test={len(test_idx)}")

    # 3. Hyperopt
    print("\nA correr Hyperopt...")
    space     = build_search_space(max_layers=5)
    trials    = Trials()
    objective = make_objective(X_train, y_train, strata_train)
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest,
        max_evals=MAX_EVALS, trials=trials,
        rstate=np.random.default_rng(RANDOM_STATE)
    )
    best_params = clean_params(space_eval(space, best))

    print("\nMelhores hiperparâmetros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # 4. Treino final
    print("\nA treinar modelo final...")
    model, x_scaler, y_scaler, final_report = train_final_model(
        X_train, y_train, X_test, y_test, best_params
    )

    # 5. Guardar
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(x_scaler, XSCALER_PATH)
    joblib.dump(y_scaler, YSCALER_PATH)
    save_metadata(best_params, gene_cols)
    save_report(best_params, trials, final_report, df, gene_cols)

    print("\nGuardado:")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  X scaler: {XSCALER_PATH}")
    print(f"  Y scaler: {YSCALER_PATH}")
    print(f"  Metadata: {METADATA_JSON}")
    print(f"  Report:   {REPORT_XLSX}")

    print("\nResumo final:")
    print(f"  Train MAE scaled: {final_report['train_mae_scaled']:.5f}")
    print(f"  Test  MAE scaled: {final_report['test_mae_scaled']:.5f}")
    for obj, mae_o, r2 in zip(TARGET_OBJECTIVES,
                               final_report["test_mae_original_per_output"],
                               final_report["test_r2_per_output"]):
        print(f"  {obj}: Test MAE original={mae_o:.6g} | R2={r2:.4f}")


if __name__ == "__main__":
    main()
