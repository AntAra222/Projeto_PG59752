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

DATA_XLSX = "dataset_final_ann.xlsx"
DATA_SHEET = 0

RAW_GLUCOSE_COL = "glucose_uptake(mmol/gDW/h)"
RAW_O2_COL = "o2_uptake(mmol/gDW/h)"
RAW_NK_COL = "num_knockouts"
RAW_BIOMASS_COL = "biomass(1/h)"
RAW_PRODUCT_COL = "product_flux(mmol/gDW/h)"

GLUCOSE_COL = "Glucose"
O2_COL = "O2"
NK_COL = "NK"
BIOMASS_COL = "Biomass"
PRODUCT_COL = "Product"

GENE_COLS = ["g1", "g2", "g3", "g4", "g5", "g6"]
NUMERIC_COLS = [GLUCOSE_COL, O2_COL, NK_COL]
DECISION_VARS = NUMERIC_COLS + GENE_COLS
TARGET_OBJECTIVES = [BIOMASS_COL, PRODUCT_COL]

COLUMN_RENAME = {
    RAW_GLUCOSE_COL: GLUCOSE_COL,
    RAW_O2_COL: O2_COL,
    RAW_NK_COL: NK_COL,
    RAW_BIOMASS_COL: BIOMASS_COL,
    RAW_PRODUCT_COL: PRODUCT_COL,
}

BOUNDS = {
    GLUCOSE_COL: (0.1, 20.0),
    O2_COL: (0.1, 20.0),
    NK_COL: (1, 6),
    "g1": (0, 1516),
    "g2": (0, 1516),
    "g3": (0, 1516),
    "g4": (0, 1516),
    "g5": (0, 1516),
    "g6": (0, 1516),
}

N_GENE_IDS = 1517       #Nº de genes 0-1516 
MAX_NK = 6              #Nº Max de knockouts
N_SPLITS = 5            #Nº de folds na crossvalidation
RANDOM_STATE = 24
MAX_EVALS = 200         #Nº de combinações de hiperparametros
PATIENCE = 25           #Se nao melhorar em 25 epohcs para
VERBOSE_FIT = 0
TEST_SIZE = 0.15        #Percentagem de dados para o teste

OUT_DIR = "metabolic_ann"
MODEL_PATH = os.path.join(OUT_DIR, "surrogate_metabolic.keras")
XSCALER_PATH = os.path.join(OUT_DIR, "x_numeric_scaler.joblib")
YSCALER_PATH = os.path.join(OUT_DIR, "y_scaler.joblib")
METADATA_JSON = os.path.join(OUT_DIR, "model_metadata.json")
REPORT_XLSX = os.path.join(OUT_DIR, "training_report.xlsx")


def set_seeds(seed: int = 24) -> None:
    """
    Garante que cada vez que corra o código o resultado seja sempre o mesmo
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


set_seeds(RANDOM_STATE)
tf.config.run_functions_eagerly(False)


def validate_gene_structure(df: pd.DataFrame) -> None:
    """
    Verifica se a estrutura dos genes está correta, vê o numero de knockouts e depois ve se so tem esse numero de genes inativos
    """
    bad_rows = []
    for idx, row in df.iterrows():
        nk = int(row[NK_COL])
        for i, col in enumerate(GENE_COLS, start=1):
            val = int(row[col])
            if i > nk and val != 0:
                bad_rows.append(idx)
                break
    if bad_rows:
        raise ValueError(f"Existem {len(bad_rows)} linhas em que genes inativos não estão a 0. Exemplos: {bad_rows[:10]}")


def load_and_preprocess(excel_path: str, sheet) -> pd.DataFrame:
    """ 
    Carrega o ficheiro Excel e limpa/valida os dados antes de os usar no treino. Remove NA e duplicados, mete os nomes mais curtos, 
    verifica se tem todas as colunas necessárias, etc. 
    """
    df = pd.read_excel(excel_path, sheet_name=sheet).drop_duplicates().reset_index(drop=True)
    df = df.rename(columns=COLUMN_RENAME)
    required = DECISION_VARS + TARGET_OBJECTIVES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltam colunas no Excel: {missing}")
    df = df[required].copy()
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    df[NK_COL] = df[NK_COL].round().astype(int)
    for c in GENE_COLS:
        df[c] = df[c].round().astype(int)
    for c, (mn, mx) in BOUNDS.items():
        outside = ~df[c].between(mn, mx)
        if outside.any():
            raise ValueError(f"Coluna '{c}' tem {outside.sum()} valores fora dos limites [{mn}, {mx}].")
    validate_gene_structure(df)
    return df


def build_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Separa os inputs numericos (X_num), os genes (X_genes), os objetivos a prever (y) e o NK (strata) para garatinr que os folds são bem divididos
    """
    X_num = df[NUMERIC_COLS].to_numpy(dtype=float)
    X_genes = df[GENE_COLS].to_numpy(dtype=np.int32)
    y = df[TARGET_OBJECTIVES].to_numpy(dtype=float)
    strata = df[NK_COL].to_numpy(dtype=int)
    return X_num, X_genes, y, strata


def make_model_inputs(X_num_s: np.ndarray, X_genes: np.ndarray) -> Dict[str, np.ndarray]:
    """ 
    Formata os dados no formato exato que o Keras espera para entrar na rede neuronal
    """
    inputs = {"numeric": X_num_s.astype("float32")}
    for j, col in enumerate(GENE_COLS):
        inputs[col] = X_genes[:, j].astype("int32")
    return inputs


def build_model(output_dim: int, params: Dict) -> keras.Model:
    """
    Constrói a arquitetura completa da rede neuronal
    """
    reg = regularizers.l2(params["l2"]) if params["l2"] > 0 else None
    numeric_input = layers.Input(shape=(len(NUMERIC_COLS),), name="numeric")
    input_layers = [numeric_input]
    concat_parts = [numeric_input]
    emb_dim = int(params["embedding_dim"])
    for col in GENE_COLS:
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
    model = keras.Model(inputs=input_layers, outputs=outputs, name="metabolic_surrogate_ann")
    opt_name = params["optimizer"]
    lr = float(params["learning_rate"])
    if opt_name == "adam":
        opt = keras.optimizers.Adam(learning_rate=lr)
    elif opt_name == "nadam":
        opt = keras.optimizers.Nadam(learning_rate=lr)
    elif opt_name == "rmsprop":
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Otimizador desconhecido: {opt_name}")
    model.compile(optimizer=opt, loss="mae", metrics=["mae"])
    return model


def build_search_space(max_layers: int = 5) -> Dict:
    """
     Define o espaço de pesquisa de hiperparâmetros para otimização
    bayesiana com Hyperopt.

    Esta função constrói um dicionário de distribuições de probabilidade
    que representa todas as configurações possíveis da rede neuronal.
    O Hyperopt utiliza este espaço para gerar e avaliar combinações de
    hiperparâmetros ao longo de MAX_EVALS tentativas, guiando a pesquisa
    para as regiões mais promissoras através do algoritmo TPE
    (Tree-structured Parzen Estimator).
    """

    acts = ["relu", "tanh", "elu", "selu", "softplus"]
    space = {
        "num_layers": hp.choice("num_layers", [1, 2, 3, 4, 5]),
        "optimizer": hp.choice("optimizer", ["adam", "nadam", "rmsprop"]),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-4), np.log(5e-2)),
        "epochs": hp.quniform("epochs", 80, 700, 1),
        "batch_size": hp.choice("batch_size", [16, 32, 64, 128]),
        "l2": hp.choice("l2", [0.0, 1e-7, 1e-6, 1e-5, 1e-4]),
        "batchnorm": hp.choice("batchnorm", [0, 1]),
        "embedding_dim": hp.choice("embedding_dim", [4, 8, 12, 16, 24, 32]),
    }
    for i in range(max_layers):
        space[f"units_{i}"] = hp.quniform(f"units_{i}", 16, 512, 1)
        space[f"dropout_{i}"] = hp.uniform(f"dropout_{i}", 0.0, 0.45)
        space[f"activation_{i}"] = hp.choice(f"activation_{i}", acts)
    return space


def clean_params(best_params: Dict) -> Dict:
    """
    O Hyperopt quando devolve os melhores parâmetros, devolve-os com tipos errados e informação a mais.
    """
    nl = int(best_params["num_layers"])
    cleaned = {
        "num_layers": nl,
        "optimizer": best_params["optimizer"],
        "learning_rate": float(best_params["learning_rate"]),
        "epochs": int(best_params["epochs"]),
        "batch_size": int(best_params["batch_size"]),
        "l2": float(best_params["l2"]),
        "batchnorm": int(best_params["batchnorm"]),
        "embedding_dim": int(best_params["embedding_dim"]),
    }
    for i in range(nl):
        cleaned[f"units_{i}"] = int(best_params[f"units_{i}"])
        cleaned[f"dropout_{i}"] = float(best_params[f"dropout_{i}"])
        cleaned[f"activation_{i}"] = best_params[f"activation_{i}"]
    return cleaned


def make_objective(X_num, X_genes, y, strata):
    """
    Cria e devolve a função objetivo utilizada pelo Hyperopt para avaliar
    cada combinação de hiperparâmetros.
    """
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    def _objective(params: Dict):
        params = clean_params(params)
        fold_mae = []
        per_output_mae_acc = []
        for tr_idx, va_idx in skf.split(X_num, strata):
            Xn_tr, Xn_va = X_num[tr_idx], X_num[va_idx]
            Xg_tr, Xg_va = X_genes[tr_idx], X_genes[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            x_scaler = StandardScaler().fit(Xn_tr)
            Xn_tr_s = x_scaler.transform(Xn_tr)
            Xn_va_s = x_scaler.transform(Xn_va)
            y_scaler = MinMaxScaler().fit(y_tr)
            y_tr_s = y_scaler.transform(y_tr)
            y_va_s = y_scaler.transform(y_va)
            model = build_model(y_tr_s.shape[1], params)
            es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0)
            model.fit(
                make_model_inputs(Xn_tr_s, Xg_tr),
                y_tr_s,
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
        avg_mae_per_output = np.mean(np.vstack(per_output_mae_acc), axis=0)
        return {
            "loss": avg_mae,
            "status": STATUS_OK,
            "avg_mae": avg_mae,
            "avg_mae_pct": float(100.0 * avg_mae),
            "avg_mae_per_output": avg_mae_per_output.tolist(),
            "params": params,
        }
    return _objective


def train_final_model(X_num_train, X_genes_train, y_train, X_num_test, X_genes_test, y_test, best_params: Dict):
    """
    Depois do Hyperopt encontrar os melhores hiperparâmetros, esta função treina o modelo definitivo
    """
    x_scaler = StandardScaler().fit(X_num_train)
    Xn_train_s = x_scaler.transform(X_num_train)
    Xn_test_s = x_scaler.transform(X_num_test)
    y_scaler = MinMaxScaler().fit(y_train)
    y_train_s = y_scaler.transform(y_train)
    y_test_s = y_scaler.transform(y_test)
    model = build_model(y_train_s.shape[1], best_params)
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=0)
    hist = model.fit(
        make_model_inputs(Xn_train_s, X_genes_train),
        y_train_s,
        validation_data=(make_model_inputs(Xn_test_s, X_genes_test), y_test_s),
        epochs=int(best_params["epochs"]),
        batch_size=int(best_params["batch_size"]),
        callbacks=[es],
        verbose=VERBOSE_FIT,
    )
    pred_train_s = model.predict(make_model_inputs(Xn_train_s, X_genes_train), verbose=0)
    pred_test_s = model.predict(make_model_inputs(Xn_test_s, X_genes_test), verbose=0)
    pred_train = y_scaler.inverse_transform(pred_train_s)
    pred_test = y_scaler.inverse_transform(pred_test_s)
    report = {
        "train_mae_scaled": float(mean_absolute_error(y_train_s, pred_train_s)),
        "test_mae_scaled": float(mean_absolute_error(y_test_s, pred_test_s)),
        "train_mae_original_avg": float(mean_absolute_error(y_train, pred_train)),
        "test_mae_original_avg": float(mean_absolute_error(y_test, pred_test)),
        "train_mae_scaled_per_output": np.mean(np.abs(pred_train_s - y_train_s), axis=0).tolist(),
        "test_mae_scaled_per_output": np.mean(np.abs(pred_test_s - y_test_s), axis=0).tolist(),
        "train_mae_original_per_output": np.mean(np.abs(pred_train - y_train), axis=0).tolist(),
        "test_mae_original_per_output": np.mean(np.abs(pred_test - y_test), axis=0).tolist(),
        "test_r2_per_output": [float(r2_score(y_test[:, j], pred_test[:, j])) for j in range(y_test.shape[1])],
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "y_test": y_test,
        "pred_test": pred_test,
    }
    return model, x_scaler, y_scaler, report


def save_report(best_params: Dict, trials: Trials, final_report: Dict, df: pd.DataFrame) -> None:
    """
    Guarda o relatório num excel
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    trial_rows = []
    for t in trials.trials:
        res = t.get("result", {})
        p = res.get("params", {})
        trial_rows.append({
            "loss_mae_scaled": res.get("loss"),
            "avg_mae_scaled": res.get("avg_mae"),
            "avg_mae_pct_scaled": res.get("avg_mae_pct"),
            "params": str(p),
        })
    trials_df = pd.DataFrame(trial_rows).sort_values("loss_mae_scaled")
    summary_df = pd.DataFrame([{
        "N": len(df),
        "inputs": ", ".join(DECISION_VARS),
        "outputs_predicted": ", ".join(TARGET_OBJECTIVES),
        "train_mae_scaled": final_report["train_mae_scaled"],
        "test_mae_scaled": final_report["test_mae_scaled"],
        "train_mae_original_avg": final_report["train_mae_original_avg"],
        "test_mae_original_avg": final_report["test_mae_original_avg"],
    }])
    per_output_df = pd.DataFrame({
        "objective": TARGET_OBJECTIVES,
        "train_mae_scaled": final_report["train_mae_scaled_per_output"],
        "test_mae_scaled": final_report["test_mae_scaled_per_output"],
        "train_mae_original": final_report["train_mae_original_per_output"],
        "test_mae_original": final_report["test_mae_original_per_output"],
        "test_r2": final_report["test_r2_per_output"],
    })
    test_pred_df = pd.DataFrame()
    for j, obj in enumerate(TARGET_OBJECTIVES):
        test_pred_df[f"true_{obj}"] = final_report["y_test"][:, j]
        test_pred_df[f"pred_{obj}"] = final_report["pred_test"][:, j]
        test_pred_df[f"abs_error_{obj}"] = np.abs(final_report["pred_test"][:, j] - final_report["y_test"][:, j])
    with pd.ExcelWriter(REPORT_XLSX) as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        pd.DataFrame([best_params]).to_excel(w, sheet_name="BestParams", index=False)
        trials_df.to_excel(w, sheet_name="HyperoptTrials", index=False)
        per_output_df.to_excel(w, sheet_name="Metrics_per_output", index=False)
        pd.DataFrame(final_report["history"]).to_excel(w, sheet_name="Train_History", index=False)
        test_pred_df.to_excel(w, sheet_name="Test_predictions", index=False)
        df.describe().T.to_excel(w, sheet_name="Data_describe")
        df[NK_COL].value_counts().sort_index().rename_axis(NK_COL).reset_index(name="count").to_excel(w, sheet_name="NK_distribution", index=False)


def save_metadata(best_params: Dict) -> None:
    """
    Guarda informaçao do modelo para usar no futuro sem ter de o correr outra vez
    """
    metadata = {
        "decision_vars": DECISION_VARS,
        "numeric_cols": NUMERIC_COLS,
        "gene_cols": GENE_COLS,
        "target_objectives_predicted_by_ann": TARGET_OBJECTIVES,
        "optimization_objectives_later": {
            "maximize": [PRODUCT_COL, BIOMASS_COL],
            "minimize_known_decision_vars": [NK_COL, GLUCOSE_COL],
        },
        "bounds": BOUNDS,
        "gene_encoding": {
            "0": "inactive gene slot when position i > NK",
            "1516": "s0001",
            "1..1515": "gene IDs",
        },
        "best_params": best_params,
    }
    with open(METADATA_JSON, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    df = load_and_preprocess(DATA_XLSX, DATA_SHEET)
    X_num, X_genes, y, strata = build_xy(df)
    print(f"N={len(df)} | X_numeric_dim={X_num.shape[1]} | X_gene_slots={X_genes.shape[1]} | y_dim={y.shape[1]}")
    print("Inputs:", DECISION_VARS)
    print("Outputs previstos pela ANN:", TARGET_OBJECTIVES)
    print("Distribuição NK:")
    print(df[NK_COL].value_counts().sort_index())
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strata)
    X_num_train, X_num_test = X_num[train_idx], X_num[test_idx]
    X_genes_train, X_genes_test = X_genes[train_idx], X_genes[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    strata_train = strata[train_idx]
    print(f"Train={len(train_idx)} | Test={len(test_idx)}")
    space = build_search_space(max_layers=5)
    trials = Trials()
    objective = make_objective(X_num_train, X_genes_train, y_train, strata_train)
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=MAX_EVALS, trials=trials, rstate=np.random.default_rng(RANDOM_STATE))
    best_params = clean_params(space_eval(space, best))
    print("\nMelhores hiperparâmetros:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    model, x_scaler, y_scaler, final_report = train_final_model(
        X_num_train, X_genes_train, y_train,
        X_num_test, X_genes_test, y_test,
        best_params,
    )
    os.makedirs(OUT_DIR, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(x_scaler, XSCALER_PATH)
    joblib.dump(y_scaler, YSCALER_PATH)
    save_metadata(best_params)
    save_report(best_params, trials, final_report, df)
    print("\nGuardado:")
    print(f"  Model:    {MODEL_PATH}")
    print(f"  X scaler: {XSCALER_PATH}")
    print(f"  Y scaler: {YSCALER_PATH}")
    print(f"  Metadata: {METADATA_JSON}")
    print(f"  Report:   {REPORT_XLSX}")
    print("\nResumo final:")
    print(f"  Train MAE scaled: {final_report['train_mae_scaled']:.5f}")
    print(f"  Test  MAE scaled: {final_report['test_mae_scaled']:.5f}")
    for obj, mae_o, r2 in zip(TARGET_OBJECTIVES, final_report["test_mae_original_per_output"], final_report["test_r2_per_output"]):
        print(f"  {obj}: Test MAE original={mae_o:.6g} | R2={r2:.4f}")


if __name__ == "__main__":
    main()
