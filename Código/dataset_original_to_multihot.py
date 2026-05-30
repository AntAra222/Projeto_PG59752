"""
create_multihot_dataset.py
--------------------------
Transforma o dataset_correto.xlsx para multi-hot encoding dos genes.

Uso:
    python create_multihot_dataset.py

Output:
    dataset_multihot.xlsx
"""

import numpy as np
import pandas as pd

INPUT_FILE  = "dataset_correto.xlsx"
OUTPUT_FILE = "dataset_multihot.xlsx"


def create_multihot_dataset(input_file: str, output_file: str) -> pd.DataFrame:
    """
    Transforma o dataset original para multi-hot encoding dos genes.

    Em vez de: knockouts = "b0002;b0237;b0898"
    Cria:      b0002=1, b0237=1, b0898=1, todos os outros=0

    Dataset final:
        [Glucose, O2, NK, b0002, b0003, ..., s0001, Biomass, Product]
    """
    # Carregar
    df = pd.read_excel(input_file)
    print(f"Dataset carregado: {df.shape}")

    # Extrair todos os genes únicos
    all_genes = sorted(set(
        gene.strip()
        for val in df["knockouts"]
        for gene in val.split(";")
    ))
    print(f"Total genes únicos: {len(all_genes)}")

    # Construir matriz multi-hot
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    matrix = np.zeros((len(df), len(all_genes)), dtype=np.int8)

    for i, val in enumerate(df["knockouts"]):
        for gene in val.split(";"):
            matrix[i, gene_to_idx[gene.strip()]] = 1

    # Juntar tudo
    df_final = pd.concat([
        df[["glucose_uptake(mmol/gDW/h)", "o2_uptake(mmol/gDW/h)", "num_knockouts"]].reset_index(drop=True),
        pd.DataFrame(matrix, columns=all_genes),
        df[["biomass(h\u207b\u00b9)", "product_flux(mmol/gDW/h)"]].reset_index(drop=True)
    ], axis=1)

    # Guardar
    df_final.to_excel(output_file, index=False)
    print(f"Guardado: {output_file} | Shape: {df_final.shape}")

    return df_final


if __name__ == "__main__":
    create_multihot_dataset(INPUT_FILE, OUTPUT_FILE)
