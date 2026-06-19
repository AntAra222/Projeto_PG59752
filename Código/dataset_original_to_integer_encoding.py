import pandas as pd

# load your file
df = pd.read_excel("dataset_correto.xlsx", sheet_name="dataset_combined")


df["knockouts_list"] = df["knockouts"].apply(
    lambda x: [g.strip().lower() for g in str(x).split(";") if g.strip()]
)

# build rank mapping for b genes
all_genes = sorted(set(g for row in df["knockouts_list"] for g in row))

b_genes = sorted([g for g in all_genes if g.startswith("b")])

gene_map = {g: i+1 for i, g in enumerate(b_genes)}

# process rows 
def process_row(genes):
    numeric = []
    
    for g in genes:
        if g == "s0001":
            numeric.append(1516)
        else:
            numeric.append(gene_map.get(g, 0))

    num_knockouts = len(numeric)

    # fixed size = 6 genes
    numeric = numeric[:6] + [0]*(6 - len(numeric))

    return pd.Series([num_knockouts] + numeric)

df[["num_knockouts", "g1", "g2", "g3", "g4", "g5", "g6"]] = df["knockouts_list"].apply(process_row)


# final column order
df = df[[
    "glucose_uptake(mmol/gDW/h)", "o2_uptake(mmol/gDW/h)", "num_knockouts",
    "g1", "g2", "g3", "g4", "g5", "g6",
    "biomass(h⁻¹)", "product_flux(mmol/gDW/h)"
]]

# save
df.to_csv("dataset_final_ann.csv", index=False)
