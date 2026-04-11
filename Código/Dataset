import cobra
import random
import pandas as pd
import numpy as np

model = cobra.io.read_sbml_model("iML1515.xml")

model.reactions.get_by_id("EX_trp__L_e").lower_bound = 0
model.reactions.get_by_id("EX_trp__L_e").upper_bound = 1000
product_rxn_id = "EX_trp__L_e"

biomass_candidates = [rxn.id for rxn in model.reactions if "biomass" in rxn.id.lower()]
if not biomass_candidates:
    raise ValueError("Nao ha reacoes de biomassa")
biomass_rxn_id = biomass_candidates[0]
print("Reação de biomassa:", biomass_rxn_id)

with model:
    model.reactions.get_by_id("EX_glc__D_e").lower_bound = -10
    model.reactions.get_by_id("EX_o2_e").lower_bound = -5
    model.objective = biomass_rxn_id
    wt_solution = model.optimize()
    wt_growth = wt_solution.objective_value
print(f"Wild-type growth (glc=-10, o2=-5): {wt_growth:.6f}")

GLUCOSE_MIN, GLUCOSE_MAX = 0.1, 20.0
O2_MIN,      O2_MAX      = 0.1, 20.0

candidate_genes = [gene.id for gene in model.genes]
print(f"Número de genes candidatos: {len(candidate_genes)}")

n_simulations = 5000
results = []

for i in range(n_simulations):
    glucose_uptake = random.uniform(GLUCOSE_MIN, GLUCOSE_MAX)
    o2_uptake      = random.uniform(O2_MIN, O2_MAX)
    n_knockouts    = random.randint(1, 6)
    knockout_genes = random.sample(candidate_genes, n_knockouts)

    with model:
        model.reactions.get_by_id("EX_glc__D_e").lower_bound = -glucose_uptake
        model.reactions.get_by_id("EX_o2_e").lower_bound     = -o2_uptake

        for gene_id in knockout_genes:
            model.genes.get_by_id(gene_id).knock_out()

        model.objective = biomass_rxn_id
        sol_growth = model.optimize()
        if sol_growth.status != "optimal":
            continue
        max_growth = sol_growth.objective_value

        if max_growth < 0.1 * wt_growth:
            results.append({
                "glucose_uptake": round(glucose_uptake, 4),
                "o2_uptake":      round(o2_uptake, 4),
                "knockouts":      ";".join(knockout_genes),
                "num_knockouts":  n_knockouts,
                "biomass":        round(max(max_growth, 0), 6),
                "product_flux":   0.0
            })
            continue

        model.reactions.get_by_id(biomass_rxn_id).lower_bound = max_growth * 0.1
        model.objective = product_rxn_id
        sol_product = model.optimize()

        if sol_product.status != "optimal":
            results.append({
                "glucose_uptake": round(glucose_uptake, 4),
                "o2_uptake":      round(o2_uptake, 4),
                "knockouts":      ";".join(knockout_genes),
                "num_knockouts":  n_knockouts,
                "biomass":        round(max(max_growth, 0), 6),
                "product_flux":   0.0
            })
            continue

        results.append({
            "glucose_uptake": round(glucose_uptake, 4),
            "o2_uptake":      round(o2_uptake, 4),
            "knockouts":      ";".join(knockout_genes),
            "num_knockouts":  n_knockouts,
            "biomass":        round(max(max_growth, 0), 6),
            "product_flux":   round(max(sol_product.objective_value, 0), 6)
        })

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{n_simulations} simulações concluídas")

df = pd.DataFrame(results)

df.to_csv("dataset_combined.csv", index=False)
