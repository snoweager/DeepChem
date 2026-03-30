"""
Download additional ADMET-relevant data from ChEMBL.
ChEMBL is the gold standard pharma bioactivity database —
used by AstraZeneca, Pfizer, GSK for internal ML models.
"""
import pandas as pd
import os
from chembl_webresource_client.new_client import new_client

os.makedirs("data/raw", exist_ok=True)

# ── 1. hERG inhibition from ChEMBL (IC50 data) ──────────────────
print("Fetching hERG IC50 data from ChEMBL...")
activity = new_client.activity
herg_data = activity.filter(
    target_chembl_id="CHEMBL240",  # hERG channel
    standard_type="IC50",
    standard_units="nM",
).only([
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_units",
])

rows = []
for rec in herg_data:
    smi = rec.get("canonical_smiles")
    val = rec.get("standard_value")
    if smi and val:
        try:
            ic50_nm = float(val)
            # Binary label: IC50 < 1000 nM = hERG blocker (positive)
            label = 1 if ic50_nm < 1000 else 0
            rows.append({
                "SMILES": smi,
                "hERG_IC50_nM": ic50_nm,
                "hERG_label": label,
                "source": "ChEMBL"
            })
        except (ValueError, TypeError):
            continue

df_herg = pd.DataFrame(rows).drop_duplicates("SMILES")
df_herg.to_csv("data/raw/chembl_herg.csv", index=False)
print(f"  Saved {len(df_herg):,} hERG molecules from ChEMBL")

# ── 2. CYP3A4 inhibition from ChEMBL ────────────────────────────
print("Fetching CYP3A4 inhibition data from ChEMBL...")
cyp_data = activity.filter(
    target_chembl_id="CHEMBL340",  # CYP3A4
    standard_type="IC50",
    standard_units="nM",
).only([
    "canonical_smiles",
    "standard_value",
])

rows = []
for rec in cyp_data:
    smi = rec.get("canonical_smiles")
    val = rec.get("standard_value")
    if smi and val:
        try:
            ic50_nm = float(val)
            label = 1 if ic50_nm < 1000 else 0
            rows.append({
                "Drug": smi,
                "CYP3A4": label,
                "CYP2C9": float("nan"),
                "CYP2D6": float("nan"),
                "source": "ChEMBL"
            })
        except (ValueError, TypeError):
            continue

df_cyp = pd.DataFrame(rows).drop_duplicates("Drug")
df_cyp.to_csv("data/raw/chembl_cyp3a4.csv", index=False)
print(f"  Saved {len(df_cyp):,} CYP3A4 molecules from ChEMBL")

# ── 3. LogP experimental values from ChEMBL ─────────────────────
print("Fetching experimental LogP from ChEMBL...")
props = new_client.activity.filter(
    standard_type="LogP",
).only(["canonical_smiles", "standard_value"])

rows = []
for rec in props[:2000]:  # cap at 2000 for speed
    smi = rec.get("canonical_smiles")
    val = rec.get("standard_value")
    if smi and val:
        try:
            rows.append({
                "smiles": smi,
                "measured log solubility in mols per litre": float(val),
                "source": "ChEMBL"
            })
        except (ValueError, TypeError):
            continue

df_logp = pd.DataFrame(rows).drop_duplicates("smiles")
df_logp.to_csv("data/raw/chembl_logp.csv", index=False)
print(f"  Saved {len(df_logp):,} LogP molecules from ChEMBL")

print("\n✅ ChEMBL data downloaded!")
print("Files added:", [f for f in os.listdir("data/raw") if "chembl" in f])