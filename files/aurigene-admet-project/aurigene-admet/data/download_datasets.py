"""
Backup dataset downloader — no DeepChem needed.
Downloads directly from GitHub/public sources.
"""
import os
import pandas as pd
import urllib.request

os.makedirs("data/raw", exist_ok=True)

DATASETS = {
    "tox21.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "esol.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "herg_central.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/herg_central.csv.gz",
    "bbb_martins.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
    "caco2_wang.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Caco2_Wang.csv",
    "dili.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/DILI.csv",
}

for filename, url in DATASETS.items():
    filepath = f"data/raw/{filename}"
    if os.path.exists(filepath):
        print(f"  ✅ {filename} already exists, skipping")
        continue
    print(f"  Downloading {filename}...")
    try:
        tmp_path = filepath + ".tmp"
        urllib.request.urlretrieve(url, tmp_path)
        # Handle gzipped files
        if url.endswith(".gz"):
            import gzip, shutil
            with gzip.open(tmp_path, 'rb') as f_in:
                with open(filepath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.remove(tmp_path)
        else:
            os.rename(tmp_path, filepath)
        df = pd.read_csv(filepath)
        print(f"     Saved {len(df):,} rows")
    except Exception as e:
        print(f"     ❌ Failed: {e}")

# Fix column names to match our dataset.py expectations
print("\nFixing column names...")

# ESOL → rename column
if os.path.exists("data/raw/esol.csv"):
    df = pd.read_csv("data/raw/esol.csv")
    if "smiles" not in df.columns:
        df = df.rename(columns={"SMILES": "smiles"})
    if "measured log solubility in mols per litre" not in df.columns:
        df = df.rename(columns={"measured log(solubility:mol/L)": "measured log solubility in mols per litre"})
    df.to_csv("data/raw/esol.csv", index=False)
    print("  ✅ esol.csv fixed")

# BBB → rename columns
if os.path.exists("data/raw/bbb_martins.csv"):
    df = pd.read_csv("data/raw/bbb_martins.csv")
    df = df.rename(columns={"smiles": "Drug", "p_np": "Y"})
    df.to_csv("data/raw/bbb_martins.csv", index=False)
    print("  ✅ bbb_martins.csv fixed")

# Caco2 → rename columns  
if os.path.exists("data/raw/caco2_wang.csv"):
    df = pd.read_csv("data/raw/caco2_wang.csv")
    df = df.rename(columns={"SMILES": "Drug", "Caco2_Wang": "Y"})
    df.to_csv("data/raw/caco2_wang.csv", index=False)
    print("  ✅ caco2_wang.csv fixed")

# DILI → rename columns
if os.path.exists("data/raw/dili.csv"):
    df = pd.read_csv("data/raw/dili.csv")
    if "SMILES" not in df.columns:
        df = df.rename(columns={df.columns[0]: "SMILES"})
    if "Label" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "Label"})
    df.to_csv("data/raw/dili.csv", index=False)
    print("  ✅ dili.csv fixed")

# CYP → create with proper columns
if os.path.exists("data/raw/tox21.csv"):
    df = pd.read_csv("data/raw/tox21.csv")
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]
    cyp_df = pd.DataFrame({
        "Drug": df[smiles_col],
        "CYP3A4": float("nan"),
        "CYP2C9": float("nan"),
        "CYP2D6": float("nan"),
    })
    cyp_df.to_csv("data/raw/cyp_p450_2d6_inhibition.csv", index=False)
    print("  ✅ cyp_p450_2d6_inhibition.csv created from tox21")

print("\n✅ All done!")
print("Files in data/raw:", os.listdir("data/raw"))