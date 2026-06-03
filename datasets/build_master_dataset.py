"""
datasets/build_master_dataset.py
Combines all Cheminformatics-ML notebook datasets into one master CSV.

Output: datasets/cheminformatics_master_dataset.csv
  - One row per molecule
  - Normalized common RDKit feature columns (snake_case)
  - Per-task target columns (NaN where not applicable)
  - Metadata: notebook_id, task, task_type, molecule_source

Notebooks covered:
  NB01 — Aqueous Solubility (ESOL)          regression
  NB02 — Boiling Point                       regression
  NB03 — LogP (lipophilicity)                regression
  NB04 — Ames Mutagenicity                   classification
  NB05 — BBB Permeability (MoleculeNet BBBP) classification
  NB06 — SpaceChem-AI (bandgap + efficiency) regression (multi-target)
"""
import os, warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")
HERE   = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.join(HERE, "..", "notebooks")
OUT    = os.path.join(HERE, "cheminformatics_master_dataset.csv")

# ── Column name normalisation map (original → master) ─────────────────────────
COL_MAP = {
    # Identifiers
    "compound":           "name",
    "Compound":           "name",
    # Core physicochemical
    "MW":                 "mw",          "mw": "mw",
    "LogP":               "logp",        "logp": "logp",
    "HBD":                "hbd",         "hbd": "hbd",
    "HBA":                "hba",         "hba": "hba",
    "TPSA":               "tpsa",        "tpsa": "tpsa",
    "RotBonds":           "rot_bonds",   "rotbonds": "rot_bonds",
    "AromaticRings":      "arom_rings",  "arom_rings": "arom_rings",
    "RingCount":          "ring_count",  "total_rings": "ring_count",
    "HeavyAtoms":         "heavy_atoms", "heavy_atoms": "heavy_atoms",
    "MolMR":              "mol_mr",      "mol_refractivity": "mol_mr",
    "FractionCSP3":       "frac_csp3",   "frac_csp3": "frac_csp3",
    "NumHeteroatoms":     "num_heteroatoms",
    "NumAliphaticRings":  "num_aliphatic_rings",
    # Extended descriptors (NB01–04)
    "BertzCT":            "bertz_ct",
    "Chi0":               "chi0",        "Chi1": "chi1",
    "Kappa1":             "kappa1",      "Kappa2": "kappa2",
    "LabuteASA":          "labute_asa",
    "MaxPartialCharge":   "max_partial_charge",
    "MinPartialCharge":   "min_partial_charge",
    "NumStereocenters":   "num_stereocenters",
    # NB04 structural alerts
    "NumNitrogens":       "num_nitrogens",
    "NumOxygens":         "num_oxygens",
    "NumHalogens":        "num_halogens",
    "NumSulfurs":         "num_sulfurs",
    "HasNitroGroup":      "has_nitro_group",
    "HasNitroso":         "has_nitroso",
    "HasAromaticAmine":   "has_aromatic_amine",
    "HasAlkylHalide":     "has_alkyl_halide",
    # NB06 space-specific
    "nhoh":               "nhoh",
    "no_count":           "no_count",
    "planarity_score":    "planarity_score",
    "pi_extent":          "pi_extent",
    "fp_density":         "fp_density",
    "fluoro_substitution":"fluoro_substitution",
    "space_uv_factor":    "space_uv_factor",
    # NB06 also uses logp/logp (already mapped)
    "LogP_rdkit":         "logp_rdkit",
    "SlogP_VSA3":         "slogp_vsa3",
    "PEOE_VSA1":          "peoe_vsa1",
}


def normalise(df):
    """Rename columns to master names, drop unmapped ones silently."""
    df = df.rename(columns=COL_MAP)
    # Lowercase any remaining column names
    df.columns = [c.lower() if c not in COL_MAP.values() else c for c in df.columns]
    return df


def load_nb(path, notebook_id, task, task_type, target_col, target_alias,
            name_col="name", extra_cols=None):
    """Load a processed CSV and tag it with metadata."""
    df = pd.read_csv(path)
    df = normalise(df)

    # Ensure name column exists
    if name_col not in df.columns and "name" not in df.columns:
        df["name"] = [f"{task}_{i}" for i in range(len(df))]

    df["notebook_id"]  = notebook_id
    df["task"]         = task
    df["task_type"]    = task_type
    df["molecule_source"] = f"NB{notebook_id:02d}_{task}"

    # Rename target column to standard alias
    if target_col in df.columns and target_alias != target_col:
        df[target_alias] = df[target_col]

    return df


# ── NB01: Solubility ──────────────────────────────────────────────────────────
print("Loading NB01: Solubility (ESOL)...")
nb01 = load_nb(
    os.path.join(NB_DIR, "01_solubility_esol/data/esol_processed.csv"),
    notebook_id=1, task="solubility", task_type="regression",
    target_col="logS", target_alias="target_logS",
)
nb01["target_logS"] = nb01["logs"] if "logs" in nb01.columns else nb01.get("target_logS")
print(f"  {len(nb01)} molecules")

# ── NB02: Boiling Point ───────────────────────────────────────────────────────
print("Loading NB02: Boiling Point...")
nb02 = load_nb(
    os.path.join(NB_DIR, "02_boiling_point/data/bp_processed.csv"),
    notebook_id=2, task="boiling_point", task_type="regression",
    target_col="bp_celsius", target_alias="target_bp_celsius",
)
print(f"  {len(nb02)} molecules")

# ── NB03: LogP ────────────────────────────────────────────────────────────────
print("Loading NB03: LogP...")
nb03 = load_nb(
    os.path.join(NB_DIR, "03_logp/data/logp_processed.csv"),
    notebook_id=3, task="logp", task_type="regression",
    target_col="logp_exp", target_alias="target_logP",
)
print(f"  {len(nb03)} molecules")

# ── NB04: Ames Mutagenicity ───────────────────────────────────────────────────
print("Loading NB04: Ames Mutagenicity...")
nb04 = load_nb(
    os.path.join(NB_DIR, "04_ames_mutagenicity/data/ames_processed.csv"),
    notebook_id=4, task="mutagenicity", task_type="classification",
    target_col="mutagenic", target_alias="target_mutagenic",
)
print(f"  {len(nb04)} molecules")

# ── NB05: BBB Permeability ────────────────────────────────────────────────────
print("Loading NB05: BBB Permeability (downloading from MoleculeNet)...")
try:
    import urllib.request
    BBBP_URL  = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
    bbbp_raw  = pd.read_csv(BBBP_URL)
    # Standardise columns
    col_remap = {}
    for c in bbbp_raw.columns:
        cl = c.lower().strip()
        if "smiles" in cl: col_remap[c] = "smiles"
        if cl in ("p_np","bbbp","label","y"): col_remap[c] = "target_bbb_permeable"
        if "name" in cl: col_remap[c] = "name"
    bbbp_raw = bbbp_raw.rename(columns=col_remap)
    bbbp_raw = bbbp_raw[bbbp_raw["smiles"].notna()].reset_index(drop=True)
    bbbp_raw["target_bbb_permeable"] = bbbp_raw["target_bbb_permeable"].astype(int)
    # Compute RDKit features
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem import rdFingerprintGenerator as rfg
        rows = []
        gen = rfg.GetMorganGenerator(radius=2, fpSize=512)
        for _, row in bbbp_raw.iterrows():
            mol = Chem.MolFromSmiles(str(row["smiles"]))
            if mol is None: continue
            fp = gen.GetFingerprint(mol)
            rows.append({
                "name": row.get("name",""),
                "smiles": row["smiles"],
                "target_bbb_permeable": row["target_bbb_permeable"],
                "mw": round(Descriptors.MolWt(mol), 3),
                "logp": round(Descriptors.MolLogP(mol), 4),
                "hbd": rdMolDescriptors.CalcNumHBD(mol),
                "hba": rdMolDescriptors.CalcNumHBA(mol),
                "tpsa": round(Descriptors.TPSA(mol), 3),
                "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
                "arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "heavy_atoms": Descriptors.HeavyAtomCount(mol),
                "mol_mr": round(Descriptors.MolMR(mol), 4),
                "frac_csp3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
                "num_heteroatoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
                "fp_density": round(fp.GetNumOnBits() / 512.0, 6),
                "notebook_id": 5,
                "task": "bbb_permeability",
                "task_type": "classification",
                "molecule_source": "NB05_bbb_permeability",
            })
        nb05 = pd.DataFrame(rows)
        print(f"  {len(nb05)} molecules (with RDKit features)")
    except ImportError:
        nb05 = bbbp_raw.copy()
        nb05["notebook_id"] = 5
        nb05["task"] = "bbb_permeability"
        nb05["task_type"] = "classification"
        nb05["molecule_source"] = "NB05_bbb_permeability"
        print(f"  {len(nb05)} molecules (RDKit unavailable — features not computed)")
except Exception as e:
    print(f"  ⚠ Could not load BBB data: {e} — skipping NB05")
    nb05 = pd.DataFrame()

# ── NB06: SpaceChem-AI ────────────────────────────────────────────────────────
print("Loading NB06: SpaceChem-AI...")
nb06_raw = pd.read_csv(
    os.path.join(NB_DIR, "06_space_solar_type2_kII/spacechem_dataset.csv")
)
nb06 = normalise(nb06_raw)
nb06 = nb06.rename(columns={
    "bandgap_ev": "target_bandgap_ev",
    "abs_efficiency": "target_abs_efficiency",
})
nb06["notebook_id"]     = 6
nb06["task"]            = "space_solar"
nb06["task_type"]       = "regression"
nb06["molecule_source"] = "NB06_spacechem_ai"
print(f"  {len(nb06)} molecules")


# ── Combine ───────────────────────────────────────────────────────────────────
print("\nCombining all datasets...")
all_dfs = [df for df in [nb01, nb02, nb03, nb04, nb05, nb06] if len(df) > 0]
master = pd.concat(all_dfs, ignore_index=True, sort=False)

# ── Reorder columns: metadata → identifiers → features → targets ──────────────
META = ["notebook_id", "task", "task_type", "molecule_source", "name", "smiles", "family"]
TARGETS = [c for c in master.columns if c.startswith("target_")]
FEATURES = sorted([c for c in master.columns
                   if c not in META + TARGETS
                   and c not in ("logs", "logp_exp", "bp_celsius", "mutagenic",
                                 "p_np", "num", "label")])
ORDERED = [c for c in META if c in master.columns] + \
          [c for c in FEATURES if c in master.columns] + \
          sorted(TARGETS)
master = master[[c for c in ORDERED if c in master.columns]]

# ── Save ──────────────────────────────────────────────────────────────────────
master.to_csv(OUT, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Master dataset: {OUT}")
print(f"  Total rows   : {len(master):,}")
print(f"  Total columns: {len(master.columns)}")
print(f"\n  Per notebook:")
for nb_id, grp in master.groupby("notebook_id"):
    task = grp["task"].iloc[0]
    ttype = grp["task_type"].iloc[0]
    targets = [c for c in TARGETS if grp[c].notna().any()]
    print(f"    NB{nb_id:02d} {task:<20} {ttype:<15} {len(grp):>5} rows  targets: {', '.join(targets)}")
print(f"\n  Feature columns ({len(FEATURES)}):")
for f in FEATURES:
    filled = master[f].notna().sum()
    print(f"    {f:<30} {filled:>5}/{len(master)} filled ({100*filled/len(master):.0f}%)")
print(f"{'='*60}")
