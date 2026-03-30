"""
featurizer.py — Molecular Featurization
=========================================
Aurigene Pharmaceutical Services | AIDD Group

Converts SMILES strings into:
  1. PyTorch Geometric Data objects (atom nodes + bond edges with features)
  2. Morgan fingerprints (2048-bit, radius=2) as auxiliary input

Atom features (74-dim total):
  - Atomic number one-hot (44)
  - Degree one-hot (0–10)
  - Formal charge
  - Num Hs one-hot (0–4)
  - Hybridization one-hot (SP, SP2, SP3, SP3D, SP3D2)
  - Aromaticity flag
  - Ring membership flag
  - LogP contribution (Crippen)
  - TPSA contribution

Bond features (12-dim total):
  - Bond type one-hot (single, double, triple, aromatic)
  - Is in ring
  - Is conjugated
  - Stereo one-hot (NONE, ANY, E, Z, CIS, TRANS)

Usage:
    from featurizer import MolecularFeaturizer
    feat = MolecularFeaturizer()
    graph = feat.smiles_to_graph("CC(=O)Nc1ccc(O)cc1")   # Paracetamol
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Crippen, rdPartialCharges
from rdkit.Chem import AllChem
from rdkit import RDLogger

# Suppress RDKit warnings for cleaner output
RDLogger.DisableLog("rdApp.*")


# ─────────────────────────────────────────────────────────────────────────────
# Atom feature vocabulary
# ─────────────────────────────────────────────────────────────────────────────

ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg",
    "Na", "Ca", "Fe", "As", "Al", "I", "B", "V", "K", "Tl",
    "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H",
    "Li", "Ge", "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr",
    "Pt", "Hg", "Pb", "Unknown",
]  # 44 entries

DEGREE_LIST        = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]   # 11 entries
NUM_HS_LIST        = [0, 1, 2, 3, 4]                         # 5 entries
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]  # 5 entries

BOND_TYPE_LIST = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]  # 4 entries

BOND_STEREO_LIST = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]  # 6 entries

# Sanity-check total dims
# Atom: 44 + 11 + 1 + 5 + 5 + 1 + 1 + 1 + 1 + 1 = 71... let me recount
# 44(symbol) + 11(degree) + 1(charge) + 5(numH) + 5(hybrid) + 1(arom) + 1(ring) + 1(logP) + 1(tpsa) + 1(partial_charge) = 71
# Bond: 4(type) + 1(ring) + 1(conjugated) + 6(stereo) = 12 ✓


def _one_hot(value, choices: list) -> List[int]:
    """One-hot encode `value` against `choices`; last entry is 'other'."""
    enc = [0] * len(choices)
    if value in choices:
        enc[choices.index(value)] = 1
    else:
        enc[-1] = 1
    return enc


def atom_features(atom: Chem.rdchem.Atom, mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    74-dimensional atom feature vector.
    Note: we compute logP/TPSA contributions at mol level, indexed by atom idx.
    """
    idx = atom.GetIdx()

    # Crippen contributions (logP, MR) per atom
    crippen_contribs = Crippen.rdMolDescriptors.CalcCrippenDescriptors
    try:
        lp_contribs = Crippen._GetAtomContribs(mol)  # list of (logP, MR) per atom
        logp_contrib = lp_contribs[idx][0]
    except Exception:
        logp_contrib = 0.0

    # TPSA contributions per atom
    try:
        tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)
        tpsa_contrib = tpsa_contribs[idx]
    except Exception:
        tpsa_contrib = 0.0

    # Gasteiger partial charge
    try:
        partial_charge = float(atom.GetDoubleProp("_GasteigerCharge"))
        if np.isnan(partial_charge) or np.isinf(partial_charge):
            partial_charge = 0.0
    except Exception:
        partial_charge = 0.0

    feats = (
        _one_hot(atom.GetSymbol(), ATOM_LIST)                   # 44
        + _one_hot(atom.GetDegree(), DEGREE_LIST)               # 11
        + [float(atom.GetFormalCharge())]                       #  1
        + _one_hot(atom.GetTotalNumHs(), NUM_HS_LIST)           #  5
        + _one_hot(atom.GetHybridization(), HYBRIDIZATION_LIST) #  5
        + [float(atom.GetIsAromatic())]                         #  1
        + [float(atom.IsInRing())]                              #  1
        + [logp_contrib]                                        #  1
        + [tpsa_contrib]                                        #  1
        + [partial_charge]                                      #  1
    )
    return np.array(feats, dtype=np.float32)  # 71 dims


def bond_features(bond: Chem.rdchem.Bond) -> np.ndarray:
    """12-dimensional bond feature vector."""
    feats = (
        _one_hot(bond.GetBondType(), BOND_TYPE_LIST)    # 4
        + [float(bond.IsInRing())]                      # 1
        + [float(bond.GetIsConjugated())]               # 1
        + _one_hot(bond.GetStereo(), BOND_STEREO_LIST)  # 6
    )
    return np.array(feats, dtype=np.float32)  # 12 dims


# ─────────────────────────────────────────────────────────────────────────────
# Main featurizer class
# ─────────────────────────────────────────────────────────────────────────────

class MolecularFeaturizer:
    """
    Converts SMILES → PyTorch Geometric Data with Morgan fingerprint.

    Args:
        fp_radius: Morgan fingerprint radius (default: 2 = ECFP4)
        fp_bits:   Number of fingerprint bits (default: 2048)
        add_hs:    Whether to add explicit hydrogens (default: False)
    """

    def __init__(
        self,
        fp_radius: int = 2,
        fp_bits:   int = 2048,
        add_hs:    bool = False,
    ):
        self.fp_radius = fp_radius
        self.fp_bits   = fp_bits
        self.add_hs    = add_hs

    def smiles_to_mol(self, smiles: str) -> Optional[Chem.rdchem.Mol]:
        """Parse, sanitize, and optionally add Hs. Returns None on failure."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
            if self.add_hs:
                mol = Chem.AddHs(mol)
            # Compute Gasteiger charges (needed for atom features)
            AllChem.ComputeGasteigerCharges(mol)
            return mol
        except Exception:
            return None

    def mol_to_fp(self, mol: Chem.rdchem.Mol) -> np.ndarray:
        """Morgan fingerprint as numpy binary array."""
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, self.fp_radius, nBits=self.fp_bits
        )
        return np.array(fp, dtype=np.float32)

    def mol_to_graph(self, mol: Chem.rdchem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            node_feats: (n_atoms, 71)
            edge_index: (2, 2*n_bonds)  — undirected, both directions
            edge_feats: (2*n_bonds, 12)
        """
        n_atoms = mol.GetNumAtoms()

        # Node features
        node_feats = np.array(
            [atom_features(mol.GetAtomWithIdx(i), mol) for i in range(n_atoms)],
            dtype=np.float32,
        )

        # Edge features (bonds appear twice for undirected graph)
        edge_src, edge_dst, edge_feats = [], [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bf    = bond_features(bond)
            edge_src += [i, j]
            edge_dst += [j, i]
            edge_feats += [bf, bf]

        if len(edge_src) == 0:
            # Single-atom molecule (rare); add self-loop
            edge_src = [0]
            edge_dst = [0]
            edge_feats = [np.zeros(12, dtype=np.float32)]

        edge_index = np.array([edge_src, edge_dst], dtype=np.int64)
        edge_feats = np.array(edge_feats, dtype=np.float32)

        return node_feats, edge_index, edge_feats

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Full pipeline: SMILES string → PyG Data object with fp attribute.
        Returns None if SMILES is invalid.
        """
        mol = self.smiles_to_mol(smiles)
        if mol is None:
            return None

        node_feats, edge_index, edge_feats = self.mol_to_graph(mol)
        fp = self.mol_to_fp(mol)

        return Data(
            x          = torch.tensor(node_feats, dtype=torch.float),
            edge_index = torch.tensor(edge_index, dtype=torch.long),
            edge_attr  = torch.tensor(edge_feats, dtype=torch.float),
            fp         = torch.tensor(fp, dtype=torch.float).unsqueeze(0),  # (1, fp_bits)
            smiles     = smiles,
        )

    def batch_featurize(
        self,
        smiles_list: List[str],
        verbose: bool = True,
    ) -> Tuple[List[Data], List[str]]:
        """
        Featurize a list of SMILES. Returns (valid_graphs, failed_smiles).
        """
        graphs, failed = [], []
        for smi in smiles_list:
            g = self.smiles_to_graph(smi)
            if g is not None:
                graphs.append(g)
            else:
                failed.append(smi)
        if verbose:
            print(f"  Featurized: {len(graphs)}/{len(smiles_list)} molecules "
                  f"({len(failed)} failed)")
        return graphs, failed


# ─────────────────────────────────────────────────────────────────────────────
# Applicability Domain: Tanimoto similarity
# ─────────────────────────────────────────────────────────────────────────────

class ApplicabilityDomain:
    """
    Tanimoto-based applicability domain checker.
    Uses Morgan fingerprints of the training set.

    Usage:
        ad = ApplicabilityDomain(threshold=0.4)
        ad.fit(train_smiles)
        flags = ad.check(query_smiles)  # True = in domain
    """

    def __init__(self, threshold: float = 0.4, k: int = 5):
        self.threshold = threshold
        self.k         = k
        self._train_fps = None
        self._feat      = MolecularFeaturizer()

    def fit(self, smiles_list: List[str]):
        fps = []
        for smi in smiles_list:
            mol = self._feat.smiles_to_mol(smi)
            if mol is not None:
                fps.append(self._feat.mol_to_fp(mol))
        self._train_fps = np.stack(fps)  # (N_train, 2048)

    def _tanimoto(self, a: np.ndarray, b: np.ndarray) -> float:
        """Tanimoto similarity between two binary fingerprint vectors."""
        inter = np.dot(a, b)
        union = np.sum(a) + np.sum(b) - inter
        return float(inter / union) if union > 0 else 0.0

    def check(self, smiles_list: List[str]) -> List[dict]:
        """
        Returns a list of dicts with keys:
            smiles, in_domain (bool), max_tanimoto (float)
        """
        assert self._train_fps is not None, "Call .fit() first."
        results = []
        for smi in smiles_list:
            mol = self._feat.smiles_to_mol(smi)
            if mol is None:
                results.append({"smiles": smi, "in_domain": False, "max_tanimoto": 0.0})
                continue
            fp = self._feat.mol_to_fp(mol)
            sims = [self._tanimoto(fp, train_fp) for train_fp in self._train_fps]
            top_k_mean = float(np.mean(sorted(sims, reverse=True)[:self.k]))
            results.append({
                "smiles": smi,
                "in_domain": top_k_mean >= self.threshold,
                "max_tanimoto": top_k_mean,
            })
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_smiles = [
        "CC(=O)Nc1ccc(O)cc1",          # Paracetamol
        "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene (aromatic)
        "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
        "INVALID_SMILES_XYZ",           # Should fail gracefully
    ]

    feat = MolecularFeaturizer()
    for smi in test_smiles:
        g = feat.smiles_to_graph(smi)
        if g:
            print(f"✅ {smi[:30]:30s}  "
                  f"atoms={g.x.shape[0]:3d}  "
                  f"bonds={g.edge_index.shape[1]//2:3d}  "
                  f"x={tuple(g.x.shape)}  "
                  f"fp={tuple(g.fp.shape)}")
        else:
            print(f"❌ Failed: {smi}")
