#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess LCE (Coulombic Efficiency) tables → .pt artifacts.
- Reads Excel with columns: smiles1..6, conc1..6, LCE
- De-duplicates entries that share the *same composition* but *different LCE* (remove both)
- Builds per-molecule DGL graphs with node/edge features
- Generates text embeddings for each SMILES with a HF model (default: ./Bert)
- Saves:
    dgl_LCE_Graph_List.pt               : List[List[DGLGraph]]
    LCE_All_conc_List.pt                : List[List[float]]
    LCE_All_List.pt                     : List[float]  (labels)
    LCE_Text_Embeding_Tensor.pt         : Tensor (L, N, 384)  # note: (L,N,384) to match your original
"""

import argparse
import os
import copy
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
import networkx as nx
from transformers import AutoTokenizer, BertModel, BertConfig

# -------------------------- constants (kept from your original) --------------------------
EDGE_NUMS = {'SINGLE': 1., 'DOUBLE': 2., 'TRIPLE': 3., 'AROMATIC': 1.5}
# Hybridization index mapping for 25-dim node feature (as in your LCE code)
HYB_INDEX_LCE = {'DSP3': 2, 'D2SP3': 3, 'SP3D3': 4, 'SP3D2': 5, 'SP2': 6, 'SP': 7, 'SP3D': 8, 'DSP2': 9, 'SP3': 9, 'S': 0, 'P': 1}

# periodic table: element → Z
ATOMIC_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27,
    'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45,
    'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,
    'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63,
    'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72,
    'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81,
    'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99,
    'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107,
    'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115,
    'Lv': 116, 'Ts': 117, 'Og': 118
}

# -------------------------- helpers --------------------------

def get_bond_type_idx(bond) -> int:
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: return 0
    if bt == Chem.rdchem.BondType.DOUBLE: return 1
    if bt == Chem.rdchem.BondType.TRIPLE: return 2
    if bt == Chem.rdchem.BondType.AROMATIC: return 3
    return 0

def build_element_index(smiles_set):
    """Create {element_symbol: idx} sorted by atomic number."""
    counts = {}
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
    # sort by Z
    sorted_syms = sorted(counts.keys(), key=lambda s: ATOMIC_Z.get(s, 999))
    return {sym: i for i, sym in enumerate(sorted_syms)}

def atom_feature_25(atom, sym2idx, hyb_map=HYB_INDEX_LCE):
    """
    Reproduce your 25-dim scheme:
      [0:V) element one-hot
      [V]   atomic number
      [V+1] aromatic
      [V+2] donor-like flag
      [V+3] acceptor-like flag
      [V+4 ...] hybridization one-hot (index offset by V+3)
      [24]  total Hs (last slot)
    """
    V = len(sym2idx)
    feat = [0.0] * 25
    sym = atom.GetSymbol()
    if sym in sym2idx:
        idx = sym2idx[sym]
        if idx < 25:  # prevent OOB just in case
            feat[idx] = 1.0

    if V + 1 < 25: feat[V]   = float(ATOMIC_Z.get(sym, 0))
    if V + 2 < 25: feat[V+1] = 1.0 if atom.GetIsAromatic() else 0.0
    # coarse donor/acceptor heuristics to mimic original
    if V + 3 < 25:
        donor = (atom.GetDegree() > 1 and atom.GetHybridization() != Chem.rdchem.HybridizationType.SP)
        feat[V+2] = 1.0 if donor else 0.0
    if V + 4 < 25:
        acceptor = (atom.GetFormalCharge() < 0 or atom.GetTotalDegree() > 2)
        feat[V+3] = 1.0 if acceptor else 0.0

    hyb = str(atom.GetHybridization())
    hyb_idx = hyb_map.get(hyb, 0)
    slot = V + 3 + hyb_idx
    if slot < 25:
        feat[slot] = 1.0

    feat[24] = float(atom.GetTotalNumHs() or 0)
    return feat

def smiles_to_dgl(smiles: str, sym2idx):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # node features
    n_feats = []
    for atom in mol.GetAtoms():
        n_feats.append(atom_feature_25(atom, sym2idx))
    n_feat = torch.tensor(n_feats, dtype=torch.float32)

    # all-directed edges (i → j, i!=j), with bond-type one-hot or "non-bond" slot, plus topological distance
    dist_mat = Chem.GetDistanceMatrix(mol)  # topological distances
    e_feat_rows, d_rows, src, dst = [], [], [], []
    N = mol.GetNumAtoms()
    bonded = { (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): get_bond_type_idx(b) for b in mol.GetBonds() }
    bonded.update({ (j,i): t for (i,j), t in list(bonded.items()) })

    for i in range(N):
        for j in range(N):
            if i == j: continue
            row = [0]*5
            if (i,j) in bonded:
                row[bonded[(i,j)]] = 1
            else:
                row[4] = 1
            e_feat_rows.append(row)
            d_rows.append([float(dist_mat[i][j])])
            src.append(i); dst.append(j)

    g = dgl.graph((torch.tensor(src), torch.tensor(dst)))
    g.ndata['n_feat'] = n_feat
    g.edata['e_feat'] = torch.tensor(e_feat_rows, dtype=torch.float32)
    g.edata['distance'] = torch.tensor(d_rows, dtype=torch.float32)
    return g

def dedup_conflict_remove_both(records):
    """
    For records with same composition dict (SMILES→conc) but different LCE, remove all conflicts.
    records: List[{'comp': {smi: conc, ...}, 'lce': float}]
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for r in records:
        key = tuple(sorted(r['comp'].items()))
        groups[key].append(r['lce'])
    to_drop = set()
    for key, vals in groups.items():
        if len(set(vals)) > 1:
            to_drop.add(key)
    kept = [r for r in records if tuple(sorted(r['comp'].items())) not in to_drop]
    return kept

# -------------------------- main pipeline --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, default="./data_Preprocess/LCE.xlsx", help="Input LCE .xlsx")
    ap.add_argument("--bert_path", type=str, default="./Bert", help="HF model path or name")
    ap.add_argument("--save_dir", type=str, default="./data_Preprocess", help="Output directory")
    ap.add_argument("--max_comp", type=int, default=6, help="Max components per formula")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1) read table
    df = pd.read_excel(args.excel)
    smiles_cols = [f"smiles{i}" for i in range(1, args.max_comp+1)]
    conc_cols   = [f"conc{i}"   for i in range(1, args.max_comp+1)]
    y_col = "LCE"

    # 2) build records
    records = []
    for i in range(len(df)):
        comp = {}
        for sc, cc in zip(smiles_cols, conc_cols):
            s = df.loc[i, sc]
            c = df.loc[i, cc]
            if pd.isna(s) or (pd.notna(c) and float(c) == 0.0):
                continue
            if pd.notna(s) and pd.notna(c):
                comp[str(s)] = float(c)
        y = float(df.loc[i, y_col])
        if len(comp) == 0:
            continue
        records.append({"comp": comp, "lce": y})

    # 3) remove conflicts (same comp, different LCE → drop all)
    records = dedup_conflict_remove_both(records)

    # 4) global SMILES set → element index
    all_smiles = set()
    for r in records:
        all_smiles.update(list(r["comp"].keys()))
    sym2idx = build_element_index(all_smiles)

    # 5) DGL graphs (+ concentration list) & labels
    dgl_list, conc_list, labels = [], [], []
    for r in records:
        graphs = []
        for smi, conc in r["comp"].items():
            g = smiles_to_dgl(smi, sym2idx)
            if g is not None:
                graphs.append(g)
        if not graphs:
            continue
        dgl_list.append(graphs)
        conc_list.append([float(r["comp"][s]) for s in r["comp"].keys()])
        labels.append(float(r["lce"]))

    # 6) text embeddings (padded to max_comp) → tensor (L, N, 384) to match your original files
    tok = AutoTokenizer.from_pretrained(args.bert_path)
    cfg = BertConfig.from_pretrained(args.bert_path)
    mdl = BertModel.from_pretrained(args.bert_path, config=cfg)
    mdl.eval()

    text_embeds = []  # List[List[384]]
    for r in records:
        vecs = []
        for smi in r["comp"].keys():
            inputs = tok(str(smi), return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                out = mdl(**inputs).last_hidden_state[:, 0, :]  # [CLS]
            vecs.append(out.squeeze(0).cpu().numpy())
        # pad to max_comp with zeros
        for _ in range(args.max_comp - len(vecs)):
            vecs.append(np.zeros((384,), dtype=np.float32))
        text_embeds.append(vecs)

    text_tensor = torch.tensor(text_embeds, dtype=torch.float32).permute(1, 0, 2)  # (L,N,384)

    # 7) save
    torch.save(dgl_list, os.path.join(args.save_dir, "LCE_Graph_List.pt"))
    torch.save(conc_list, os.path.join(args.save_dir, "LCE_All_conc_List.pt"))
    torch.save(labels,   os.path.join(args.save_dir, "LCE_All_List.pt"))
    torch.save(text_tensor, os.path.join(args.save_dir, "LCE_Text_Embeding_Tensor.pt"))
    print("[LCE] Saved: LCE_Graph_List.pt, LCE_All_conc_List.pt, LCE_All_List.pt, LCE_Text_Embeding_Tensor.pt")

if __name__ == "__main__":
    main()
