#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess Conductivity tables → .pt artifacts.
- Reads Excel with columns:
    S1 SMILES .. S4 SMILES, Salt1 SMILES, Salt2 SMILES
    S1 Mol % .. S4 Mol %,  Salt1 Mol %,  Salt2 Mol %
    Temperature (oC), Conductivity (S/cm)
- Builds per-molecule DGL graphs with node/edge features (32-dim node feats, matching your model)
- Generates text embeddings for each SMILES with a HF model (default: ./Bert)
- Saves (names aligned with your training script defaults):
    Liquid_Unique_Electropy_DGL.pt
    Liquid_Unique_Electropy_Comp_List.pt
    Liquid_Unique_Electropy_labels_List.pt
    Liquid_Unique_Electropy_Text_Embeding_Tensor.pt
    Liquid_Unique_Electropy_Temperature.pt
  (text tensor shape: (L, N, 384) to remain compatible)
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import dgl
from rdkit import Chem
import networkx as nx
from transformers import AutoTokenizer, BertModel, BertConfig

# -------------------------- constants (kept from your original) --------------------------
# Hybridization index mapping for 32-dim node feature (your conductivity code)
HYB_INDEX_COND = {
    'DSP3': 2, 'D2SP3': 3, 'SP3D3': 4, 'SP3D2': 5, 'SP2': 6, 'SP': 7,
    'SP3D': 8, 'DSP2': 9, 'SP3': 10, 'S': 0, 'P': 1
}

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

def get_bond_type_idx(bond) -> int:
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE: return 0
    if bt == Chem.rdchem.BondType.DOUBLE: return 1
    if bt == Chem.rdchem.BondType.TRIPLE: return 2
    if bt == Chem.rdchem.BondType.AROMATIC: return 3
    return 0

def build_element_index(smiles_set):
    counts = {}
    for smi in smiles_set:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            counts[sym] = counts.get(sym, 0) + 1
    sorted_syms = sorted(counts.keys(), key=lambda s: ATOMIC_Z.get(s, 999))
    return {sym: i for i, sym in enumerate(sorted_syms)}

def atom_feature_32(atom, sym2idx, hyb_map=HYB_INDEX_COND):
    """
    32-dim scheme from your conductivity code:
      [0:V) element one-hot
      [V]   atomic number
      [V+1] aromatic
      [V+2] donor-like flag
      [V+3] acceptor-like flag
      [V+4 ...] hybridization one-hot (offset by V+3)
      [31]  total Hs
    """
    V = len(sym2idx)
    feat = [0.0] * 32
    sym = atom.GetSymbol()
    if sym in sym2idx:
        idx = sym2idx[sym]
        if idx < 32:
            feat[idx] = 1.0

    if V + 1 < 32: feat[V]   = float(ATOMIC_Z.get(sym, 0))
    if V + 2 < 32: feat[V+1] = 1.0 if atom.GetIsAromatic() else 0.0
    if V + 3 < 32:
        donor = (atom.GetDegree() > 1 and atom.GetHybridization() != Chem.rdchem.HybridizationType.SP)
        feat[V+2] = 1.0 if donor else 0.0
    if V + 4 < 32:
        acceptor = (atom.GetFormalCharge() < 0 or atom.GetTotalDegree() > 2)
        feat[V+3] = 1.0 if acceptor else 0.0

    hyb = str(atom.GetHybridization())
    hyb_idx = hyb_map.get(hyb, 0)
    slot = V + 3 + hyb_idx
    if slot < 32:
        feat[slot] = 1.0

    feat[31] = float(atom.GetTotalNumHs() or 0)
    return feat

def smiles_to_dgl(smiles: str, sym2idx):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    n_feats = [atom_feature_32(a, sym2idx) for a in mol.GetAtoms()]
    n_feat = torch.tensor(n_feats, dtype=torch.float32)

    dist_mat = Chem.GetDistanceMatrix(mol)
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--excel", type=str, required=True, help="Input Liquid.xlsx (conductivity)")
    ap.add_argument("--save_dir", type=str, default="./data_Preprocess", help="Output directory")
    ap.add_argument("--bert_path", type=str, default="./Bert", help="HF model path or name")
    ap.add_argument("--max_comp", type=int, default=6, help="Max components per formula")
    args = ap.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # 1) read table
    df = pd.read_excel(args.excel)

    smi_cols = ["S1 SMILES", "S2 SMILES", "S3 SMILES", "S4 SMILES", "Salt1 SMILES", "Salt2 SMILES"]
    mol_cols = ["S1 Mol %", "S2 Mol %", "S3 Mol %", "S4 Mol %", "Salt1 Mol %", "Salt2 Mol %"]
    y_col   = "Conductivity (S/cm)"
    t_col   = "Temperature (oC)"

    # 2) build list of formulas: each is [{SMILES: conc}, ..., {'Temperature': T}]
    formulas = []
    conc_list, labels, temps = [], [], []

    for _, row in df.iterrows():
        comp = []
        comp_row = []
        for s_col, c_col in zip(smi_cols, mol_cols):
            s = row.get(s_col, np.nan)
            c = row.get(c_col, np.nan)
            if pd.isna(s) or pd.isna(c):
                continue
            s = str(s)
            c = float(c)
            comp.append({s: c})
            comp_row.append(c)

        if len(comp) == 0:
            continue

        T  = float(row.get(t_col, np.nan)) if not pd.isna(row.get(t_col, np.nan)) else 25.0
        y  = float(row.get(y_col, np.nan)) if not pd.isna(row.get(y_col, np.nan)) else np.nan
        if np.isnan(y):  # skip if label missing
            continue

        comp.append({"Temperature": T})
        formulas.append(comp)
        conc_list.append(comp_row)
        labels.append(y)
        temps.append(T)

    # 3) global element index from SMILES set
    smiles_set = set()
    for comp in formulas:
        for d in comp:
            for k in d.keys():
                if k != "Temperature":
                    smiles_set.add(k)
    sym2idx = build_element_index(smiles_set)

    # 4) DGL graphs per formula
    dgl_list = []
    for comp in formulas:
        graphs = []
        for d in comp:
            for k, v in d.items():
                if k == "Temperature":
                    continue
                g = smiles_to_dgl(k, sym2idx)
                if g is not None:
                    graphs.append(g)
        dgl_list.append(graphs)

    # 5) text embeddings (padded to max_comp) → (L, N, 384)
    tok = AutoTokenizer.from_pretrained(args.bert_path)
    cfg = BertConfig.from_pretrained(args.bert_path)
    mdl = BertModel.from_pretrained(args.bert_path, config=cfg)
    mdl.eval()

    text_embeds = []
    for comp in formulas:
        vecs = []
        cnt = 0
        for d in comp:
            for k, v in d.items():
                if k == "Temperature":
                    continue
                inputs = tok(str(k), return_tensors="pt", padding=True, truncation=True, max_length=512)
                with torch.no_grad():
                    out = mdl(**inputs).last_hidden_state[:, 0, :]  # [CLS]
                vecs.append(out.squeeze(0).cpu().numpy())
                cnt += 1
        for _ in range(args.max_comp - cnt):
            vecs.append(np.zeros((384,), dtype=np.float32))
        text_embeds.append(vecs)
    text_tensor = torch.tensor(text_embeds, dtype=torch.float32).permute(1, 0, 2)  # (L,N,384)

    # 6) save with names aligned to your cleaned training script
    torch.save(dgl_list, os.path.join(args.save_dir, "Cond_Graph_List.pt"))
    torch.save(conc_list,     os.path.join(args.save_dir, "Cond_All_conc_List.pt"))
    torch.save(labels,         os.path.join(args.save_dir, "Cond_All_List.pt"))
    torch.save(text_tensor,    os.path.join(args.save_dir, "Cond_Text_Embeding_Tensor.pt"))
    torch.save(temps,          os.path.join(args.save_dir, "Cond_Temperature.pt"))
    print("[Conductivity] Saved: Liquid_Unique_Electropy_*.pt files")

if __name__ == "__main__":
    main()
