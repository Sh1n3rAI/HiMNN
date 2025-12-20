# train_cond_no_val.py
# Conductivity predictor – no validation split
# License: Apache-2.0

import os
import argparse
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import dgl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model_cond import HiMNN_Cond


# ------------------------------- reproducibility -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True


# ------------------------------- dataset helpers -------------------------------

def pad_to_len(xs: List[float], L: int, pad: float = 0.0) -> List[float]:
    xs = list(xs)
    return xs[:L] + [pad] * max(0, L - len(xs))


def build_edge_mats_from_molgraphs(mol_graph_list: List[List[dgl.DGLGraph]]) -> List[List[torch.Tensor]]:
    mats = []
    for formula in mol_graph_list:
        per_formula = []
        for mol in formula:
            n = mol.num_nodes()
            e_feat = mol.edata['e_feat']
            dense = torch.zeros(n, n, dtype=torch.float32)
            slot_per_row = n - 1
            for i in range(n):
                ncol = 0
                for j in range(slot_per_row):
                    slot = i * slot_per_row + j
                    if slot >= e_feat.size(0):
                        break
                    if e_feat[slot][4] != 0:
                        continue
                    oh = e_feat[slot][0:3]
                    idx = [k + 1 for k, v in enumerate(oh) if v == 1]
                    val = idx[0] if len(idx) else 0
                    if j >= i:
                        if ncol == 0:
                            ncol = j + 1
                        dense[i, ncol] = val
                        ncol += 1
                    else:
                        dense[i, j] = val
            dense = dense + dense.T
            dense = dense + torch.eye(n)
            per_formula.append(dense)
        mats.append(per_formula)
    return mats


class CondDataset(Dataset):
    def __init__(self, graphs, conc, labels, n_mols, text, edges, temps):
        self.graphs = graphs
        self.conc = conc
        self.labels = labels
        self.n_mols = n_mols
        self.text = text
        self.edges = edges
        self.temps = temps

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return (
            self.graphs[idx],
            torch.tensor(self.conc[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            torch.tensor(self.n_mols[idx], dtype=torch.long),
            torch.tensor(self.text[idx], dtype=torch.float32),
            [e.clone() for e in self.edges[idx]],
            torch.tensor(self.temps[idx], dtype=torch.float32),
        )


def collate_fn(batch):
    g = dgl.batch([b[0] for b in batch])
    conc = torch.stack([b[1] for b in batch])
    y = torch.stack([b[2] for b in batch])
    n_mol = torch.stack([b[3] for b in batch])
    text = torch.stack([b[4] for b in batch])
    edges = [b[5] for b in batch]
    temp = torch.stack([b[6] for b in batch])
    return g, conc, y, n_mol, text, edges, temp


# ------------------------------- train & eval -------------------------------

def forward_batch(model: HiMNN_Cond, g, conc, n_mol, text, edges, temp, device):
    g = g.to(device)
    node_feats = g.ndata['n_feat'].to(device)
    return model(
        node_feats=node_feats,
        proportions=conc.to(device),
        num_mols=n_mol.to(device),
        text_emb=text.to(device),
        edge_mats=edges,
        temperature=temp.to(device),
    )


def train_one_epoch(model, loader, optimizer, loss_fn, lambda_kl, device):
    model.train()
    losses, kls = [], []
    ys, preds = [], []
    for g, conc, y, n_mol, text, edges, temp in loader:
        pred, kl_ab, kl_ba, _ = forward_batch(model, g, conc, n_mol, text, edges, temp, device)
        y = y.to(device).unsqueeze(1)
        mse = loss_fn(pred, y).mean()
        loss = mse + lambda_kl * (kl_ab + kl_ba)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(mse.detach().item())
        ys.append(y.detach().cpu().view(-1))
        preds.append(pred.detach().cpu().view(-1))
        kls.append((kl_ab.detach().item(), kl_ba.detach().item()))

    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(preds).numpy()
    r2 = r2_score(y_all, p_all) if len(y_all) > 1 else float("nan")
    kl_mean = (float(np.mean([k[0] for k in kls])) if kls else 0.0,
               float(np.mean([k[1] for k in kls])) if kls else 0.0)
    return float(np.mean(losses)), r2, kl_mean


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    losses, ys, preds = [], [], []
    for g, conc, y, n_mol, text, edges, temp in loader:
        pred, _, _, _ = forward_batch(model, g, conc, n_mol, text, edges, temp, device)
        y = y.to(device).unsqueeze(1)
        mse = loss_fn(pred, y).mean()
        losses.append(mse.item())
        ys.append(y.detach().cpu().view(-1))
        preds.append(pred.detach().cpu().view(-1))
    y_all = torch.cat(ys).numpy()
    p_all = torch.cat(preds).numpy()
    r2 = r2_score(y_all, p_all) if len(y_all) > 1 else float("nan")
    return float(np.mean(losses)), r2, p_all, y_all


# ------------------------------- loading -------------------------------

def load_dataset(args):
    mol_graph = torch.load(os.path.join(args.data_dir, args.mol_graph_pt))
    if args.limit is not None:
        mol_graph = mol_graph[:args.limit]

    graphs = [dgl.batch(mols) for mols in mol_graph]
    n_mols = [len(mols) for mols in mol_graph]

    if args.edge_mats_pt:
        edge_mats = torch.load(os.path.join(args.data_dir, args.edge_mats_pt))
        edge_mats = edge_mats[:len(mol_graph)] if args.limit else edge_mats
    else:
        edge_mats = build_edge_mats_from_molgraphs(mol_graph)

    labels = torch.load(os.path.join(args.data_dir, args.label_pt))
    conc = torch.load(os.path.join(args.data_dir, args.conc_pt))
    text = torch.load(os.path.join(args.data_dir, args.text_pt))
    temps = torch.load(os.path.join(args.data_dir, args.temp_pt))

    if args.limit is not None:
        labels = labels[:args.limit]
        conc = conc[:args.limit]
        temps = temps[:args.limit]

    conc = [pad_to_len(c, args.max_len, 0.0) for c in conc]

    if isinstance(text, torch.Tensor):
        if text.dim() == 3 and text.size(0) != len(conc):
            text = text.transpose(0, 1)
        text = text[:len(conc)]
    else:
        raise ValueError("text embeddings must be a 3D tensor")

    dataset = CondDataset(graphs, conc, labels, n_mols, text, edge_mats, temps)
    return dataset


# ------------------------------- args & main -------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Conductivity predictor – no validation split")
    p.add_argument("--data_dir", type=str, default="./data_Preprocess")
    p.add_argument("--mol_graph_pt", type=str, default="Cond_Graph_List.pt")
    p.add_argument("--edge_mats_pt", type=str, default="Cond_edge_List.pt")
    p.add_argument("--label_pt", type=str, default="Cond_All_List.pt")
    p.add_argument("--conc_pt", type=str, default="Cond_All_conc_List.pt")
    p.add_argument("--text_pt", type=str, default="Cond_Text_Embeding_Tensor.pt")
    p.add_argument("--temp_pt", type=str, default="Cond_Temperature.pt")

    # p.add_argument("--limit", type=int, default=1946)
    p.add_argument("--max_len", type=int, default=6)
    p.add_argument("--batch_size", type=int, default=15)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--lambda_kl", type=float, default=0.08)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=111)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_pred_dir", type=str, default="./outputs_cond")
    p.add_argument("--save_ckpt", type=str, default="./outputs_cond/best_model.pth")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.save_pred_dir, exist_ok=True)

    dataset = load_dataset(args)

    # 仅划分训练集和测试集
    idx_train, idx_test = train_test_split(
        list(range(len(dataset))),
        test_size=args.test_ratio,
        random_state=args.seed
    )

    def subset(ds, indices):
        return torch.utils.data.Subset(ds, indices)

    train_loader = DataLoader(
        subset(dataset, idx_train),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        subset(dataset, idx_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=True
    )

    model = HiMNN_Cond(atom_in_dim=32, use_temperature=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss(reduction="none")

    for epoch in range(1, args.epochs + 1):
        tr_mse, tr_r2, (kl_ab, kl_ba) = train_one_epoch(
            model, train_loader, optimizer, loss_fn, args.lambda_kl, device
        )
        if epoch % 50 == 0 or epoch == 1:
            print(f"[Epoch {epoch:04d}] Train MSE={tr_mse:.6f}  R2={tr_r2:.4f}  "
                  f"KL_ab={kl_ab:.6f} KL_ba={kl_ba:.6f}")


    te_mse, te_r2, pred, y = evaluate(model, test_loader, loss_fn, device)
    rmse = float(np.sqrt(te_mse))
    print(f"[Final] Test RMSE={rmse:.6f}  MSE={te_mse:.6f}  R2={te_r2:.6f}")


    if args.save_pred_dir:
        os.makedirs(args.save_pred_dir, exist_ok=True)
        torch.save(torch.tensor(y),   os.path.join(args.save_pred_dir, f"labels_seed{args.seed}.pt"))
        torch.save(torch.tensor(pred), os.path.join(args.save_pred_dir, f"preds_seed{args.seed}.pt"))


if __name__ == "__main__":
    main()