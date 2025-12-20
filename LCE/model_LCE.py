# models.py
# Copyright (c) 2025.
# Licensed under the Apache 2.0 License.

from typing import Tuple
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, GATv2Conv


# ------------------------------- utils: relational KL -------------------------------

def _pairwise_sim(z: torch.Tensor, sim: str = "cosine") -> torch.Tensor:
    """Compute pairwise similarity matrix (B, B) from embeddings z (B, C)."""
    if sim == "cosine":
        z = F.normalize(z, dim=1, eps=1e-8)
        return z @ z.T
    elif sim == "dot":
        return z @ z.T
    else:
        raise ValueError(f"Unknown sim='{sim}', choose 'cosine' or 'dot'.")


def _row_softmax(sim_mat: torch.Tensor, tau: float = 0.07, exclude_self: bool = True) -> torch.Tensor:
    """Row-wise softmax with optional self-mask and numeric stability."""
    b = sim_mat.size(0)
    logits = sim_mat / max(tau, 1e-8)
    if exclude_self and b > 1:
        mask = torch.eye(b, dtype=torch.bool, device=sim_mat.device)
        logits = logits.masked_fill(mask, float("-inf"))
    p = torch.softmax(logits, dim=1)
    if not torch.isfinite(p).all():
        p = torch.nan_to_num(p, nan=1.0 / b, posinf=1.0 / b, neginf=0.0)
        p = p / p.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return p


def _kl_rowwise(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-wise KL(P||Q) then average over rows."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=1).mean()


def symmetric_relational_kl(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    tau: float = 0.07,
    sim: str = "cosine",
    exclude_self: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """KL(P_a||P_b) + KL(P_b||P_a) between row-softmaxed pairwise-sim distributions."""
    sa = _pairwise_sim(z_a, sim=sim)
    sb = _pairwise_sim(z_b, sim=sim)
    pa = _row_softmax(sa, tau=tau, exclude_self=exclude_self)
    pb = _row_softmax(sb, tau=tau, exclude_self=exclude_self)
    kl_ab = _kl_rowwise(pa, pb)
    kl_ba = _kl_rowwise(pb, pa)
    return kl_ab, kl_ba, (kl_ab + kl_ba)


# ------------------------------- tiny MLP blocks for fusion -------------------------------

class FFN1D(nn.Module):
    """Simple 1D FFN with residual."""
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FeatLevelFusion1D(nn.Module):
    """Joint-key + (MHSA + cross-attn) feature-level fusion for two vectors g/t."""
    def __init__(self, embed_dim: int = 96, num_heads: int = 8, attn_drop: float = 0.1):
        super().__init__()
        c = embed_dim
        self.joint_key = nn.Sequential(
            nn.Linear(2 * c, c // 4),
            nn.GELU(),
            nn.Linear(c // 4, c // 4),
            nn.GELU(),
            nn.Linear(c // 4, c),
        )
        self.ln_g = nn.LayerNorm(c)
        self.ln_t = nn.LayerNorm(c)

        self.attn_g  = nn.MultiheadAttention(c, num_heads, dropout=attn_drop, batch_first=True)  # MHSA for g
        self.attn_t  = nn.MultiheadAttention(c, num_heads, dropout=attn_drop, batch_first=True)  # MHSA for t
        self.attn_gx = nn.MultiheadAttention(c, num_heads, dropout=attn_drop, batch_first=True)  # cross: g<-key
        self.attn_tx = nn.MultiheadAttention(c, num_heads, dropout=attn_drop, batch_first=True)  # cross: t<-key

        self.ln_out_g = nn.LayerNorm(c)
        self.ln_out_t = nn.LayerNorm(c)

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # g, t: (B, C)
        key = self.joint_key(torch.cat([g, t], dim=-1))  # (B, C)
        g_seq = self.ln_g(g).unsqueeze(1)  # (B, 1, C)
        t_seq = self.ln_t(t).unsqueeze(1)  # (B, 1, C)
        k_seq = key.unsqueeze(1)           # (B, 1, C)

        g_mhsa, _ = self.attn_g(g_seq, g_seq, g_seq)      # (B,1,C)
        t_mhsa, _ = self.attn_t(t_seq, t_seq, t_seq)
        g_mhca, _ = self.attn_gx(g_seq, k_seq, g_seq)     # (B,1,C)
        t_mhca, _ = self.attn_tx(t_seq, k_seq, t_seq)

        g_out = self.ln_out_g((g_mhsa + g_mhca).squeeze(1))  # (B,C)
        t_out = self.ln_out_t((t_mhsa + t_mhca).squeeze(1))  # (B,C)
        return g_out, t_out


class FeatFuseBlock1D(nn.Module):
    """Feature-level fusion followed by FFN on the merged signal."""
    def __init__(self, embed_dim: int = 96, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.1):
        super().__init__()
        self.fuse_op = FeatLevelFusion1D(embed_dim, num_heads, attn_drop)
        self.ln = nn.LayerNorm(embed_dim)
        self.ffn = FFN1D(embed_dim, mlp_ratio, drop)

    def forward(self, g: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        g, t = self.fuse_op(g, t)      # (B,C)
        x = g + t                      # (B,C)
        x = self.ffn(self.ln(x))       # (B,C)
        return x, g, t                 # also return per-modality for KL


class DecisionFuse1D(nn.Module):
    """Decision-level fusion for 1D regression via learned convex gate over (a+b)."""
    def __init__(self, in_mode: str = "sum"):
        super().__init__()
        if in_mode == "sum":
            self.gate = nn.Linear(1, 1)
            self.use_concat = False
        else:
            self.gate = nn.Linear(2, 1)
            self.use_concat = True

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # a,b: (B,1)
        gate_in = torch.cat([a, b], dim=-1) if self.use_concat else (a + b)
        w = torch.sigmoid(self.gate(gate_in))
        return w * (a + b)


class MultiModalRegressor1D(nn.Module):
    """
    Two-stage fusion for 1D regression:
      1) Feature-level (joint-key + MHSA + cross-attn)
      2) Decision-level convex gating
    """
    def __init__(self, graph_dim: int, text_dim: int, embed_dim: int = 96, num_heads: int = 8,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.1):
        super().__init__()
        c = embed_dim
        self.g_proj = nn.Sequential(nn.Linear(graph_dim, c), nn.LayerNorm(c), nn.GELU())
        self.t_proj = nn.Sequential(nn.Linear(text_dim,  c), nn.LayerNorm(c), nn.GELU())

        self.feat_fuse = FeatFuseBlock1D(c, num_heads, mlp_ratio, drop, attn_drop)

        self.head_g = nn.Linear(c, 1)
        self.head_t = nn.Linear(c, 1)
        self.head_x = nn.Linear(c, 1)

        self.dec_fuse_gt = DecisionFuse1D(in_mode="sum")
        self.dec_fuse_fx = DecisionFuse1D(in_mode="sum")

    def forward(self, graph_vec: torch.Tensor, text_vec: torch.Tensor):
        # graph_vec, text_vec: (B, D)
        g = self.g_proj(graph_vec)
        t = self.t_proj(text_vec)
        x, g, t = self.feat_fuse(g, t)

        g_hat = self.head_g(g)     # (B,1)
        t_hat = self.head_t(t)     # (B,1)
        x_hat = self.head_x(x)     # (B,1)

        pred_fu     = self.dec_fuse_gt(g_hat, t_hat)       # (B,1)
        pred_logits = self.dec_fuse_fx(pred_fu, x_hat)     # (B,1)
        return pred_logits, pred_fu, x_hat, g_hat, t_hat, g, t


# ------------------------------- GNN plumbing -------------------------------------

def get_gnn_layer(name: str, in_channels: int, out_channels: int, heads: int):
    """Factory for popular GNN layers (PyG)."""
    if name == 'gcn':
        return GCNConv(in_channels, out_channels)
    elif name == 'gat':
        return GATConv(-1, out_channels, heads)
    elif name == 'sage':
        return SAGEConv(in_channels, out_channels)
    elif name == 'gin':
        return GINConv(nn.Linear(in_channels, out_channels), train_eps=True)
    elif name == 'gat2':
        return GATv2Conv(-1, out_channels, heads)
    raise ValueError(name)


# ------------------------------- Main combined model -------------------------------------

class HiMNN_LCE(nn.Module):
    """
    Graph-text multimodal regressor.

    Per-sample pipeline:
      (1) Per-molecule virtual-node GNN → molecule embeddings (64d)
      (2) Compatibility-weighted molecule graph (structure + text + concentration) → formula embedding (64d)
      (3) Text side: per-molecule text [384] + concentration appended → Transformer pooling → 64d
      (4) Two-stage fusion + symmetric relational KL on fused latent g/t
    """
    def __init__(
        self,
        atom_in_dim: int = 30,
        text_in_dim: int = 384,
        gnn_name: str = "gcn",
        embed_out_dim: int = 64,
        trans_hidden: int = 64,
        trans_heads: int = 8,
        trans_layers: int = 6,
        fusion_embed: int = 96,
        fusion_heads: int = 8,
        fusion_drop: float = 0.0,
        fusion_attn_drop: float = 0.1,
        # RBF
    ):
        super().__init__()

        # (A) Atom-level GNN to get per-molecule embedding via a virtual node
        self.convs = nn.ModuleList()
        for i in range(2):
            in_ch  = atom_in_dim if i == 0 else 32
            out_ch = embed_out_dim if i == 1 else 32
            heads  = 1 if (i == 1 or 'gat' not in gnn_name) else 8
            self.convs.append(get_gnn_layer(gnn_name, in_ch, out_ch, heads))

        # (B) Molecule-graph aggregation (uses compatibility + concentrations)
        self.convs_mol = nn.ModuleList()
        for i in range(2):
            in_ch  = embed_out_dim if i == 0 else 128
            out_ch = embed_out_dim if i == 1 else 128
            heads  = 1 if (i == 1 or 'gat' not in gnn_name) else 8
            self.convs_mol.append(get_gnn_layer(gnn_name, in_ch, out_ch, heads))

        # (C) Text pooling on [text|conc] with a lightweight Transformer
        self.pool_attn = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=text_in_dim + 1, nhead=max(1, trans_heads // 1), dropout=0.1, batch_first=True
            ),
            num_layers=trans_layers,
        )
        self.pool_mlp1 = nn.Linear(text_in_dim + 1, 128)
        self.pool_mlp2 = nn.Linear(128, trans_hidden)  # -> 64

        # (D) Small projection for compatibility
        self.emb_linear = nn.Linear(text_in_dim, embed_out_dim)  # 384 -> 64

        # (E) Two-stage fusion head
        self.mm_regressor = MultiModalRegressor1D(
            graph_dim=embed_out_dim,
            text_dim=trans_hidden,
            embed_dim=fusion_embed,
            num_heads=fusion_heads,
            mlp_ratio=4.0,
            drop=fusion_drop,
            attn_drop=fusion_attn_drop,
        )

        # Learnable compatibility parameters
        self.Wg = nn.Parameter(torch.randn(embed_out_dim, embed_out_dim))
        self.Wt = nn.Parameter(torch.randn(embed_out_dim, embed_out_dim))
        self.lambda_param = nn.Parameter(torch.tensor(0.5))  # λ ∈ (0,1)
        self.sigmoid = nn.Sigmoid()

        # ---------- RBF kernel params----------
        # mix weight between graph-RBF and text-RBF
        self.lambda_gt_param = nn.Parameter(torch.tensor(0.5))  # sigmoid -> (0,1)

        # RBF bandwidths (fixed buffers; you can tune these as hyperparams)
        self.register_buffer("rbf_gamma_g", torch.tensor(1.0))
        self.register_buffer("rbf_gamma_t", torch.tensor(1.0))

    @torch.no_grad()
    def _dense_to_edge_index(self, dense: torch.Tensor) -> torch.Tensor:
        """Convert dense adjacency (N,N) into COO edge_index (2,E) on same device."""
        return torch.nonzero(dense, as_tuple=False).T

    def _pairwise_sqdist(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pairwise squared Euclidean distance.
        x: (m, c) -> dist2: (m, m)
        """
        x2 = (x * x).sum(dim=1, keepdim=True)  # (m,1)
        dist2 = x2 + x2.T - 2.0 * (x @ x.T)  # (m,m)
        return dist2.clamp_min(0.0)

    def _molecule_embed_from_atoms(self, node_feats: torch.Tensor, dense_adj: torch.Tensor, offset: int) -> Tuple[torch.Tensor, int]:
        """
        Build a virtual-node molecule graph and run a 2-layer GNN.
        - node_feats: (sum_atoms, atom_in_dim) over the batch
        - dense_adj:  (A, A) for this molecule (values >0 mean an edge)
        - offset:     current cursor in node_feats
        Return:
            mol_vec: (embed_out_dim,), new_offset
        """
        dev = node_feats.device
        length = dense_adj.size(0)  # #atoms in this molecule

        # Append virtual node fully connected to atoms (self-loop excluded)
        ones_row = torch.ones(1, length, device=dev)
        dense_v = torch.cat([dense_adj.to(dev), ones_row], dim=0)
        dense_v = torch.cat([dense_v, torch.ones(length + 1, 1, device=dev)], dim=1)
        dense_v[length, length] = 0  # no self-loop on virtual

        edge_index = self._dense_to_edge_index(dense_v)  # (2, E)

        # Gather atom features and append a zero virtual node
        atom = node_feats[offset: offset + length]
        atom = torch.cat([atom, torch.zeros(1, atom.size(1), device=dev)], dim=0)  # (A+1, Din)

        x = atom
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  # PyG GCNConv supports edge_index

        mol_vec = x[length]  # virtual node output
        return mol_vec, offset + length

    def forward(
        self,
        node_feats: torch.Tensor,         # (sum_atoms, atom_in_dim)
        proportions: torch.Tensor,        # (B, L), concentrations with padding zeros
        num_atoms: torch.Tensor,          # (B,), number of molecules per formula
        emb: torch.Tensor,                # (B, L, 384) text embeddings (padded)
        edge_list: list,                  # list of list[tensor(A,A)] per formula
    ):
        """
        Return:
            pred        : (B,1)
            kl_ab,kl_ba : scalars
            kl_sum      : scalar
        """
        dev = node_feats.device
        bsz = proportions.size(0)

        # (1) Per-molecule representation
        mol_vecs = []
        cursor = 0
        for i in range(bsz):
            for dense_adj in edge_list[i]:
                v, cursor = self._molecule_embed_from_atoms(node_feats, dense_adj, cursor)
                mol_vecs.append(v.unsqueeze(0))
        mol_vecs = torch.cat(mol_vecs, dim=0)  # (sum_mols, 64)

        # (2) Build text pool with a final "concentration" column; drop padded rows (all-zero)
        emb2d = emb.reshape(-1, emb.size(-1))                      # (B*L, 384)
        nonzero_mask = (emb2d != 0).any(dim=1)
        emb_valid = emb2d[nonzero_mask]                            # (sum_mols, 384)
        pool = torch.cat([emb_valid.to(dev), torch.zeros(emb_valid.size(0), 1, device=dev)], dim=1)  # (sum_mols, 385)

        # Fill concentrations into the last column for each molecule in each formula
        pos = 0
        for i in range(bsz):
            m = int(num_atoms[i].item())
            for k in range(m):
                pool[pos + k, -1] = proportions[i, k]
            pos += m

        # (3) Per-formula text pooling → 64d
        pools = []
        pos = 0
        for i in range(bsz):
            m = int(num_atoms[i].item())
            seq = pool[pos:pos + m].unsqueeze(0)                # (1, m, 385)
            pos += m
            seq = self.pool_attn(seq).squeeze(0)                # (m, 385)
            seq = self.pool_mlp1(seq)                           # (m, 128)
            seq = self.pool_mlp2(seq)                           # (m, 64)
            pools.append(seq.sum(dim=0, keepdim=True))          # (1, 64)
        text_pool = torch.cat(pools, dim=0)                     # (B, 64)

        # (4) Compatibility-weighted molecule graph per formula → 64d
        formula_vecs = []
        pos = 0
        for i in range(bsz):
            m = int(num_atoms[i].item())
            g_feat = mol_vecs[pos:pos + m]                      # (m, 64)
            t_feat = self.emb_linear(emb[i, :m].to(dev))        # (m, 64)
            conc   = proportions[i, :m]                         # (m,)

            # compatibility scores
            # g_score = (g_feat @ self.Wg) @ g_feat.T             # (m, m)
            # t_score = (t_feat @ self.Wt) @ t_feat.T             # (m, m)
            # compat  = torch.sigmoid(g_score + t_score)          # (m, m)

            # ---------- RBF-kernel interaction prior ----------
            # project first (optional but keeps learnable alignment like your original Wg/Wt)
            g_z = g_feat @ self.Wg  # (m,64)
            t_z = t_feat @ self.Wt  # (m,64)

            dg2 = self._pairwise_sqdist(g_z)  # (m,m)
            dt2 = self._pairwise_sqdist(t_z)  # (m,m)

            kg = torch.exp(-dg2 / self.rbf_gamma_g.clamp_min(1e-8))  # (m,m)
            kt = torch.exp(-dt2 / self.rbf_gamma_t.clamp_min(1e-8))  # (m,m)

            w = torch.sigmoid(self.lambda_gt_param)  # scalar in (0,1)
            compat = w * kg + (1.0 - w) * kt  # (m,m)
            # ---------- RBF-kernel interaction prior ----------

            conc_mat = torch.outer(conc, conc)                  # (m, m)
            lam = self.sigmoid(self.lambda_param)               # scalar in (0,1)

            # degree-based uniform baseline
            degree = 1.0 / max(1, m - 1)
            uniform = degree * torch.ones(m, m, device=dev)
            uniform += (1 - degree) * torch.eye(m, device=dev)

            # convex combination + virtual formula node carrying concentrations
            dense = lam * uniform + (1 - lam) * (compat * conc_mat)   # (m, m)
            dense = torch.cat([dense, torch.ones(1, m, device=dev)], dim=0)
            for j in range(m):
                dense[m, j] = conc[j]
            dense = torch.cat([dense, torch.zeros(m + 1, 1, device=dev)], dim=1)
            for j in range(m):
                dense[j, m] = conc[j]

            adj = SparseTensor.from_dense(dense)  # PyG supports SparseTensor
            x = torch.cat([g_feat, torch.zeros(1, g_feat.size(1), device=dev)], dim=0)  # (m+1, 64)

            for conv in self.convs_mol:
                x = F.relu(conv(x, adj))

            formula_vecs.append(x[m].unsqueeze(0))              # (1, 64)
            pos += m

        graph_vec = torch.cat(formula_vecs, dim=0)              # (B, 64)

        # (5) Two-stage fusion and relational KL
        pred, pred_fu, x_hat, g_hat, t_hat, g_lat, t_lat = self.mm_regressor(graph_vec, text_pool)
        kl_ab, kl_ba, kl_sum = symmetric_relational_kl(g_lat, t_lat, tau=0.07, sim="cosine", exclude_self=True)
        return pred, kl_ab, kl_ba, kl_sum
