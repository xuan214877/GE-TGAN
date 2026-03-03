import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Dict
import dgl
import dgl.function as fn
from config import TIME_EMB_DIM

class TemporalEncoding(nn.Module):
    def __init__(self, dim: int = TIME_EMB_DIM):
        super().__init__()
        self.dim = dim
        self.omega = nn.Parameter(torch.randn(dim // 2))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.squeeze(-1) if t.dim() > 1 else t
        angles = torch.outer(t, self.omega.to(t.device))
        sin_val = torch.sin(angles)
        cos_val = torch.cos(angles)

        pe = torch.zeros(t.size(0), self.dim, device=t.device)
        pe[:, 0::2] = sin_val
        pe[:, 1::2] = cos_val

        return pe * math.sqrt(1.0 / self.dim)

class TGATHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        time_emb_dim: int = TIME_EMB_DIM,
        dropout: float = 0.5
    ):
        super().__init__()

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, out_dim, bias=False)

        self.time_enc = TemporalEncoding(time_emb_dim)
        if time_emb_dim != out_dim:
            self.time_proj = nn.Linear(time_emb_dim, out_dim, bias=False)
        else:
            self.time_proj = nn.Identity()

        self.attn_fc = nn.Linear(3 * out_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = dropout

    def forward(self, g, node_feats, time_diff):
        h = node_feats
        src, dst = g.edges()

        h_src = self.W(h[src])
        h_dst = self.W(h[dst])
        v_src = self.Wv(h[src])

        t_emb = self.time_proj(self.time_enc(time_diff))
        edge_feat = torch.cat([h_src, h_dst, t_emb], dim=1)

        e_ij = self.attn_fc(edge_feat).squeeze(-1)
        e_ij = self.leaky_relu(e_ij)

        alpha = dgl.ops.edge_softmax(g, e_ij)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        g.edata['m'] = v_src * alpha.unsqueeze(-1)
        g.update_all(fn.copy_e('m', 'm'), fn.sum('m', 'h_agg'))

        return g.ndata['h_agg']



class TGATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, time_emb_dim: int = TIME_EMB_DIM, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

        self.heads = nn.ModuleList([
            TGATHead(in_dim, self.head_dim, time_emb_dim)
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, g, node_feats, time_feats):
        head_outputs = []
        for head in self.heads:
            head_out = head(g, node_feats, time_feats)
            head_outputs.append(head_out)

        combined = torch.cat(head_outputs, dim=1)
        combined = F.dropout(combined, p=self.dropout, training=self.training)

        out = self.proj(combined)
        out = F.relu(self.norm(out))
        return out

class TGATModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layer1 = TGATLayer(hidden_dim, hidden_dim, num_heads=num_heads, dropout=dropout)
        self.layer2 = TGATLayer(hidden_dim, out_dim, num_heads=num_heads, dropout=dropout)
        self.dropout = dropout

    def forward(self, g, node_feats, time_feats):
        h = F.relu(self.input_proj(node_feats))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.layer1(g, h, time_feats)
        h = self.layer2(g, h, time_feats)
        return h

class LinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, u_emb, v_emb):
        return torch.sum(u_emb * v_emb, dim=1) / math.sqrt(u_emb.size(1))