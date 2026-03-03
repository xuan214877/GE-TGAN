import dgl
import json
import torch
import os
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from config import device


def generate_embeddings(
    g: dgl.DGLGraph,
    model: nn.Module,
    save_path: str,
    metadata: Dict,
    num_nodes: int,
    stats: Dict,
    final_year: int = 2025
):
    
    full_g = dgl.add_reverse_edges(g, copy_edata=True)
    full_node_feats = full_g.ndata['feat']

    node_degrees = torch.tensor(
        metadata['total_degrees'],
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    node_frequencies = torch.tensor(
        metadata['total_frequencies'],
        dtype=torch.float32,
        device=device
    ).unsqueeze(1)

    node_degrees = (node_degrees - stats['degree_mean']) / stats['degree_std']
    node_frequencies = (node_frequencies - stats['freq_mean']) / stats['freq_std']

    full_node_feats = torch.cat(
        [full_node_feats, node_degrees, node_frequencies],
        dim=1
    )

    edge_years = full_g.edata['edge_year'].float()

    full_time_feats = final_year - edge_years
    full_time_feats = torch.clamp(full_time_feats, min=0)

    if full_g.num_edges() != full_time_feats.size(0):
        full_time_feats = full_time_feats.repeat(2)

    model.eval()
    with torch.no_grad():
        try:
            final_emb = model(
                full_g,
                full_node_feats,
                full_time_feats
            ).cpu().numpy()
        except RuntimeError:
            device_cpu = torch.device("cpu")
            model.to(device_cpu)
            full_g = full_g.to(device_cpu)
            full_node_feats = full_node_feats.to(device_cpu)
            full_time_feats = full_time_feats.to(device_cpu)
            final_emb = model(
                full_g,
                full_node_feats,
                full_time_feats
            ).cpu().numpy()
            model.to(device)

    embedding_results = [
        {
            "keyword": metadata['pruned_nodes'][i]['keyword'],
            "tgat_embedding": final_emb[i].tolist()
        }
        for i in range(num_nodes)
        if i < len(metadata['pruned_nodes'])
    ]

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_results, f, indent=2)

