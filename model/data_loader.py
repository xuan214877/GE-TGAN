import dgl
import json
import os
import torch
from typing import Tuple, List, Dict
from config import device
import numpy as np


def load_pruned_data(pruned_dir: str, embedding_path: str) -> Tuple[dgl.DGLGraph, List[int], Dict[str, torch.Tensor]]:
    graphs, _ = dgl.load_graphs(os.path.join(pruned_dir, 'pruned_graph1.bin'))
    g = graphs[0].to(device)
    print(f"nodes={g.num_nodes()}, edges={g.num_edges()}")

    if 'edge_year' not in g.edata:
        raise ValueError("missing 'edge_year'")

    edges_path = os.path.join(pruned_dir, 'pruned_edges1.json')
    edge_metadata = []
    if os.path.exists(edges_path):
        with open(edges_path, 'r', encoding='utf-8') as f:
            edge_metadata = json.load(f)
        print(f"{len(edge_metadata)} edges")
        if len(edge_metadata) > 0 and 'edge_year' not in edge_metadata[0]:
            print("missing edge_year")
    else:
        print("missing files")

    nodes_path = os.path.join(pruned_dir, 'pruned_nodes1.json')
    pruned_nodes = []
    if os.path.exists(nodes_path):
        with open(nodes_path, 'r', encoding='utf-8') as f:
            pruned_nodes = json.load(f)
        print(f"{len(pruned_nodes)} nodes")

        if len(pruned_nodes) > 0 and 'initial_feature' in pruned_nodes[0]:
            initial_features = [n['initial_feature'] for n in pruned_nodes]
            g.ndata['feat'] = torch.tensor(initial_features, dtype=torch.float32, device=device)
            print(f"nodes dimension: {g.ndata['feat'].shape}")
        else:
            print("warning: missing initial_feature")
    else:
        print("warning: missing pruned file")

    if 'feat' not in g.ndata:
        text_embeddings = {}
        if embedding_path and os.path.exists(embedding_path):
            try:
                with open(embedding_path, 'r', encoding='utf-8') as f:
                    embedding_data = json.load(f)
                for node in embedding_data.get('nodes', []):
                    keyword = node.get('keyword')
                    text_feat = node.get('initial_feature', np.random.rand(770).tolist())
                    text_embeddings[keyword] = text_feat
            except Exception as e:
                print(f"{str(e)}, use random feature")

        features = []
        for node_data in pruned_nodes:
            keyword = node_data['keyword']
            features.append(text_embeddings.get(keyword, np.random.rand(770).tolist()))
        g.ndata['feat'] = torch.tensor(features, dtype=torch.float32, device=device)
        
    total_degrees = []
    total_frequencies = []
    for node in pruned_nodes:
        degree = sum(node.get('degree_per_year', {}).values())
        frequency = sum(node.get('frequency_per_year', {}).values())
        total_degrees.append(degree)
        total_frequencies.append(frequency)

    metadata = {
        'edge_metadata': edge_metadata,
        'pruned_nodes': pruned_nodes,
        'total_degrees': total_degrees,
        'total_frequencies': total_frequencies
    }

    return g, edge_metadata, metadata