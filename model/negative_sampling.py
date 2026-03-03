import numpy as np
import torch
import dgl
from typing import List, Tuple, Dict

def negative_sampling(g: dgl.DGLGraph, num_neg: int, existing_edges: Dict[str, List[int]],
                      sub_g: dgl.DGLGraph, dynamic_ratio: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    neg_src, neg_dst = [], []
    existing_pairs = set(zip(existing_edges['source_idx'], existing_edges['target_idx']))
    sub_nodes = sub_g.nodes().tolist()
    actual_num_neg = int(num_neg * dynamic_ratio)
    if actual_num_neg < 1:
        actual_num_neg = 1

    for _ in range(actual_num_neg):
        src = np.random.choice(sub_nodes)
        dst = np.random.choice(sub_nodes)
        while (src, dst) in existing_pairs or src == dst:
            src = np.random.choice(sub_nodes)
            dst = np.random.choice(sub_nodes)
        neg_src.append(src)
        neg_dst.append(dst)

    return (torch.tensor(neg_src, device=g.device, dtype=torch.long),
            torch.tensor(neg_dst, device=g.device, dtype=torch.long))

def get_dynamic_neg_ratio_by_graph_size(sub_g: dgl.DGLGraph) -> float:
    num_nodes = sub_g.num_nodes()
    small_graph_threshold = 1000
    large_graph_threshold = 3000

    if num_nodes < small_graph_threshold:
        return 0.2
    elif num_nodes > large_graph_threshold:
        return 1.0
    else:
        ratio = 0.2 + 0.8 * ((num_nodes - small_graph_threshold) /
                             (large_graph_threshold - small_graph_threshold))
        return ratio