import dgl
import torch
import json
import os
import itertools
import networkx as nx
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict
import torch.nn as nn
from sklearn.neighbors import LocalOutlierFactor

USE_CUDA = torch.cuda.is_available()
DEBUG_MODE = False
device = torch.device("cuda" if USE_CUDA and not DEBUG_MODE else "cpu")


def load_pruned_network(save_dir: str) -> Tuple[List[Dict], List[Dict], dgl.DGLGraph, Dict[str, int]]:
    nodes_path = os.path.join(save_dir, 'pruned_nodes1.json')
    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Not exists: {nodes_path}")
    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    keyword_to_orig_idx = {node['keyword']: idx for idx, node in enumerate(nodes)}
    edges_path = os.path.join(save_dir, 'pruned_edges1.json')
    if not os.path.exists(edges_path):
        raise FileNotFoundError(f"Not exists: {edges_path}")
    with open(edges_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)
    graph_path = os.path.join(save_dir, 'pruned_graph1.bin')
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Not exists: {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    graph = graphs[0]
    return nodes, edges, graph, keyword_to_orig_idx


def dgl_to_nx(graph: dgl.DGLGraph, nodes: List[Dict]) -> nx.Graph:
    node_attrs = []
    if 'feat' in graph.ndata:
        node_attrs.append('feat')
    edge_attrs = []
    if 'earliest_year' in graph.edata:
        edge_attrs.append('earliest_year')
    nx_graph = dgl.to_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs).to_undirected()
    for idx, node in enumerate(nodes):
        if idx in nx_graph.nodes():
            nx_graph.nodes[idx]['keyword'] = node['keyword']
            nx_graph.nodes[idx]['original_index'] = idx
            nx_graph.nodes[idx]['co_occurrence_years'] = node.get('co_occurrence_years', [])
    return nx_graph


def get_subgraph(nx_graph: nx.Graph, nodes: set, edges: list) -> nx.Graph:
    valid_edges = [e for e in edges if e[0] in nodes and e[1] in nodes]
    subgraph = nx.Graph()
    subgraph.add_nodes_from(nodes)
    subgraph.add_edges_from(valid_edges)
    return subgraph


def find_path3_subgraphs(nx_graph: nx.Graph) -> Tuple[List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    for u in tqdm(nx_graph.nodes()):
        for v in nx_graph.neighbors(u):
            if u == v:
                continue
            for w in nx_graph.neighbors(v):
                if w != u and w not in (u, v):
                    subgraphs.append(([u, v, w], [0, 1, 0]))
    unique_subgraphs = []
    seen = set()
    for path, orbit in subgraphs:
        sorted_path = tuple(sorted(path))
        if sorted_path not in seen:
            seen.add(sorted_path)
            unique_subgraphs.append((path, orbit))
    subgraph_nodes = set(itertools.chain.from_iterable([p for p, _ in unique_subgraphs]))
    subgraph_edges = set(frozenset([p[i], p[i + 1]]) for p, _ in unique_subgraphs for i in range(2))
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return unique_subgraphs, subgraph_nodes, subgraph_edges


def find_triangle_subgraphs(nx_graph: nx.Graph) -> Tuple[List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    triangles = set()
    for node in tqdm(nx_graph.nodes()):
        neighbors = list(nx_graph.neighbors(node))
        for i, u in enumerate(neighbors):
            for v in neighbors[i + 1:]:
                if nx_graph.has_edge(u, v):
                    triangle = tuple(sorted([node, u, v]))
                    if triangle not in triangles:
                        triangles.add(triangle)
                        subgraphs.append((triangle, [0, 0, 0]))
    subgraph_nodes = set(itertools.chain.from_iterable([c for c, _ in subgraphs]))
    subgraph_edges = set(frozenset([u, v]) for c, _ in subgraphs for u, v in itertools.combinations(c, 2))
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return subgraphs, subgraph_nodes, subgraph_edges


def find_linear_4node_subgraphs(nx_graph: nx.Graph, retained_nodes: set, neighbor_index: Dict) -> Tuple[
    List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    seen_hashes = set()
    valid_nodes = list(retained_nodes)
    valid_nodes.sort(key=lambda x: nx_graph.degree(x), reverse=True)
    for u in tqdm(valid_nodes):
        valid_v = [v for v in neighbor_index[u] if v in retained_nodes and len(neighbor_index[v]) >= 2]
        for v in valid_v:
            valid_w = [w for w in neighbor_index[v] if w in retained_nodes and w != u and len(neighbor_index[w]) >= 2]
            for w in valid_w:
                valid_x = [x for x in neighbor_index[w] if x in retained_nodes and x not in (u, v)]
                for x in valid_x:
                    path = [u, v, w, x]
                    sorted_tuple = tuple(sorted(path))
                    path_hash = hash(sorted_tuple)
                    if path_hash not in seen_hashes:
                        seen_hashes.add(path_hash)
                        subgraphs.append((path, [0, 1, 1, 0]))
    subgraph_nodes = set(itertools.chain.from_iterable([p for p, _ in subgraphs]))
    subgraph_edges = set(frozenset([path[i], path[i + 1]]) for p, _ in subgraphs for i in range(3))
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return subgraphs, subgraph_nodes, subgraph_edges


def find_star4_subgraphs(nx_graph: nx.Graph) -> Tuple[List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    MAX_NEIGHBORS = 50
    for center in tqdm(nx_graph.nodes()):
        neighbors = list(nx_graph.neighbors(center))
        if len(neighbors) > MAX_NEIGHBORS or len(neighbors) < 3:
            continue
        for leaves in itertools.combinations(neighbors, 3):
            has_edge = False
            for u, v in itertools.combinations(leaves, 2):
                if v in nx_graph[u]:
                    has_edge = True
                    break
            if not has_edge:
                nodes = [center] + list(leaves)
                subgraphs.append((nodes, [0] + [1] * 3))
    subgraph_nodes = set(itertools.chain.from_iterable([n for n, _ in subgraphs]))
    subgraph_edges = set(frozenset([nodes[0], leaf]) for nodes, _ in subgraphs for leaf in nodes[1:])
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return subgraphs, subgraph_nodes, subgraph_edges


def find_g6_subgraphs(nx_graph: nx.Graph) -> Tuple[List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    seen_hashes = set()
    BATCH_SIZE = 500
    all_nodes = list(nx_graph.nodes())
    num_batches = (len(all_nodes) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH_SIZE
        end = min((batch_idx + 1) * BATCH_SIZE, len(all_nodes))
        batch_nodes = all_nodes[start:end]
        for node in tqdm(batch_nodes):
            neighbors = list(nx_graph.neighbors(node))
            if len(neighbors) < 2:
                continue
            for i, u in enumerate(neighbors):
                for v in neighbors[i + 1:]:
                    if nx_graph.has_edge(u, v):
                        triangle = (node, u, v)
                        for anchor in triangle:
                            for hang in nx_graph.neighbors(anchor):
                                if hang not in triangle:
                                    g6_nodes = [anchor, u, v, hang] if anchor != u and anchor != v else \
                                        [anchor, node, v, hang] if anchor == u else \
                                            [anchor, node, u, hang]
                                    sorted_tuple = tuple(sorted(g6_nodes))
                                    g6_hash = hash(sorted_tuple)
                                    if g6_hash not in seen_hashes:
                                        seen_hashes.add(g6_hash)
                                        subgraphs.append((g6_nodes, [0, 1, 1, 2]))
    subgraph_nodes = set(itertools.chain.from_iterable([p for p, _ in subgraphs]))
    subgraph_edges = set()
    for nodes, _ in subgraphs:
        subgraph_edges.add(frozenset([nodes[0], nodes[1]]))
        subgraph_edges.add(frozenset([nodes[0], nodes[2]]))
        subgraph_edges.add(frozenset([nodes[1], nodes[2]]))
        subgraph_edges.add(frozenset([nodes[0], nodes[3]]))
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return subgraphs, subgraph_nodes, subgraph_edges


def find_g7_subgraphs(nx_graph: nx.Graph) -> Tuple[List[Tuple[List[int], List[int]]], set, list]:
    subgraphs = []
    seen_hashes = set()
    edges = list(nx_graph.edges())

    for (u, v) in tqdm(edges):
        neighbors_u = set(nx_graph.neighbors(u))
        neighbors_v = set(nx_graph.neighbors(v))
        common_neighbors = neighbors_u & neighbors_v
        common_neighbors.discard(u)
        common_neighbors.discard(v)

        for w, x in itertools.combinations(common_neighbors, 2):
            if not nx_graph.has_edge(w, x):
                g7_nodes = sorted([u, v, w, x])
                g7_hash = hash(tuple(g7_nodes))
                if g7_hash not in seen_hashes:
                    seen_hashes.add(g7_hash)
                    subgraphs.append((g7_nodes, [0, 0, 1, 1]))

    subgraph_nodes = set(itertools.chain.from_iterable([nodes for nodes, _ in subgraphs]))
    subgraph_edges = set()
    for nodes, _ in subgraphs:
        u, v, w, x = nodes
        subgraph_edges.update([
            frozenset([u, v]), frozenset([u, w]), frozenset([u, x]),
            frozenset([v, w]), frozenset([v, x])
        ])
    subgraph_edges = [tuple(sorted(e)) for e in subgraph_edges]
    return subgraphs, subgraph_nodes, subgraph_edges


def sequential_graphlet_filter_original(
        nx_graph: nx.Graph,
        target_graphlet_specs: List[Tuple[str, callable]],
        neighbor_index: Dict
) -> Tuple[Dict[str, List[Tuple[List[int], List[int]]]], set, list]:
    graphlet_subgraphs = {}
    current_nodes = set(nx_graph.nodes())
    current_edges = [tuple(sorted(e)) for e in nx_graph.edges()]
    for idx, (graphlet_name, graphlet_func) in enumerate(target_graphlet_specs):
        print(f"\n=== graphlet {graphlet_name} ===")
        current_subgraph = get_subgraph(nx_graph, current_nodes, current_edges)
        if graphlet_name == "G3":
            subgraphs, sub_nodes, sub_edges = graphlet_func(current_subgraph, current_nodes, neighbor_index)
        else:
            subgraphs, sub_nodes, sub_edges = graphlet_func(current_subgraph)
        current_nodes &= sub_nodes
        current_edges = [e for e in current_edges if e in sub_edges]
        if not current_nodes or not current_edges:
            return graphlet_subgraphs, set(), []
        graphlet_subgraphs[graphlet_name] = subgraphs

    return graphlet_subgraphs, current_nodes, current_edges


def sequential_graphlet_filter_filtered(
        nx_graph: nx.Graph,
        target_graphlet_specs: List[Tuple[str, callable]],
        neighbor_index: Dict
) -> Tuple[Dict[str, List[Tuple[List[int], List[int]]]], set, list]:
    graphlet_subgraphs = {}
    current_nodes = set(nx_graph.nodes())
    current_edges = [tuple(sorted(e)) for e in nx_graph.edges()]
    for idx, (graphlet_name, graphlet_func) in enumerate(target_graphlet_specs):
        print(f"\n=== graphlet {graphlet_name} ===")
        current_subgraph = get_subgraph(nx_graph, current_nodes, current_edges)
        if graphlet_name == "G3":
            subgraphs, sub_nodes, sub_edges = graphlet_func(current_subgraph, current_nodes, neighbor_index)
        else:
            subgraphs, sub_nodes, sub_edges = graphlet_func(current_subgraph)
        current_nodes &= sub_nodes
        current_edges = [e for e in current_edges if e in sub_edges]
        if not current_nodes or not current_edges:
            return graphlet_subgraphs, set(), []
        graphlet_subgraphs[graphlet_name] = subgraphs

    return graphlet_subgraphs, current_nodes, current_edges


def build_filtered_network(
        original_nodes: List[Dict],
        original_edges: List[Dict],
        original_graph: dgl.DGLGraph,
        keyword_to_orig_idx: Dict[str, int],
        retained_orig_nodes: set,
        retained_orig_edges: list,
        new_emb_dict: Dict[str, List[float]]
) -> Tuple[List[Dict], List[Dict], dgl.DGLGraph, Dict[int, int], Dict[int, int]]:
    if not retained_orig_nodes:
        return [], [], None, {}, {}
    retained_orig_indices = sorted(retained_orig_nodes)
    num_filtered_nodes = len(retained_orig_indices)
    orig2new = {orig_idx: new_idx for new_idx, orig_idx in enumerate(retained_orig_indices)}
    new2orig = {new_idx: orig_idx for orig_idx, new_idx in orig2new.items()}

    filtered_nodes = []
    for orig_idx in retained_orig_indices:
        orig_node = original_nodes[orig_idx]
        keyword = orig_node['keyword']
        if keyword not in new_emb_dict:
            raise KeyError(f"'{keyword}' not found")
        new_feat = new_emb_dict[keyword]
        co_occurrence_years = orig_node.get('co_occurrence_years', [])
        co_occurrence_years = [int(y) for y in co_occurrence_years if str(y).isdigit()]
        filtered_node = {
            'keyword': keyword,
            'initial_feature': new_feat,
            'co_occurrence_years': co_occurrence_years,
            'degree_per_year': orig_node.get('degree_per_year', {}),
            'frequency_per_year': orig_node.get('frequency_per_year', {})
        }
        filtered_nodes.append(filtered_node)

    orig_edge_map = {}
    for edge in original_edges:
        src_key = edge['source']
        tgt_key = edge['target']
        if src_key not in keyword_to_orig_idx or tgt_key not in keyword_to_orig_idx:
            continue
        src_orig = keyword_to_orig_idx[src_key]
        tgt_orig = keyword_to_orig_idx[tgt_key]
        orig_edge_map[frozenset([src_orig, tgt_orig])] = edge
    filtered_edges = []
    for edge_tuple in retained_orig_edges:
        edge_key = frozenset(edge_tuple)
        if edge_key in orig_edge_map:
            src_orig, tgt_orig = edge_tuple
            if src_orig in retained_orig_nodes and tgt_orig in retained_orig_nodes:
                filtered_edges.append(orig_edge_map[edge_key].copy())
    unique_edges = {tuple(sorted([e['source'], e['target']])): e for e in filtered_edges}
    filtered_edges = list(unique_edges.values())
    num_filtered_edges = len(filtered_edges)
    if not filtered_edges:
        return filtered_nodes, filtered_edges, None, orig2new, new2orig

    src_new = []
    dst_new = []
    for edge in filtered_edges:
        src_key = edge['source']
        tgt_key = edge['target']
        if src_key not in keyword_to_orig_idx or tgt_key not in keyword_to_orig_idx:
            continue
        src_orig = keyword_to_orig_idx[src_key]
        tgt_orig = keyword_to_orig_idx[tgt_key]
        if src_orig in retained_orig_nodes and tgt_orig in retained_orig_nodes:
            src_new.append(orig2new[src_orig])
            dst_new.append(orig2new[tgt_orig])

    filtered_graph = dgl.graph((src_new, dst_new), num_nodes=num_filtered_nodes).to(original_graph.device)
    initial_feats = torch.tensor(
        [node['initial_feature'] for node in filtered_nodes],
        dtype=torch.float32,
        device=filtered_graph.device
    )
    filtered_graph.ndata['feat'] = initial_feats

    if filtered_edges and 'earliest_year' in filtered_edges[0]:
        earliest_years = torch.tensor(
            [edge['earliest_year'] for edge in filtered_edges],
            dtype=torch.long,
            device=filtered_graph.device
        )
        filtered_graph.edata['earliest_year'] = earliest_years

    return filtered_nodes, filtered_edges, filtered_graph, orig2new, new2orig


def count_node_orbits(
        graphlet_subgraphs: Dict[str, List[Tuple[List[int], List[int]]]],
        num_nodes: int
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    orbit_counts = {}
    orbit_metadata = {}
    for graphlet_name, subgraphs in graphlet_subgraphs.items():
        if not subgraphs:
            continue
        first_subgraph = subgraphs[0]
        if not isinstance(first_subgraph, tuple) or len(first_subgraph) != 2:
            continue
        _, orbits = first_subgraph
        num_orbits = len(orbits)
        if num_orbits == 0:
            continue
        orbit_metadata[graphlet_name] = num_orbits
        counts = torch.zeros((num_nodes, num_orbits), dtype=torch.int32, device=device)
        for subgraph in subgraphs:
            if not isinstance(subgraph, tuple) or len(subgraph) != 2:
                continue
            nodes, orbits = subgraph
            if len(nodes) != len(orbits):
                continue
            for node, orbit in zip(nodes, orbits):
                if 0 <= node < num_nodes:
                    counts[node, orbit] += 1
        orbit_counts[graphlet_name] = counts
    return orbit_counts, orbit_metadata


def save_raw_orbit_counts(
        orbit_counts: Dict[str, torch.Tensor],
        orbit_metadata: Dict[str, int],
        filtered_final_nodes: List[int],
        filtered_nodes: List[Dict],
        new2orig: Dict[int, int],
        save_path: str
) -> None:
    raw_orbit_data = {
        "orbit_metadata": orbit_metadata,
        "nodes": []
    }
    for i, local_idx in enumerate(filtered_final_nodes):
        orig_idx = new2orig[local_idx]
        node = filtered_nodes[local_idx]
        node_orbit_counts = {
            "keyword": node['keyword'],
            "local_index": int(local_idx),
            "original_index": int(orig_idx),
            "orbit_counts_by_graphlet": {}
        }
        for graphlet_name, counts_tensor in orbit_counts.items():
            if i < counts_tensor.shape[0]:
                counts = counts_tensor[i].cpu().numpy().tolist()
                node_orbit_counts["orbit_counts_by_graphlet"][graphlet_name] = counts
        raw_orbit_data["nodes"].append(node_orbit_counts)

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(raw_orbit_data, f, indent=2, ensure_ascii=False)


def save_normalized_orbit_features(
        normalized_orbit_features: torch.Tensor,
        filtered_final_nodes: List[int],
        filtered_nodes: List[Dict],
        new2orig: Dict[int, int],
        orbit_metadata: Dict[str, int],
        save_path: str
) -> None:
    feature_dim_info = []
    current_dim = 0
    for graphlet_name in sorted(orbit_metadata.keys()):
        num_orbits = orbit_metadata[graphlet_name]
        feature_dim_info.append({
            "graphlet_name": graphlet_name,
            "orbit_count": num_orbits,
            "feature_dim_range": [current_dim, current_dim + num_orbits - 1]
        })
        current_dim += num_orbits

    normalized_orbit_data = {
        "feature_dim_info": feature_dim_info,
        "total_feature_dim": current_dim,
        "nodes": []
    }

    for i, local_idx in enumerate(filtered_final_nodes):
        orig_idx = new2orig[local_idx]
        node = filtered_nodes[local_idx]
        if i < normalized_orbit_features.shape[0]:
            orbit_feat = normalized_orbit_features[i].cpu().numpy().tolist()
            orbit_feat = [float(x) for x in orbit_feat]
        else:
            orbit_feat = []

        normalized_orbit_data["nodes"].append({
            "keyword": node['keyword'],
            "local_index": int(local_idx),
            "original_index": int(orig_idx),
            "normalized_orbit_feature": orbit_feat
        })
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_orbit_data, f, indent=2, ensure_ascii=False)


def concatenate_embeddings_with_orbit(
        filtered_nodes: List[Dict],
        filtered_final_nodes: List[int],
        new2orig: Dict[int, int],
        original_nodes: List[Dict],
        emb_dict: Dict[str, List[float]],
        normalized_orbit_features: torch.Tensor,
        save_path: str
) -> None:
    num_valid_nodes = len(filtered_final_nodes)
    final_embeddings = []
    node_idx_map = {orig_idx: node for orig_idx, node in enumerate(filtered_nodes)}
    for i in range(num_valid_nodes):
        local_idx = filtered_final_nodes[i]
        orig_idx = new2orig[local_idx]
        node = node_idx_map[local_idx]
        keyword = node['keyword']
        if keyword not in emb_dict:
            raise KeyError(f"{keyword} not found")
        tgat_emb = [float(x) for x in emb_dict[keyword]]
        orbit_feat = normalized_orbit_features[i].cpu().numpy().tolist()
        orbit_feat = [float(x) for x in orbit_feat]
        concatenated_emb = tgat_emb + orbit_feat
        concatenated_emb = [float(x) for x in concatenated_emb]
        co_occurrence_years = node['co_occurrence_years']
        final_embeddings.append({
            "keyword": keyword,
            "original_index": int(orig_idx),
            "final_embedding": concatenated_emb,
            "tgat_embedding": tgat_emb,
            "orbit_feature": orbit_feat,
            "time_info": {
                "co_occurrence_years": co_occurrence_years
            }
        })
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(final_embeddings, f, indent=2, ensure_ascii=False)


def precompute_candidate_nodes(nx_graph: nx.Graph) -> set:
    candidate_nodes = {n for n in nx_graph.nodes() if nx_graph.degree(n) >= 2}
    return candidate_nodes


def build_indices(nx_graph: nx.Graph, candidate_nodes: set) -> Tuple[Dict, Dict]:
    neighbor_index = {
        n: {neighbor for neighbor in nx_graph.neighbors(n) if neighbor in candidate_nodes}
        for n in candidate_nodes
    }
    edge_index = {
        frozenset([u, v]) for u in candidate_nodes for v in neighbor_index[u]
    }
    return neighbor_index, edge_index



def load_existing_embeddings(emb_path: str) -> Dict[str, List[float]]:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Not found: {emb_path}")
    with open(emb_path, 'r', encoding='utf-8') as f:
        embeddings = json.load(f)
    emb_dict = {
        item['keyword']: [float(x) for x in item['embedding']]
        for item in embeddings
    }
    return emb_dict


def main():
    PRUNED_DIR = "E:/preprocess/pruned_network"
    EMBEDDING_PATH = "E:/model/checkpoint.json"
    FILTERED_DIR = "E:/meta_graph"
    TARGET_GRAPHLET_SPECS = [
        ('G1', find_path3_subgraphs),
        ('G2', find_triangle_subgraphs),
        ('G4', find_star4_subgraphs),
        ('G6', find_g6_subgraphs)
    ]

    FINAL_EMB_SAVE_PATH = os.path.join(FILTERED_DIR, 'final_embedding_with_orbit.json')
    RAW_ORBIT_COUNTS_PATH = os.path.join(FILTERED_DIR, 'raw_orbit_counts.json')
    NORMALIZED_ORBIT_FEATURES_PATH = os.path.join(FILTERED_DIR, 'normalized_orbit_features.json')

    try:
        original_nodes, original_edges, original_graph, keyword_to_orig_idx = load_pruned_network(PRUNED_DIR)

        new_emb_dict = load_existing_embeddings(EMBEDDING_PATH)

        original_nx_graph = dgl_to_nx(original_graph, original_nodes)

        candidate_nodes = precompute_candidate_nodes(original_nx_graph)
        neighbor_index, _ = build_indices(original_nx_graph, candidate_nodes)

        _, retained_orig_nodes, retained_orig_edges = sequential_graphlet_filter_original(
            original_nx_graph, TARGET_GRAPHLET_SPECS, neighbor_index
        )
        if not retained_orig_nodes or not retained_orig_edges:
            return

        filtered_nodes, filtered_edges, filtered_graph_step5, orig2new, new2orig = build_filtered_network(
            original_nodes, original_edges, original_graph, keyword_to_orig_idx,
            retained_orig_nodes, retained_orig_edges, new_emb_dict
        )
        if not filtered_nodes or not filtered_edges or filtered_graph_step5 is None:
            return

        filtered_nx_graph = dgl_to_nx(filtered_graph_step5, filtered_nodes)
        filtered_candidate_nodes = precompute_candidate_nodes(filtered_nx_graph)
        filtered_neighbor_index, _ = build_indices(filtered_nx_graph, filtered_candidate_nodes)
        graphlet_subgraphs, filtered_final_nodes, filtered_final_edges = sequential_graphlet_filter_filtered(
            filtered_nx_graph, TARGET_GRAPHLET_SPECS, filtered_neighbor_index
        )
        filtered_final_nodes = sorted(list(filtered_final_nodes))
        num_filtered_final_nodes = len(filtered_final_nodes)
        if num_filtered_final_nodes == 0:
            return

        orbit_counts, orbit_metadata = count_node_orbits(graphlet_subgraphs, num_filtered_final_nodes)
        if not orbit_counts:
            return

        save_raw_orbit_counts(
            orbit_counts=orbit_counts,
            orbit_metadata=orbit_metadata,
            filtered_final_nodes=filtered_final_nodes,
            filtered_nodes=filtered_nodes,
            new2orig=new2orig,
            save_path=RAW_ORBIT_COUNTS_PATH
        )

        orbit_features = torch.cat([orbit_counts[name] for name in sorted(orbit_counts.keys())], dim=1)
        if orbit_features.shape[0] != num_filtered_final_nodes:
            raise ValueError(f"{orbit_features.shape[0]} != {num_filtered_final_nodes}")
        orbit_mean = orbit_features.float().mean(dim=0)
        orbit_std = orbit_features.float().std(dim=0)
        orbit_std[orbit_std == 0] = 1.0
        normalized_orbit_features = (orbit_features.float() - orbit_mean) / orbit_std
        print(f"normalized orbit feature shape：{normalized_orbit_features.shape}")

        save_normalized_orbit_features(
            normalized_orbit_features=normalized_orbit_features,
            filtered_final_nodes=filtered_final_nodes,
            filtered_nodes=filtered_nodes,
            new2orig=new2orig,
            orbit_metadata=orbit_metadata,
            save_path=NORMALIZED_ORBIT_FEATURES_PATH
        )

        emb_dict = load_existing_embeddings(EMBEDDING_PATH)
        concatenate_embeddings_with_orbit(
            filtered_nodes=filtered_nodes,
            filtered_final_nodes=filtered_final_nodes,
            new2orig=new2orig,
            original_nodes=original_nodes,
            emb_dict=emb_dict,
            normalized_orbit_features=normalized_orbit_features,
            save_path=FINAL_EMB_SAVE_PATH
        )

        final_orig_indices = [new2orig[local_idx] for local_idx in filtered_final_nodes]
        final_orig2new = {orig_idx: i for i, orig_idx in enumerate(final_orig_indices)}

        final_edges = []
        src_new = []
        dst_new = []
        for edge in filtered_edges:
            src_key = edge['source']
            tgt_key = edge['target']
            if src_key not in keyword_to_orig_idx or tgt_key not in keyword_to_orig_idx:
                continue
            src_orig = keyword_to_orig_idx[src_key]
            tgt_orig = keyword_to_orig_idx[tgt_key]
            if src_orig in final_orig_indices and tgt_orig in final_orig_indices:
                final_edges.append(edge)
                src_new.append(final_orig2new[src_orig])
                dst_new.append(final_orig2new[tgt_orig])

        final_dgl_graph = dgl.graph((src_new, dst_new), num_nodes=num_filtered_final_nodes).to(device)

        with open(FINAL_EMB_SAVE_PATH, 'r', encoding='utf-8') as f:
            final_embeddings = json.load(f)
        keyword_to_embedding = {item['keyword']: item['final_embedding'] for item in final_embeddings}

        final_features = []
        for local_idx in filtered_final_nodes:
            keyword = filtered_nodes[local_idx]['keyword']
            if keyword in keyword_to_embedding:
                final_features.append(keyword_to_embedding[keyword])
            else:
                final_features.append(filtered_nodes[local_idx]['initial_feature'])

        final_features = torch.tensor(final_features, dtype=torch.float32, device=device)
        final_dgl_graph.ndata['feat'] = final_features

        os.makedirs(FILTERED_DIR, exist_ok=True)
        final_filtered_nodes = [filtered_nodes[i] for i in filtered_final_nodes]
        final_filtered_edges = [e for e in final_edges]

        with open(os.path.join(FILTERED_DIR, 'filtered_nodes.json'), 'w', encoding='utf-8') as f:
            json.dump(final_filtered_nodes, f, indent=2, ensure_ascii=False)
        with open(os.path.join(FILTERED_DIR, 'filtered_edges.json'), 'w', encoding='utf-8') as f:
            json.dump(final_filtered_edges, f, indent=2, ensure_ascii=False)
        dgl.save_graphs(os.path.join(FILTERED_DIR, 'filtered_graph.bin'), [final_dgl_graph])

    except Exception as e:
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()