import os
import time
import json
import math
import random
import itertools
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from multiprocessing import Pool, cpu_count

SIGNIFICANCE_PARAMS = {
    'P': 0.05,
    'f_min': 2,
    'delta_min': 0.05
}


def precompute_template_hashes(templates):
    template_hashes = defaultdict(dict)
    template_edge_counts = defaultdict(dict)
    for name, template in templates.items():
        k = len(template.nodes())
        template_clean = nx.Graph(template.edges())
        try:
            from networkx.algorithms.isomorphism import weisfeiler_lehman_graph_hash
            thash = weisfeiler_lehman_graph_hash(template_clean)
        except ImportError:
            thash = nx.to_graph6_bytes(template_clean).decode()
        template_hashes[k][name] = thash
        template_edge_counts[k][name] = template_clean.number_of_edges()
    return template_hashes, template_edge_counts


def process_subset(args):
    subgraph, nodes_subset, k, template_hashes_k, template_edge_counts_k = args
    nodes_set = set(nodes_subset)
    if len(nodes_set) != k:
        return []

    sub_edges = []
    for u in nodes_subset:
        for v in subgraph.neighbors(u):
            if v in nodes_set and u < v:
                sub_edges.append((u, v))
    sub_edge_count = len(sub_edges)
    subg_clean = nx.Graph(sub_edges)
    try:
        from networkx.algorithms.isomorphism import weisfeiler_lehman_graph_hash
        subg_hash = weisfeiler_lehman_graph_hash(subg_clean)
    except ImportError:
        subg_hash = nx.to_graph6_bytes(subg_clean).decode()
    matches = []
    for name, edge_count in template_edge_counts_k.items():
        if sub_edge_count != edge_count:
            continue
        if subg_hash == template_hashes_k[name]:
            matches.append((name, tuple(sorted(nodes_subset))))
    return matches


def get_graphlet_details(subgraph, k, template_hashes, template_edge_counts,
                         node_sample_size=None, combo_sample_size=None):
    details = defaultdict(lambda: {'count': 0, 'subgraphs': []})
    nodes = list(subgraph.nodes())
    if not nodes or len(nodes) < k:
        return details

    if node_sample_size and node_sample_size < len(nodes):
        node_degrees = sorted(subgraph.degree(), key=lambda x: x[1], reverse=True)
        top_nodes = [n for n, d in node_degrees[:node_sample_size]]
        nodes = top_nodes
        subgraph = subgraph.subgraph(nodes)

    k_templates_hash = template_hashes.get(k, {})
    k_templates_edges = template_edge_counts.get(k, {})
    if not k_templates_hash:
        return details
    total_combos = math.comb(len(nodes), k)
    combos = itertools.combinations(nodes, k)
    if combo_sample_size and combo_sample_size < total_combos:
        combo_list = list(itertools.islice(combos, combo_sample_size * 2))
        random.shuffle(combo_list)
        combos = itertools.islice(combo_list, combo_sample_size)
    args_list = [
        (subgraph, subset, k, k_templates_hash, k_templates_edges)
        for subset in combos
    ]
    max_processes = max(1, cpu_count() - 4)
    with Pool(processes=max_processes) as pool:
        for res in tqdm(pool.imap(process_subset, args_list),
                        total=min(total_combos, combo_sample_size)):
            for name, nodes_subset in res:
                details[name]['count'] += 1
                if len(details[name]['subgraphs']) < 1000:
                    details[name]['subgraphs'].append(nodes_subset)
    if combo_sample_size and combo_sample_size < total_combos:
        scale_factor = total_combos / combo_sample_size
        for name in details:
            details[name]['count'] = int(details[name]['count'] * scale_factor)
    return details


def create_random_network(real_network, seed=None):
    n = real_network.number_of_nodes()
    m = real_network.number_of_edges()
    return nx.gnm_random_graph(n, m, seed=seed)


def is_significant(real_count, rand_counts, params):
    if real_count < params['f_min']:
        return False
    rand_mean = sum(rand_counts) / len(rand_counts) if rand_counts else 0
    if rand_mean == 0:
        return True
    if (real_count - rand_mean) <= params['delta_min'] * rand_mean:
        return False
    p_value = sum(1 for rc in rand_counts if rc < real_count) / len(rand_counts)
    return p_value >= params['P']


def analyze_graphlet_significance(edge_info, templates, template_hashes, template_edge_counts,
                                  node_sample_size=5000,
                                  combo_sample_size=200000,
                                  num_rand_networks=3):
    G = nx.Graph()
    for edge in tqdm(edge_info):
        G.add_edge(edge['source'], edge['target'])

    if G.number_of_nodes() == 0:
        return {"error": "There are no nodes in the network"}

    real_graphlets = {}
    for k in [3, 4]:
        details = get_graphlet_details(
            G, k, template_hashes, template_edge_counts,
            node_sample_size=node_sample_size,
            combo_sample_size=combo_sample_size
        )
        real_graphlets.update(details)

    rand_graphlets = defaultdict(list)
    SEEDS = [42, 123, 2025][:num_rand_networks]
    for seed in tqdm(SEEDS):
        G_rand = create_random_network(G, seed=seed)
        for k in [3, 4]:
            details = get_graphlet_details(
                G_rand, k, template_hashes, template_edge_counts,
                node_sample_size=node_sample_size,
                combo_sample_size=combo_sample_size
            )
            for name, info in details.items():
                rand_graphlets[name].append(info['count'])

    significant = defaultdict(dict)
    for name, real_info in real_graphlets.items():
        if name not in rand_graphlets:
            significant[name] = real_info
            continue
        if is_significant(real_info['count'], rand_graphlets[name], SIGNIFICANCE_PARAMS):
            significant[name] = real_info

    sig_nodes = set()
    sig_edges = set()
    for name, info in significant.items():
        for subgraph_nodes in info['subgraphs']:
            sig_nodes.update(subgraph_nodes)
            for u, v in itertools.combinations(subgraph_nodes, 2):
                if G.has_edge(u, v):
                    sig_edges.add(tuple(sorted((u, v))))

    sig_subgraph = G.subgraph(sig_nodes).copy()
    sig_subgraph.remove_edges_from([e for e in sig_subgraph.edges()
                                    if tuple(sorted(e)) not in sig_edges])

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "significant_graphlets": {
            name: {
                "count": info["count"],
                "subgraph_count": len(info["subgraphs"]),
                "subgraphs": info["subgraphs"]
            }
            for name, info in significant.items()
        },
        "significant_nodes": list(sig_nodes),
        "significant_edges": list(sig_edges),
        "subgraph_stats": {
            "nodes": sig_subgraph.number_of_nodes(),
            "edges": sig_subgraph.number_of_edges()
        },
        "params": SIGNIFICANCE_PARAMS
    }


def create_graphlet_templates():
    templates = {}
    templates['G1'] = nx.Graph([(0, 1), (1, 2)])
    templates['G2'] = nx.Graph([(0, 1), (1, 2), (0, 2)])
    templates['G3'] = nx.Graph([(0, 1), (1, 2), (2, 3)])
    templates['G4'] = nx.Graph([(0, 1), (0, 2), (0, 3)])
    templates['G5'] = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    templates['G6'] = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2)])
    templates['G7'] = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)])
    templates['G8'] = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    return templates


def main():
    SAVE_DIR = "E:/preprocess/pruned_network"
    EDGE_PATH = os.path.join(SAVE_DIR, "pruned_edges1.json")
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(EDGE_PATH, "r", encoding="utf-8") as f:
        edge_info = json.load(f)

    templates = create_graphlet_templates()
    template_hashes, template_edge_counts = precompute_template_hashes(templates)

    results = analyze_graphlet_significance(
        edge_info=edge_info,
        templates=templates,
        template_hashes=template_hashes,
        template_edge_counts=template_edge_counts,
        node_sample_size=1000,
        combo_sample_size=100000,
        num_rand_networks=2
    )

    output_path = os.path.join(SAVE_DIR, "whole_graph_significance_optimized.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()