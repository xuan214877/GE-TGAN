import os
import time
import json
import math
import random
import itertools
import community as community_louvain
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
from multiprocessing import Pool, cpu_count
from graphlet_templates import create_graphlet_templates

os.environ["NX_CUGRAPH_AUTOCONFIG"] = "False"

def process_subset(args):
    subgraph, nodes_subset, k_templates = args
    subg = subgraph.subgraph(nodes_subset).copy()
    subg = nx.Graph(subg.edges())

    try:
        from networkx.algorithms.isomorphism import weisfeiler_lehman_graph_hash
        subg_hash = weisfeiler_lehman_graph_hash(subg)
    except ImportError:
        subg_hash = nx.to_graph6_bytes(subg).decode()

    for name, template in k_templates.items():
        template_clean = nx.Graph(template.edges())
        try:
            template_hash = weisfeiler_lehman_graph_hash(template_clean)
        except:
            template_hash = nx.to_graph6_bytes(template_clean).decode()
        if subg_hash == template_hash:
            return name
    return None


def count_graphlets_with_sampling(subgraph, k, templates, node_sample_ratio=0.3, combo_sample_base=100000):
    counts = defaultdict(int)
    nodes = list(subgraph.nodes())
    n_total = len(nodes)
    if n_total < k:
        return counts, 0

    node_degrees = dict(subgraph.degree(nodes))
    sorted_nodes = sorted(nodes, key=lambda x: node_degrees[x], reverse=True)
    top_degree_nodes = sorted_nodes[:int(n_total * 0.3)]

    communities = []
    if n_total >= 10:
        partition = community_louvain.best_partition(subgraph)
        community_dict = defaultdict(list)
        for node, comm_id in partition.items():
            community_dict[comm_id].append(node)
        communities = [comm for comm in community_dict.values() if 5 <= len(comm) <= n_total * 0.5]
        

    sample_size = max(int(n_total * node_sample_ratio), k)
    sample_nodes = set()

    take_top = min(len(top_degree_nodes), int(sample_size * 0.4))
    sample_nodes.update(top_degree_nodes[:take_top])

    if communities:
        community_nodes = []
        comm_quota = int(sample_size * 0.3)
        total_comm_size = sum(len(comm) for comm in communities)
        for comm in communities:
            comm_ratio = len(comm) / total_comm_size
            comm_take = max(1, int(comm_quota * comm_ratio))
            comm_sample = random.sample(comm, min(len(comm), comm_take))
            community_nodes.extend(comm_sample)
        sample_nodes.update(community_nodes)

    remaining = sample_size - len(sample_nodes)
    if remaining > 0:
        non_sample = [n for n in nodes if n not in sample_nodes]
        if non_sample:
            take_remaining = min(remaining, len(non_sample))
            sample_nodes.update(random.sample(non_sample, take_remaining))

    sample_nodes = list(sample_nodes)
    subgraph = subgraph.subgraph(sample_nodes).copy()
    sample_nodes = list(subgraph.nodes())
    actual_sample_size = len(sample_nodes)
    if actual_sample_size < k:
        return counts, actual_sample_size

    if actual_sample_size > 1:
        possible_edges = actual_sample_size * (actual_sample_size - 1) / 2
        density = subgraph.number_of_edges() / possible_edges
    else:
        density = 0
    combo_sample_size = int(combo_sample_base * (1 - 0.5 * density))
    total_combos = math.comb(actual_sample_size, k)
    combo_sample_size = min(combo_sample_size, total_combos)

    k_templates = {name: t for name, t in templates.items() if len(t.nodes()) == k}
    if not k_templates:
        raise ValueError(f"Not fund: {k} graphlet template")

    combos = itertools.islice(itertools.combinations(sample_nodes, k), combo_sample_size)
    args_list = [(subgraph, subset, k_templates) for subset in combos]
    if not args_list:
        return counts, actual_sample_size

    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        results = pool.imap_unordered(process_subset, args_list)
        for res in tqdm(results, total=len(args_list), desc=f"k={k}", position=0, leave=True):
            if res:
                counts[res] += 1

    if k == 3 and not counts:
        edge_count = subgraph.number_of_edges()
        if edge_count > 0:
            counts["G1"] = 1
    return counts, actual_sample_size


def calculate_acceleration(window_results, graphlet_name):
    valid_counts = []
    valid_windows = []
    for wr in window_results:
        cnt = wr["counts"].get(graphlet_name, 0)
        valid_cnt = cnt if cnt > 0 else 1e-6
        valid_counts.append(valid_cnt)
        valid_windows.append(wr["window"])

    if len(valid_counts) < 3:
        return []

    growth_rates = []
    for i in range(1, len(valid_counts)):
        prev = valid_counts[i - 1]
        curr = valid_counts[i]
        growth_rates.append((curr - prev) / prev)

    accelerations = []
    for i in range(1, len(growth_rates)):
        prev_i = growth_rates[i - 1]
        curr_i = growth_rates[i]
        if abs(prev_i) < 1e-9:
            accel = 0.0 if abs(curr_i) < 1e-9 else float('inf')
        else:
            accel = (curr_i - prev_i) / prev_i

        accelerations.append({
            "windows": (valid_windows[i], valid_windows[i + 1]),
            "growth_rate_prev": prev_i,
            "growth_rate_curr": curr_i,
            "acceleration": accel
        })
    return accelerations


def analyze_graphlet_evolution(edge_info, templates,
                               node_sample_ratio=0.3,
                               combo_sample_base=100000,
                               acceleration_threshold=0.5,
                               save_dir="results"):
    windows = [(2015, 2018)]
    for year in range(2019, 2026):
        windows.append((year, year))

    window_pbar = tqdm(windows, position=0, leave=True)

    window_results = []
    intermediate_dir = os.path.join(save_dir, "intermediate_results")
    os.makedirs(intermediate_dir, exist_ok=True)

    for idx, win in enumerate(window_pbar):
        start, end = win
        window_pbar.set_postfix_str(f"time window: {start}-{end}")
        G = nx.Graph()
        for edge in edge_info:
            if any(start <= y <= end for y in edge.get('co_occurrence_years', [])):
                G.add_edge(edge['source'], edge['target'])

        graphlet_counts = defaultdict(int)
        max_sample_size = 0
        if G.number_of_nodes() >= 3:
            for k in [3, 4]:
                if G.number_of_nodes() < k:
                    continue
                counts, sample_size = count_graphlets_with_sampling(
                    G, k, templates,
                    node_sample_ratio=node_sample_ratio,
                    combo_sample_base=combo_sample_base
                )
                if sample_size > max_sample_size:
                    max_sample_size = sample_size
                for name, cnt in counts.items():
                    graphlet_counts[name] += cnt

        current_result = {
            "window": (start, end),
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "sample_size": max_sample_size,
            "counts": dict(graphlet_counts)
        }
        window_results.append(current_result)

        for g_name, cnt in graphlet_counts.items():
            print(f"  {g_name}: {cnt}")
        if not graphlet_counts:
            print("  Warning: No graphlet detected!")

        intermediate_file = os.path.join(intermediate_dir, f"window_{start}-{end}.json")
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(current_result, f, indent=2, ensure_ascii=False)

    significant_accelerations = defaultdict(list)
    all_graphlets = set()
    for wr in window_results:
        all_graphlets.update(wr["counts"].keys())

    for g_name in all_graphlets:
        accels = calculate_acceleration(window_results, g_name)
        for accel_info in accels:
            if accel_info["acceleration"] > acceleration_threshold:
                significant_accelerations[g_name].append(accel_info)

    return {
        "window_details": window_results,
        "significant_accelerations": dict(significant_accelerations)
    }


def main():
    SAVE_DIR = "E:/meta_graph"
    edge_path = "E:/preprocess/pruned_network"
    EDGE_PATH = os.path.join(edge_path, "pruned_edges1.json")

    os.makedirs(SAVE_DIR, exist_ok=True)

    try:
        with open(EDGE_PATH, "r", encoding="utf-8") as f:
            edge_info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"check: {EDGE_PATH}")

    templates = create_graphlet_templates()

    results = analyze_graphlet_evolution(
        edge_info=edge_info,
        templates=templates,
        node_sample_ratio=0.6,
        combo_sample_base=300000,
        acceleration_threshold=0.3,
        save_dir=SAVE_DIR
    )

    final_result_path = os.path.join(SAVE_DIR, "graphlet_optimized_results.json")
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()