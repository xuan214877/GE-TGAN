import json
import os
from collections import defaultdict

def load_cluster_mapping(cluster_result_path):
    keyword_to_cluster = {}
    cluster_to_keywords = defaultdict(list)

    with open(cluster_result_path, 'r', encoding='utf-8') as f:
        cluster_data = json.load(f)

    for item in cluster_data:
        keyword = item['keyword']
        cluster_id = item['cluster_id']
        keyword_to_cluster[keyword] = cluster_id
        cluster_to_keywords[cluster_id].append(keyword)

    return keyword_to_cluster, cluster_to_keywords

def load_cooccurrence_edges(edges_path):
    with open(edges_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)

    keyword_pairs_with_time = []
    for edge in edges:
        src = edge['source']
        tgt = edge['target']
        time = edge.get('co_occurrence_years')
        if src != tgt and time is not None:
            if isinstance(time, list):
                if time:
                    keyword_pairs_with_time.append((src, tgt, time))
            else:
                keyword_pairs_with_time.append((src, tgt, [time]))
    return keyword_pairs_with_time


def get_max_time(edges_path):
    with open(edges_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)
    times = []
    for edge in edges:
        time = edge.get('co_occurrence_years')
        if time is not None:
            if isinstance(time, list):
                times.append(max(time) if time else 0)
            else:
                times.append(time)
    return max(times) if times else 0

def calculate_cluster_correlation(keyword_pairs_with_time, keyword_to_cluster, t_now, beta):
    cluster_edge_weights = defaultdict(float)
    total_valid_edges = 0
    invalid_edges = 0
    for src, tgt, time_list in keyword_pairs_with_time:
        if src not in keyword_to_cluster or tgt not in keyword_to_cluster:
            invalid_edges += 1
            continue

        c1 = keyword_to_cluster[src]
        c2 = keyword_to_cluster[tgt]

        if c1 != c2:
            for t in time_list:
                time_diff = t_now - t
                if time_diff < 0:
                    time_diff = 0
                weight = 1 / (beta ** time_diff) if beta != 0 else 1.0
                cluster_pair = frozenset([c1, c2])
                cluster_edge_weights[cluster_pair] += 1 * weight
                total_valid_edges += 1

    return cluster_edge_weights


def normalize_correlation_strength(cluster_edge_weights, cluster_to_keywords):
    normalized_strength = {}
    for cluster_pair, total_weight in cluster_edge_weights.items():
        c1, c2 = cluster_pair
        size1 = len(cluster_to_keywords[c1])
        size2 = len(cluster_to_keywords[c2])
        total_nodes = size1 + size2

        if total_nodes == 0:
            normalized = 0.0
        else:
            normalized = total_weight / total_nodes
        normalized_strength[(c1, c2)] = {
            "total_weight": total_weight,
            "normalized_strength": normalized,
            "cluster_sizes": (size1, size2),
            "total_nodes": total_nodes
        }
    return normalized_strength

def save_correlation_results(normalized_strength, output_path):
    results = {}
    for (c1, c2), data in normalized_strength.items():
        pair_key = f"cluster_{c1}-cluster_{c2}"
        results[pair_key] = data

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def main():
    CLUSTER_RESULT_PATH = "E:/kmeans/labeled_opportunities_k13.json"
    EDGES_PATH = "E:/meta_graph/filtered_edges.json"
    OUTPUT_PATH = "E:/merge/cluster_correlation_strength.json"
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    keyword_to_cluster, cluster_to_keywords = load_cluster_mapping(CLUSTER_RESULT_PATH)

    keyword_pairs_with_time = load_cooccurrence_edges(EDGES_PATH)

    t_now = get_max_time(EDGES_PATH)
    beta = 0.9

    cluster_edge_weights = calculate_cluster_correlation(keyword_pairs_with_time, keyword_to_cluster, t_now, beta)

    normalized_strength = normalize_correlation_strength(cluster_edge_weights, cluster_to_keywords)

    save_correlation_results(normalized_strength, OUTPUT_PATH)

if __name__ == "__main__":
    main()