import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Dict, Tuple
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

plt.rcParams["axes.unicode_minus"] = False


def get_custom_window(year: int) -> Tuple[int, List[int]]:
    if 2015 <= year <= 2018:
        return 2018, [2015, 2018]
    else:
        return year, [year, year]


def load_final_embeddings(embedding_path: str) -> Tuple[
    Dict[int, List[Dict]], Dict[str, int], List[int]]:
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Not found: {embedding_path}")

    with open(embedding_path, 'r', encoding='utf-8') as f:
        try:
            embeddings = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"{str(e)}")

    all_years = set()
    empty_time_info_count = 0
    empty_co_years_count = 0

    for idx, item in enumerate(embeddings):
        if 'time_info' not in item:
            empty_time_info_count += 1
            continue

        time_info = item['time_info']

        if 'co_occurrence_years' not in time_info:
            continue
        co_years = time_info['co_occurrence_years']
        if not isinstance(co_years, list):
            continue

        valid_years = []
        for y in co_years:
            if isinstance(y, int):
                valid_years.append(y)
            else:
                print(f"warning {idx} co_occurrence_years: {y}")
        if not valid_years:
            empty_co_years_count += 1
            continue

        all_years.update(valid_years)

    window_map = {}
    for year in all_years:
        window_label, window_years = get_custom_window(year)
        window_map[window_label] = window_years
    window_labels = sorted(window_map.keys())
    for label in window_labels:
        if label not in window_map:
            window_map[label] = [label, label]

    time_nodes = defaultdict(list)
    keyword_to_idx = {item['keyword']: idx for idx, item in enumerate(embeddings)}

    for item in embeddings:
        keyword = item['keyword']
        time_info = item.get('time_info', {})
        co_years = time_info.get('co_occurrence_years', [])
        valid_years = [y for y in co_years if isinstance(y, int)]
        node_windows = set()
        for year in valid_years:
            window_label, _ = get_custom_window(year)
            node_windows.add(window_label)

        for window_label in node_windows:
            node_with_window = item.copy()
            node_with_window['time_window'] = window_label
            node_with_window['window_years'] = window_map[window_label]
            time_nodes[window_label].append(node_with_window)

    return time_nodes, keyword_to_idx, window_labels


def calculate_lof_for_window(embeddings: List[Dict], n_neighbors: int = 20) -> np.ndarray:
    if not embeddings:
        return np.array([])

    embedding_matrix = np.array([item['final_embedding'] for item in embeddings])
    if len(embedding_matrix) <= n_neighbors:
        n_neighbors = max(1, len(embedding_matrix) - 1)

    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embedding_matrix)

    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=False)
    lof.fit(scaled_embeddings)
    lof_scores = -lof.negative_outlier_factor_

    for i, item in enumerate(embeddings):
        item['lof_score'] = lof_scores[i]

    return lof_scores


def identify_expansion_phase_opportunities(
        time_nodes: Dict[int, List[Dict]],
        window_labels: List[int],
        threshold_factor: float = 1.0
) -> Tuple[List[Dict], Dict[int, Dict]]:
    opportunities = []
    time_stats = {}
    last_window = max(window_labels)

    for window in window_labels:
        nodes = time_nodes.get(window, [])
        if not nodes:
            continue
        lof_scores = [node['lof_score'] for node in nodes]
        time_stats[window] = {
            'mean': np.mean(lof_scores),
            'std': np.std(lof_scores),
            'threshold': np.mean(lof_scores) + threshold_factor * np.std(lof_scores),
            'node_count': len(nodes)
        }

    all_keywords = {node['keyword'] for nodes in time_nodes.values() for node in nodes}
    keyword_history = defaultdict(lambda: {'scores': [], 'windows': []})

    for keyword in all_keywords:
        for window in window_labels:
            node = next((n for n in time_nodes.get(window, []) if n['keyword'] == keyword), None)
            if node:
                keyword_history[keyword]['scores'].append(node['lof_score'])
                keyword_history[keyword]['windows'].append(window)

    for keyword, history in keyword_history.items():
        scores = history['scores']
        windows = history['windows']
        if len(scores) < 2:
            continue
        for i in range(1, len(scores)):
            prev_window = windows[i - 1]
            curr_window = windows[i]
            prev_score = scores[i - 1]
            curr_score = scores[i]
            if curr_score > prev_score and curr_window in time_stats:
                threshold = time_stats[curr_window]['threshold']
                if curr_score > threshold:
                    curr_node = next((n for n in time_nodes[curr_window] if n['keyword'] == keyword), None)
                    if curr_node:
                        curr_node['opportunity_type'] = 'expansion_phase'
                        curr_node['lof_growth'] = curr_score - prev_score
                        curr_node['prev_window'] = prev_window
                        curr_node['prev_lof_score'] = prev_score
                        curr_node['prev_window_years'] = curr_node.get('window_years', [prev_window, prev_window])
                        curr_node['curr_window_years'] = curr_node['window_years']
                        opportunities.append(curr_node)
                    break

    if last_window in time_nodes and last_window in time_stats:
        threshold = time_stats[last_window]['threshold']
        for node in time_nodes[last_window]:
            if node['lof_score'] > threshold:
                if not any(opp['keyword'] == node['keyword'] and opp['time_window'] == last_window for opp in
                           opportunities):
                    node['opportunity_type'] = 'last_window_outlier'
                    node['lof_growth'] = 0
                    node['prev_window'] = None
                    node['prev_lof_score'] = None
                    opportunities.append(node)

    return opportunities, time_stats


def visualize_opportunity_growth(opportunities: List[Dict], time_nodes: Dict[int, List[Dict]],
                                 window_labels: List[int]):
    plt.figure(figsize=(14, 8))
    all_scores = []
    for window in window_labels:
        nodes = time_nodes.get(window, [])
        all_scores.extend([node['lof_score'] for node in nodes if 'lof_score' in node])
    if not all_scores:
        return

    percentiles = np.percentile(all_scores, [25, 75])
    plt.fill_between(range(len(window_labels)), [percentiles[0]] * len(window_labels),
                     [percentiles[1]] * len(window_labels),
                     color='lightgray', alpha=0.5, label='LOF 25%-75%区间')

    window_display_labels = []
    for label in window_labels:
        if label == 2018:
            window_display_labels.append("2015-2018")
        else:
            window_display_labels.append(str(label))

    for opp in opportunities[:10]:
        keyword = opp['keyword']
        valid_windows = []
        valid_scores = []
        for window in window_labels:
            node = next((n for n in time_nodes.get(window, []) if n['keyword'] == keyword), None)
            if node:
                valid_windows.append(window)
                valid_scores.append(node['lof_score'])
        expansion_window = opp['time_window']
        exp_idx = valid_windows.index(expansion_window) if expansion_window in valid_windows else -1
        if exp_idx == -1:
            continue
        if opp['opportunity_type'] == 'expansion_phase':
            prev_window = opp['prev_window']
            prev_idx = valid_windows.index(prev_window) if prev_window in valid_windows else -1
            if prev_idx != -1:
                plt.scatter([prev_idx, exp_idx], [valid_scores[prev_idx], valid_scores[exp_idx]],
                            s=120, marker='*', edgecolors='red', zorder=10)
        else:
            plt.scatter(exp_idx, valid_scores[exp_idx],
                        s=120, marker='*', edgecolors='blue', zorder=10)
        window_indices = [window_labels.index(w) for w in valid_windows]
        plt.plot(window_indices, valid_scores, marker='o',
                 label=f"{keyword[:8]}（{opp['opportunity_type']}）", alpha=0.8)

    plt.xlabel('time window')
    plt.ylabel('LOF score')
    plt.title('trend')
    plt.xticks(range(len(window_labels)), window_display_labels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()


def analyze_phase_distribution(time_nodes: Dict[int, List[Dict]], window_labels: List[int]) -> pd.DataFrame:
    phase_counts = defaultdict(lambda: defaultdict(int))
    for window in window_labels:
        nodes = time_nodes.get(window, [])
        for node in nodes:
            score = node['lof_score']
            if score > 1.8:
                phase = 'isolation'
            elif score > 1.5:
                phase = 'expansion'
            elif score > 1.2:
                phase = 'aggregation'
            else:
                phase = 'maturity'
            phase_counts[window][phase] += 1

    window_display_map = {label: "2015-2018" if label == 2018 else str(label) for label in window_labels}
    phase_df = pd.DataFrame.from_dict(phase_counts, orient='index').fillna(0)
    phase_df = phase_df.reindex(columns=['expansion', 'aggregation', 'maturity', 'isolation'], fill_value=0)
    phase_df.index = [window_display_map[idx] for idx in phase_df.index]
    phase_df = phase_df.reindex([window_display_map[label] for label in window_labels])

    plt.figure(figsize=(12, 6))
    phase_df.plot(kind='bar', stacked=True, width=0.8)
    plt.xlabel('time window')
    plt.ylabel('numbers')
    plt.title('technology phase distribution')
    plt.legend(title='technology phase')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return phase_df


def main():
    EMBEDDING_PATH = "E:/meta_graph/concatenated_embeddings1.json"
    OUTPUT_PATH = "E:/lof"
    N_NEIGHBORS = 20
    THRESHOLD_FACTOR = 1.0

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    try:
        time_nodes, keyword_to_idx, window_labels = load_final_embeddings(EMBEDDING_PATH)
        for window in window_labels:
            nodes = time_nodes.get(window, [])
            if not nodes:
                continue
            lof_scores = calculate_lof_for_window(nodes, N_NEIGHBORS)

        opportunities, time_stats = identify_expansion_phase_opportunities(time_nodes, window_labels, THRESHOLD_FACTOR)

        output_file = os.path.join(OUTPUT_PATH, "all_opportunities.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(opportunities, f, indent=2, ensure_ascii=False)

        visualize_opportunity_growth(opportunities, time_nodes, window_labels)
        phase_df = analyze_phase_distribution(time_nodes, window_labels)
        phase_df.to_csv(os.path.join(OUTPUT_PATH, "phase_distribution.csv"), index_label='time_window')

        enhanced_time_stats = {}
        for window, stats in time_stats.items():
            window_years = next((node['window_years'] for nodes in time_nodes.values() for node in nodes if node['time_window'] == window), [window, window])
            enhanced_time_stats[window] = {
                **stats,
                'window_years': window_years
            }
        stats_file = os.path.join(OUTPUT_PATH, "time_window_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_time_stats, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print(f"{str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()