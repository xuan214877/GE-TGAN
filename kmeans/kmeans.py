import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

plt.rcParams["axes.unicode_minus"] = False

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def convert_numpy_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data


def load_excel_data(excel_path: str) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    df = pd.read_excel(excel_path, engine='openpyxl')

    required_cols = ["Keyword", "Cluster_embedding", "Visual_embedding", "Main_opp_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    def str_to_emb(emb_str: str) -> np.ndarray:
        try:
            emb_str = emb_str.strip().replace('\n', '').replace(' ', '').replace("'", '"')
            return np.array(json.loads(emb_str), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse embedding: {e}")

    cluster_embeddings = []
    viz_embeddings = []
    opportunities = []

    for idx, row in df.iterrows():
        try:
            cluster_emb = str_to_emb(row["Cluster_embedding"])
            viz_emb = str_to_emb(row["Visual_embedding"])

            cluster_embeddings.append(cluster_emb)
            viz_embeddings.append(viz_emb)

            opp = {
                "keyword": row["Keyword"],
                "final_embedding": cluster_emb.tolist(),
                "opportunity_type": row["Main_opp_type"],

                "original_repeat_count": row.get("Original_repeat_count", 1),
                "lof_score": row.get("lof_score", 0.0),
                "lof_growth": row.get("lof_growth", 0.0)
            }
            opportunities.append(opp)
        except Exception as e:
            print(f"Skipping row {idx + 1}: {e}")
            continue

    cluster_embeddings = np.array(cluster_embeddings)
    viz_embeddings = np.array(viz_embeddings)

    if len(cluster_embeddings) == 0:
        raise ValueError("No valid embedding data found")

    cluster_embeddings = normalize(cluster_embeddings, norm='l2')

    return cluster_embeddings, viz_embeddings, opportunities


def run_kmeans_clustering(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, KMeans]:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
        max_iter=1000,
        algorithm='lloyd'
    )
    cluster_labels = kmeans.fit_predict(embeddings)

    return cluster_labels, kmeans


def visualize_clustering_results(viz_embeddings: np.ndarray, labels: np.ndarray, k: int, output_dir: str):
    fig = plt.figure(figsize=(12, 10))

    if viz_embeddings.shape[1] == 3:
        ax = fig.add_subplot(111, projection='3d')
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(
                viz_embeddings[mask, 0],
                viz_embeddings[mask, 1],
                viz_embeddings[mask, 2],
                label=f'Cluster {cluster_id + 1} ({np.sum(mask)})',
                s=30,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)
        ax.set_zlabel('UMAP Dimension 3', fontsize=14)
    else:
        ax = fig.add_subplot(111)
        viz_2d = viz_embeddings[:, :2]
        for cluster_id in range(k):
            mask = labels == cluster_id
            ax.scatter(
                viz_2d[mask, 0],
                viz_2d[mask, 1],
                label=f'Cluster {cluster_id + 1} ({np.sum(mask)})',
                s=30,
                alpha=0.8,
                edgecolors='white',
                linewidth=0.5
            )
        ax.set_xlabel('UMAP Dimension 1', fontsize=14)
        ax.set_ylabel('UMAP Dimension 2', fontsize=14)

    ax.set_title(f'Technical Opportunity KMeans Clustering Results (k={k})', fontsize=16, pad=20)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, f'kmeans_clustering_result_k{k}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()


def analyze_cluster_statistics(opportunities: List[Dict], labels: np.ndarray, k: int) -> List[Dict]:
    cluster_stats = []

    for cluster_id in range(k):
        cluster_opps = [opp for opp, label in zip(opportunities, labels) if label == cluster_id]
        cluster_size = len(cluster_opps)

        lof_scores = [opp.get('lof_score', 0) for opp in cluster_opps]
        avg_lof = np.mean(lof_scores) if lof_scores else 0

        opp_types = {}
        for opp in cluster_opps:
            opp_type = opp.get('opportunity_type', 'unknown')
            opp_types[opp_type] = opp_types.get(opp_type, 0) + 1

        sorted_opps = sorted(cluster_opps, key=lambda x: x.get('lof_score', 0), reverse=True)
        top_keywords = [opp['keyword'] for opp in sorted_opps[:10]]

        cluster_stats.append({
            "cluster_id": cluster_id + 1,
            "size": cluster_size,
            "avg_lof_score": round(avg_lof, 4),
            "opportunity_type_distribution": opp_types,
            "top_10_keywords": ','.join(top_keywords)
        })

    return cluster_stats


def save_clustering_results(
        opportunities: List[Dict],
        labels: np.ndarray,
        cluster_stats: List[Dict],
        kmeans_model: KMeans,
        output_dir: str,
        k: int
) -> None:
    labeled_opportunities = []
    labels_python = labels.tolist() if isinstance(labels, np.ndarray) else labels

    for opp, label in zip(opportunities, labels_python):
        labeled_opp = opp.copy()
        labeled_opp['cluster_id'] = label + 1
        labeled_opp = convert_numpy_types(labeled_opp)
        labeled_opportunities.append(labeled_opp)

    excel_path = os.path.join(output_dir, f'kmeans_clustering_results_k{k}.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        opp_data = []
        for opp in labeled_opportunities:
            opp_data.append({
                'Keyword': opp.get('keyword', ''),
                'Cluster_ID': opp.get('cluster_id', ''),
                'LOF_Score': round(opp.get('lof_score', 0), 4),
                'Opportunity_Type': opp.get('opportunity_type', ''),
                'LOF_Growth': round(opp.get('lof_growth', 0), 4)
            })

        df_opp = pd.DataFrame(opp_data)
        df_opp = df_opp.sort_values(by=['Cluster_ID', 'LOF_Score'], ascending=[True, False])
        df_opp.to_excel(writer, sheet_name='Opportunity_Details', index=False)

        cluster_data = []
        for stats in cluster_stats:
            type_dist = stats.get('opportunity_type_distribution', {})
            type_str = '; '.join([f'{t}:{cnt}' for t, cnt in type_dist.items()])

            cluster_data.append({
                'Cluster_ID': stats.get('cluster_id', ''),
                'Cluster_Size': stats.get('size', 0),
                'Avg_LOF_Score': stats.get('avg_lof_score', 0),
                'Opportunity_Type_Distribution': type_str,
                'Top_10_Keywords': stats.get('top_10_keywords', '')
            })

        df_cluster = pd.DataFrame(cluster_data)
        df_cluster = df_cluster.sort_values(by='Cluster_ID', ascending=True)
        df_cluster.to_excel(writer, sheet_name='Cluster_Statistics', index=False)

    labeled_path = os.path.join(output_dir, f'labeled_opportunities_k{k}.json')
    with open(labeled_path, 'w', encoding='utf-8') as f:
        json.dump(labeled_opportunities, f, indent=2, ensure_ascii=False)

    stats_clean = convert_numpy_types(cluster_stats)
    stats_path = os.path.join(output_dir, f'cluster_statistics_k{k}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_clean, f, indent=2, ensure_ascii=False)

    model_params = {
        "k": k,
        "cluster_centers": kmeans_model.cluster_centers_.tolist(),
        "inertia": float(kmeans_model.inertia_),
        "random_state": 42,
        "n_init": 20
    }
    params_path = os.path.join(output_dir, f'kmeans_model_params_k{k}.json')
    with open(params_path, 'w', encoding='utf-8') as f:
        json.dump(model_params, f, indent=2)

def main():
    K = 13
    EXCEL_PATH = "E:/lof/unique_keywords_opportunities.xlsx"
    OUTPUT_DIR = "E:/kmeans"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cluster_embeddings, viz_embeddings, opportunities = load_excel_data(EXCEL_PATH)

    cluster_labels, kmeans_model = run_kmeans_clustering(cluster_embeddings, K)

    visualize_clustering_results(viz_embeddings, cluster_labels, K, OUTPUT_DIR)

    cluster_stats = analyze_cluster_statistics(opportunities, cluster_labels, K)

    save_clustering_results(opportunities, cluster_labels, cluster_stats, kmeans_model, OUTPUT_DIR, K)


if __name__ == "__main__":
    main()