import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import normalize
from umap import UMAP
from typing import List, Dict, Tuple
from tqdm import tqdm
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
plt.rcParams["axes.unicode_minus"] = False

DPI = 300
FIGSIZE_3D = (10, 8)
FIGSIZE_2D = (10, 6)

def convert_numpy_types(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.float32) or isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, np.int32) or isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    else:
        return data

def load_filtered_excel_embeddings(excel_path: str) -> Tuple[np.ndarray, List[Dict]]:
    df = pd.read_excel(excel_path, engine='openpyxl')
    required_col = "Cluster_embedding"
    if required_col not in df.columns:
        raise ValueError(f"missing: {required_col}")

    def str_to_emb(emb_str: str) -> np.ndarray:
        try:
            emb_str = emb_str.strip().replace('\n', '').replace(' ', '').replace("'", '"')
            return np.array(json.loads(emb_str), dtype=np.float32)
        except Exception as e:
            raise ValueError(f"{e}\n:{emb_str}")

    embeddings = []
    valid_opportunities = []
    for idx, row in df.iterrows():
        try:
            emb = str_to_emb(row[required_col])
            embeddings.append(emb)
            opp = {
                "keyword": row.get("Keyword", f"Keyword_{idx}"),
                "final_embedding": emb.tolist(),
                "opportunity_type": row.get("Main_opp_type", ""),
                "time_windows": row.get("Time_windows", ""),
                "original_repeat_count": row.get("Original_repeat_count", 1)
            }
            valid_opportunities.append(opp)
        except Exception as e:
            print(f"skip {idx + 1}: {e}")
            continue
    normalized_embeddings = normalize(embeddings, norm='l2')
    return normalized_embeddings, valid_opportunities


def umap_dim_reduction(embeddings: np.ndarray,
                       cluster_components: int = 20,
                       viz_components: int = 3,
                       n_neighbors: int = 20,
                       min_dist: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    umap_cluster = UMAP(
        n_components=cluster_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        n_jobs=-1
    )
    cluster_embeddings = umap_cluster.fit_transform(embeddings)

    umap_viz = UMAP(
        n_components=viz_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric='cosine',
        random_state=42,
        n_jobs=-1
    )
    viz_embeddings = umap_viz.fit_transform(embeddings)
    return cluster_embeddings, viz_embeddings


def grid_search_umap(embeddings: np.ndarray) -> Dict:
    n_neighbors_list = [15, 20, 25, 30]
    min_dist_list = [0.01, 0.05, 0.1]
    best_score = 0
    best_params = None
    best_embeddings = None
    for n_neighbors in tqdm(n_neighbors_list, desc="n_neighbors"):
        for min_dist in min_dist_list:
            cluster_emb, _ = umap_dim_reduction(
                embeddings,
                n_neighbors=n_neighbors,
                min_dist=min_dist
            )
            avg_sil_score = 0
            valid_k = 0
            for k in range(5, 16):
                if k >= len(cluster_emb) or len(
                        set(KMeans(n_clusters=k, random_state=42).fit_predict(cluster_emb))) < 2:
                    continue
                labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(cluster_emb)
                score = silhouette_score(cluster_emb, labels)
                avg_sil_score += score
                valid_k += 1
            if valid_k > 0:
                avg_sil_score /= valid_k
                if avg_sil_score > best_score:
                    best_score = avg_sil_score
                    best_params = {"n_neighbors": n_neighbors, "min_dist": min_dist}
                    best_embeddings = cluster_emb

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_embeddings": best_embeddings
    }


def calculate_clustering_metrics(embeddings: np.ndarray, max_k: int = 20) -> Dict:
    k_range = range(2, max_k + 1)
    metrics = {
        "k_range": [],
        "silhouette": [],
        "calinski_harabasz": [],
        "davies_bouldin": []
    }
    for k in tqdm(k_range):
        if k >= len(embeddings):
            break
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
            max_iter=1000
        )
        labels = kmeans.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue

        metrics["k_range"].append(int(k))
        metrics["silhouette"].append(float(silhouette_score(embeddings, labels)))
        metrics["calinski_harabasz"].append(float(calinski_harabasz_score(embeddings, labels)))
        metrics["davies_bouldin"].append(float(davies_bouldin_score(embeddings, labels)))

    return metrics

def save_individual_plots(viz_embeddings: np.ndarray, metrics: Dict, output_dir: str):
    fig1 = plt.figure(figsize=FIGSIZE_3D)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(viz_embeddings[:, 0], viz_embeddings[:, 1], viz_embeddings[:, 2],
                s=20, alpha=0.8, c='#1f77b4', edgecolors='#0a3d62')
    ax1.set_xlabel('UMAP dimension1', fontsize=12)
    ax1.set_ylabel('UMAP dimension2', fontsize=12)
    ax1.set_zlabel('UMAP dimension3', fontsize=12)
    ax1.set_title('UMAP 3D', fontsize=14, pad=20)
    ax1.grid(alpha=0.3)

    umap_3d_path = os.path.join(output_dir, '01_umap_3d_distribution.png')
    plt.savefig(umap_3d_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig1)

    fig2 = plt.figure(figsize=FIGSIZE_2D)
    ax2 = fig2.add_subplot(111)
    ax2.plot(metrics["k_range"], metrics["silhouette"], 'ro-', linewidth=2, markersize=6, label='Silhouette')
    if metrics["silhouette"]:
        best_k_idx = np.argmax(metrics["silhouette"])
        best_k = metrics["k_range"][best_k_idx]
        best_score = metrics["silhouette"][best_k_idx]
        ax2.scatter(best_k, best_score, color='darkred', s=100, zorder=5,
                    label=f'best_k={best_k} (score={best_score:.4f})')
    ax2.set_xlabel('k', fontsize=12)
    ax2.set_ylabel('silhouette_score', fontsize=12)
    ax2.set_title('silhouette_curve', fontsize=14, pad=20)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='best', fontsize=10)
    silhouette_path = os.path.join(output_dir, '02_silhouette_score.png')
    plt.savefig(silhouette_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig2)


    fig3 = plt.figure(figsize=FIGSIZE_2D)
    ax3 = fig3.add_subplot(111)
    ax3.plot(metrics["k_range"], metrics["calinski_harabasz"], 'bo-', linewidth=2, markersize=6, label='CH')
    if metrics["calinski_harabasz"]:
        best_k_idx = np.argmax(metrics["calinski_harabasz"])
        best_k = metrics["k_range"][best_k_idx]
        best_score = metrics["calinski_harabasz"][best_k_idx]
        ax3.scatter(best_k, best_score, color='darkblue', s=100, zorder=5,
                    label=f'k={best_k} (score={best_score:.2f})')
    ax3.set_xlabel('k', fontsize=12)
    ax3.set_ylabel('Calinski-Harabasz', fontsize=12)
    ax3.set_title('CH_curve', fontsize=14, pad=20)
    ax3.grid(alpha=0.3)
    ax3.legend(loc='best', fontsize=10)
    ch_path = os.path.join(output_dir, '03_calinski_harabasz_score.png')
    plt.savefig(ch_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig3)


    fig4 = plt.figure(figsize=FIGSIZE_2D)
    ax4 = fig4.add_subplot(111)
    ax4.plot(metrics["k_range"], metrics["davies_bouldin"], 'go-', linewidth=2, markersize=6, label='DB')
    if metrics["davies_bouldin"]:
        best_k_idx = np.argmin(metrics["davies_bouldin"])
        best_k = metrics["k_range"][best_k_idx]
        best_score = metrics["davies_bouldin"][best_k_idx]
        ax4.scatter(best_k, best_score, color='darkgreen', s=100, zorder=5,
                    label=f'k={best_k} (score={best_score:.4f})')
    ax4.set_xlabel('k', fontsize=12)
    ax4.set_ylabel('Davies-Bouldin', fontsize=12)
    ax4.set_title('DB_curve', fontsize=14, pad=20)
    ax4.grid(alpha=0.3)
    ax4.legend(loc='best', fontsize=10)

    db_path = os.path.join(output_dir, '04_davies_bouldin_score.png')
    plt.savefig(db_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig4)

def main():
    FILTERED_EXCEL_PATH = "E:/lof/unique_keywords_opportunities.xlsx"
    OUTPUT_DIR = "E:/kmeans"
    MAX_K = 50
    UMAP_CLUSTER_COMPONENTS = 20
    UMAP_VIZ_COMPONENTS = 3
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    normalized_embeddings, _ = load_filtered_excel_embeddings(FILTERED_EXCEL_PATH)

    umap_search_result = grid_search_umap(normalized_embeddings)
    best_umap_params = umap_search_result["best_params"]
    best_cluster_embeddings = umap_search_result["best_embeddings"]

    _, viz_embeddings = umap_dim_reduction(
        normalized_embeddings,
        cluster_components=UMAP_CLUSTER_COMPONENTS,
        viz_components=UMAP_VIZ_COMPONENTS,
        **best_umap_params
    )

    np.save(os.path.join(OUTPUT_DIR, 'umap_cluster_embeddings_best_filtered.npy'), best_cluster_embeddings)
    np.save(os.path.join(OUTPUT_DIR, 'umap_viz_embeddings_best_filtered.npy'), viz_embeddings)

    clustering_metrics = calculate_clustering_metrics(best_cluster_embeddings, MAX_K)

    save_individual_plots(viz_embeddings, clustering_metrics, OUTPUT_DIR)

    final_results = {
        "umap_best_params": best_umap_params,
        "umap_best_score": float(umap_search_result["best_score"]),
        "clustering_metrics": clustering_metrics,
        "params": {
            "max_k": int(MAX_K),
            "umap_cluster_components": int(UMAP_CLUSTER_COMPONENTS),
            "umap_viz_components": int(UMAP_VIZ_COMPONENTS),
            "data_source": FILTERED_EXCEL_PATH
        }
    }

    final_results_clean = convert_numpy_types(final_results)

    with open(os.path.join(OUTPUT_DIR, "clustering_results_filtered.json"), 'w', encoding='utf-8') as f:
        json.dump(final_results_clean, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()