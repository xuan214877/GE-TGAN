import json
import os
from typing import Dict, List, Tuple


def load_new_embeddings(emb_path: str) -> Dict[str, List[float]]:
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"Not exists: {emb_path}")
    with open(emb_path, 'r', encoding='utf-8') as f:
        emb_data = json.load(f)
    emb_dict = {
        item['keyword']: [float(x) for x in item['embedding']]
        for item in emb_data
    }
    return emb_dict


def load_time_info_nodes(time_node_path: str) -> Dict[str, List[int]]:
    if not os.path.exists(time_node_path):
        raise FileNotFoundError(f"Not exists: {time_node_path}")
    with open(time_node_path, 'r', encoding='utf-8') as f:
        node_data = json.load(f)
    time_info_dict = {
        node['keyword']: node.get('co_occurrence_years', [])
        for node in node_data
    }
    print(f"total {len(time_info_dict)} nodes")
    return time_info_dict


def load_orbit_features(orbit_path: str) -> Tuple[Dict[str, List[float]], int]:
    if not os.path.exists(orbit_path):
        raise FileNotFoundError(f"Not exists: {orbit_path}")
    with open(orbit_path, 'r', encoding='utf-8') as f:
        orbit_data = json.load(f)
    orbit_dict = {
        item['keyword']: [float(x) for x in item['normalized_orbit_feature']]
        for item in orbit_data['nodes']
    }
    total_dim = orbit_data.get('total_feature_dim', 0)
    return orbit_dict, total_dim


def concatenate_embeddings_with_time_info(
        new_emb_dict: Dict[str, List[float]],
        time_info_dict: Dict[str, List[int]],
        orbit_dict: Dict[str, List[float]],
        save_path: str
) -> None:
    common_keywords = set(new_emb_dict.keys()) & set(time_info_dict.keys()) & set(orbit_dict.keys())
    concatenated = []
    for keyword in common_keywords:
        original_emb = new_emb_dict[keyword]
        orbit_feat = orbit_dict[keyword]
        co_occurrence_years = time_info_dict[keyword]

        valid_years = [y for y in co_occurrence_years if isinstance(y, int)]

        final_embedding = original_emb + orbit_feat

        concatenated.append({
            "keyword": keyword,
            "original_embedding": original_emb,
            "orbit_feature": orbit_feat,
            "final_embedding": final_embedding,
            "time_info": {
                "co_occurrence_years": valid_years
            },
            "embedding_dim": len(original_emb),
            "orbit_dim": len(orbit_feat),
            "total_dim": len(final_embedding)
        })

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(concatenated, f, indent=2, ensure_ascii=False)


def main():
    NEW_EMB_PATH = "E:/model/checkpoint.json"
    TIME_NODE_PATH = "E:/meta_graph/filtered_nodes.json"
    ORBIT_FEATURE_PATH = "E:/meta_graph/normalized_orbit_features.json"
    SAVE_PATH = "E:/meta_graph/concatenated_embeddings1.json"

    new_emb_dict = load_new_embeddings(NEW_EMB_PATH)
    time_info_dict = load_time_info_nodes(TIME_NODE_PATH)
    orbit_dict, orbit_dim = load_orbit_features(ORBIT_FEATURE_PATH)

    concatenate_embeddings_with_time_info(new_emb_dict, time_info_dict, orbit_dict, SAVE_PATH)


if __name__ == "__main__":
    main()