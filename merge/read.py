import json
import pandas as pd


def read_correlation_results(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = []
        for pair_key, details in data.items():
            clusters = pair_key.replace("cluster_", "").split("-")
            cluster1, cluster2 = int(clusters[0]), int(clusters[1])
            results.append({
                "cluster1": cluster1,
                "cluster2": cluster2,
                "total_weight": details["total_weight"],
                "normalized_strength": details["normalized_strength"],
                "size_cluster1": details["cluster_sizes"][0],
                "size_cluster2": details["cluster_sizes"][1]
            })
        df = pd.DataFrame(results)
        return df

    except FileNotFoundError:
        print(f"Not found: {json_path}")
        return None
    except json.JSONDecodeError:
        print(f"Not valid: {json_path}")
        return None
    except Exception as e:
        print(f"{str(e)}")
        return None


def main():
    json_path = "E:/merge/cluster_correlation_strength.json"
    correlation_df = read_correlation_results(json_path)

    if correlation_df is not None:
        csv_path = json_path.replace(".json", ".csv")
        correlation_df.to_csv(csv_path, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
