import json
import os
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple

pd.set_option('display.unicode.ambiguous_as_wide', True)
pd.set_option('display.unicode.east_asian_width', True)


def load_data(opportunities_path: str) -> List[Dict]:
    with open(opportunities_path, 'r', encoding='utf-8') as f:
        opportunities = json.load(f)

    valid_opportunities = []
    for opp in opportunities:
        if 'final_embedding' not in opp:
            continue
        final_emb = np.array(opp['final_embedding'], dtype=np.float32)
        valid_opp = {
            'keyword': opp.get('keyword', ''),
            'final_embedding': final_emb,
            'opportunity_type': opp.get('opportunity_type', ''),
            'time_window': opp.get('time_window', ''),
            'original_data': opp
        }
        valid_opportunities.append(valid_opp)

    return valid_opportunities


def aggregate_keywords(opportunities: List[Dict]) -> List[Dict]:
    keyword_dict = defaultdict(lambda: {
        "embeddings": [],
        "opportunity_types": [],
        "time_windows": [],
        "original_data_list": []
    })

    for idx, opp in enumerate(opportunities):
        keyword = opp["keyword"]
        entry = keyword_dict[keyword]
        entry["embeddings"].append(opp["final_embedding"])
        entry["opportunity_types"].append(opp["opportunity_type"])
        entry["time_windows"].append(opp["time_window"])
        if not entry["original_data_list"]:
            entry["original_data_list"].append(opp["original_data"])
    unique_opportunities = []
    for keyword, data in keyword_dict.items():
        embeddings_matrix = np.array(data["embeddings"])
        avg_embedding = np.mean(embeddings_matrix, axis=0).tolist()
        main_opp_type = Counter(data["opportunity_types"]).most_common(1)[0][0]
        time_windows_str = "|".join(sorted(set([str(tw) for tw in data["time_windows"]])))
        first_data = data["original_data_list"][0] if data["original_data_list"] else {}
        unique_opp = {
            "Keyword": keyword,
            "Main_opp_type": main_opp_type,
            "Time_windows": time_windows_str,
            "Original_repeat_count": len(data["embeddings"]),
            "Embedding": str(avg_embedding),
            "LOF_Score": first_data.get('lof_score', 0.0),
            "LOF_Growth": first_data.get('lof_growth', 0.0),
            "Prev_Window": first_data.get('prev_window', 'None'),
            "Current_Window": first_data.get('time_window', ''),
            "Opportunity_Type": first_data.get('opportunity_type', '')
        }
        unique_opportunities.append(unique_opp)

    return unique_opportunities


def save_to_excel(unique_opportunities: List[Dict], output_path: str):
    df = pd.DataFrame(unique_opportunities)
    core_cols = ["Keyword", "Main_opp_type", "Time_windows", "Original_repeat_count"]
    other_cols = [col for col in df.columns if col not in core_cols]
    df = df[core_cols + other_cols]
    df.to_excel(output_path, index=False, engine='openpyxl')


def main():
    OPPORTUNITIES_PATH = "E:/lof/all_opportunities.json"
    OUTPUT_EXCEL_PATH = "E:/lof/unique_keywords_opportunities.xlsx"

    output_dir = os.path.dirname(OUTPUT_EXCEL_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    opportunities = load_data(OPPORTUNITIES_PATH)

    unique_opportunities = aggregate_keywords(opportunities)

    save_to_excel(unique_opportunities, OUTPUT_EXCEL_PATH)


if __name__ == "__main__":
    main()