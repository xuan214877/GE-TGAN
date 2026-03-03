import json
import os
import ast
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict


class KeywordEmbedder:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "E:/preprocess/scibert/scibert_scivocab_uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print(f"Initialized model on {self.device}")

    def process_excel(self, input_path):
        try:
            df = pd.read_excel(input_path)

            keyword_records = defaultdict(list) 
            keyword_years = defaultdict(list)    
            keyword_freq = defaultdict(int)
            edges = defaultdict(list)
            degree_dict = defaultdict(int)
            total_keywords = 0

            for row_idx, row in df.iterrows():
                try:
                    keywords = ast.literal_eval(row['keywords'])
                    if isinstance(keywords, list):
                        year = int(row['year'])  
                        unique_kw = [kw.strip().lower() for kw in keywords if kw.strip()]
                        unique_kw = list(set(unique_kw))

                        for kw in unique_kw:
                            keyword_freq[kw] += 1
                            keyword_records[kw].append(row_idx)  
                            keyword_years[kw].append(year)      
                            total_keywords += 1

                        for i in range(len(unique_kw)):
                            for j in range(i + 1, len(unique_kw)):
                                src, dst = sorted([unique_kw[i], unique_kw[j]])
                                edges[(src, dst)].append(year)
                                degree_dict[src] += 1
                                degree_dict[dst] += 1

                except Exception as e:
                    print(f"fail: {row_idx + 1}: {str(e)}")
                    continue

            unique_keywords = list(keyword_records.keys())
            print(f" {total_keywords} keywords, {len(unique_keywords)} unique keywords")

            embeddings = self._generate_embeddings(unique_keywords)

            result = self._build_output_structure(
                keyword_records=keyword_records,
                keyword_years=keyword_years,  
                unique_keywords=unique_keywords,
                embeddings=embeddings,
                keyword_freq=keyword_freq,
                degree_dict=degree_dict,
                edges=edges,
                total_keywords=total_keywords
            )

            output_path = "E:/preprocess/data/keywords_textembeddings.json"
            self._save_json(result, output_path)
            return output_path

        except Exception as e:
            print(f"fail: {str(e)}")
            return None

    def _generate_embeddings(self, keywords, batch_size=16):
        embeddings = []
        for i in tqdm(range(0, len(keywords), batch_size), unit="batch"):
            batch = keywords[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend([e.tolist() for e in batch_embeddings])
        return embeddings

    def _build_output_structure(
            self,
            keyword_records,
            keyword_years,  
            unique_keywords,
            embeddings,
            keyword_freq,
            degree_dict,
            edges,
            total_keywords
    ):
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_keywords": total_keywords,
                "unique_keywords": len(unique_keywords),
                "embedding_dim": 768,
                "model_version": "scibert_scivocab_uncased"
            },
            "nodes": [
                {
                    "keyword": kw,
                    "text_embedding": emb,
                    "initial_feature": emb,
                    "frequency": keyword_freq[kw],
                    "degree": degree_dict[kw],
                    "occurrences": {
                        "count": len(keyword_records[kw]),
                        "original_rows": [idx + 1 for idx in keyword_records[kw]],
                        "first_occurrence_row": keyword_records[kw][0] + 1 if keyword_records[kw] else None,
                        "years": keyword_years[kw],  
                        "first_occurrence_year": min(keyword_years[kw]) if keyword_years[kw] else 2025  
                    }
                }
                for kw, emb in zip(unique_keywords, embeddings)
            ],
            "edges": [
                {
                    "source": src,
                    "target": dst,
                    "co_occurrence_years": list(set(years))
                }
                for (src, dst), years in edges.items()
            ]
        }

    def _save_json(self, data, output_path):
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"fail: {str(e)}")


if __name__ == "__main__":
    embedder = KeywordEmbedder()
    embedder.process_excel("E:/preprocess/data/WORDS.xlsx")