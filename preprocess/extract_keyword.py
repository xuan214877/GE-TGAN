import pandas as pd
import numpy as np
from tqdm import tqdm
import yake
import os
import re


def clean_keywords(keyword_str):
    if pd.isna(keyword_str):
        return np.nan
    cleaned = re.sub(r'<[^>]+>', '', str(keyword_str))
    keywords = [k.strip() for k in re.split(r'[,;]', cleaned) if k.strip()]
    return str(keywords) if keywords else np.nan


def process_single_file(input_file, output_suffix='_processed'):
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=2,
        dedupLim=0.9,
        dedupFunc='seqm',
        windowsSize=1,
        top=5,
        features=None
    )
    if not input_file.lower().endswith(('.xlsx', '.xls')) or os.path.basename(input_file).startswith('~$'):
        print(f"escape: {input_file}")
        return

    try:
        with pd.ExcelFile(input_file, engine='openpyxl' if input_file.endswith('.xlsx') else None) as excel:
            df = pd.read_excel(excel, dtype={
                'Author keywords': 'string',
                'Index keywords': 'string',
                'WordNetLemmatizer': 'string'
            })

        required_columns = ['Author keywords', 'Index keywords', 'WordNetLemmatizer']
        if not set(required_columns).issubset(df.columns):
            missing = set(required_columns) - set(df.columns)
            print(f"\nskip {os.path.basename(input_file)}，missing: {missing}")
            return

        df['Author keywords'] = df['Author keywords'].apply(clean_keywords)
        df['Index keywords'] = df['Index keywords'].apply(clean_keywords)

        has_author = df['Author keywords'].notna()
        has_index = df['Index keywords'].notna() & ~has_author

        df['Final_Keywords'] = np.nan

        df.loc[has_author, 'Final_Keywords'] = df.loc[has_author, 'Author keywords']

        df.loc[has_index, 'Final_Keywords'] = df.loc[has_index, 'Index keywords']

        need_yake = df['Final_Keywords'].isna()
        yake_texts = df.loc[need_yake, 'WordNetLemmatizer'].dropna()

        yake_results = {}
        for idx, text in tqdm(yake_texts.items()):
            try:
                keywords = [kw[0] for kw in kw_extractor.extract_keywords(text)]
                yake_results[idx] = str(keywords) if keywords else np.nan
            except Exception as e:
                print(f"\nYAKE fail row {idx}: {str(e)}")
                yake_results[idx] = np.nan

        df.loc[need_yake, 'Final_Keywords'] = pd.Series(yake_results)


        df['Final_Keywords'] = df['Final_Keywords'].apply(
            lambda x: str(eval(x)) if isinstance(x, str) and not x.startswith('[') else x
        )


        output_path = r"E:/preprocess/data/WORDS.xlsx"
        df.to_excel(output_path, index=False, engine='openpyxl')

    except Exception as e:
        print(f"\nfail {os.path.basename(input_file)}")
        print(f"{type(e).__name__}")
        print(f"{str(e)}")
        if 'df' in locals():
            print(f"{df.head(1)}")


if __name__ == "__main__":
    input_file = r"E:/preprocess/data/title_abstract_combined_lemmatized.xlsx"

    if not os.path.exists(input_file):
        print(f"Not exists: {input_file}")
    else:
        process_single_file(input_file)