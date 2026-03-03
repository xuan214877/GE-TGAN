import pandas as pd

input_path = "E:/preprocess/data/rawdata.xlsx"
df = pd.read_excel(input_path, sheet_name=1)

df["Merged"] = df["Title"].str.cat(df["Abstract"], sep="\n")

df_new = df.copy()

output_path = "E:/preprocess/data/title_abstract_combined.xlsx"
df_new.to_excel(output_path, index=False)

