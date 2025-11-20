import pandas as pd
import os, pathlib

data_path = os.path.join(pathlib.Path(__file__).parent.parent, "data")

df1 = pd.read_excel(os.path.join(data_path, "data_taobao_processed.xlsx"))
print(df1.head())
df2 = pd.read_excel(os.path.join(data_path, "data_combined.xlsx"))
print(df2.head())

# df = pd.concat([df1, df2], axis=0, ignore_index=True)

# df.to_csv(os.path.join(data_path, "data_merge.csv"))
