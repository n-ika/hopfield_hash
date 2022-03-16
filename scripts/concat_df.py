import pandas as pd
import os

path="./output/results/"
df_list=os.listdir(path)

data = pd.concat([pd.read_csv(path+df) for df in df_list if df.endswith(".csv")], sort=False, ignore_index=True)

data.to_csv(path+"data.csv", columns=["threshold", "N", "n", "errors", "p", "k", "type", "g", "U"], index=False)