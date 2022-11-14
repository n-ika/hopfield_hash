import pandas as pd
import os
import sys

path = sys.argv[1]

for dir in os.listdir(path):
    if os.path.isdir(path+dir):
        df_list = os.listdir(path+dir)
        data = pd.concat([pd.read_csv(os.path.join(path, dir, df)) for df in df_list if df.endswith(".csv")], sort=False, ignore_index=True)
        
        data.to_csv(path+"data_"+dir+".csv", index=False)
    print("completed: ",dir)

print("DONE")
