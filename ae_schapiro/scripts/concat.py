import pandas as pd
import numpy as np
import sys
import os

EPs = sys.argv[1]

# path = './results/'
# if not os.path.exists(path):
#     os.makedirs(path)

res = []
for EP in EPs:
    dfs = [x for x in os.listdir('./results') if x.startswith('res_ep_'+EP)]
    for filename in dfs:
        df=pd.read_csv('./results/'+filename)
        res.append(df)
    fin = pd.concat(res)
    fin.to_csv('./stats/stats_'+EP+'.csv')