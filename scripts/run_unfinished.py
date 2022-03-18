import sys
import pickle
from test_sys import *
import argparse
import timeit
import pandas as pd
import os

method = "amit"
test = "errors"

DF_PATH = sys.argv[1]
START = sys.argv[2]
STOP = sys.argv[3]

df_full = pd.read_csv(DF_PATH)

df = df_full.loc[int(START):int(STOP)]

for row in df.itertuples():
    start = timeit.default_timer()
    
    specs = "_".join([test, method, str(row.p)+"p", str(row.N)+"N", str(row.n)+"n", str(row.U)+"U", str(row.g)+"g", str(row.d)+"d", str(row.s)+"s"])

    filepath = './output/results/' + str(row.N) + "N/" + specs + '.csv'
    
    if not os.path.isfile(filepath):

        results_artif = test_memory(METHOD = method, TEST = test, mfccs_vectors=None, U = row.U, N = row.N, n = row.n, g = row.g, p = row.p, d = row.d, SEED = row.s)
        results_artif["g"] = row.g
        results_artif["U"] = row.U
        results_artif["d"] = row.d
        results_artif["n"] = row.n
        results_artif["seed"] = row.s
    
        results_artif.to_csv(filepath)

        stop = timeit.default_timer()
        print('Time: ', stop - start)
        print(specs, '\n DONE')
