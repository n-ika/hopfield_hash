import sys
import pickle
from test_sys import *
import timeit

VEC = sys.argv[1]
MTD = sys.argv[2] # default, amits, tsodyks
TEST = sys.argv[3] # PR, errors, distance

with open(VEC, "rb") as fp:
     mfccs_vectors = pickle.load(fp)

# U_list = [-1, 0, 100]
# N_list = [100, 500, 1000]
# g_list = [1, 10, 100]
# p_list = [0.01, 0.1, 0.5]
# # n_list = [1000]
U_list = [-1, 0, 100]
N_list = [100, 200]
g_list = [1, 10, 100]
p_list = [0.01, 0.1, 0.5]

results = pd.DataFrame()

for U in U_list:
     for N in N_list:
          for g in g_list:

               start = timeit.default_timer()
               # results_hash = test_memory(METHOD = MTD, TEST = TEST, mfccs_vectors=mfccs_vectors, U = U, N = N, g = g, p_list = p_list)

               results_artif = test_memory(METHOD = MTD, TEST = TEST, U = U, N = N, g = g, p_list = p_list)

               results_artif["g"] = g
               results_artif["U"] = U

               specs = "_".join(map(str, [TEST, MTD, U, N, g]))
               # results_artif.to_csv('../results/' + specs + '.csv')
               results_artif.to_csv('../results/_' + specs + '.csv')

               results = pd.concat([results,results_artif], ignore_index=True)
               
               stop = timeit.default_timer()
               print('Time: ', stop - start)  


print('Time: ', stop - start)  
results.to_csv('../results/_all_params.csv')

