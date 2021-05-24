import sys
import pickle
from test_sys import *
import timeit

VEC = sys.argv[1]
MTD = sys.argv[2] # default, amits, tsodyks
TEST = sys.argv[3] # PR, errors, distance

with open(VEC, "rb") as fp:
     mfccs_vectors = pickle.load(fp)

U = 0
N = 1000
g_list = [100]
p_list = [0.1]
n_list = [100]

start = timeit.default_timer()

for g in g_list:

     # results_hash = test_memory(METHOD = MTD, TEST = TEST, mfccs_vectors=mfccs_vectors, U = U, N = N, g = g, p_list = p_list, n_list = n_list)

     results_artif = test_memory(METHOD = MTD, TEST = TEST, U = U, N = N, g = g, p_list = p_list, n_list = n_list)

     # results_hash.to_csv('../results/'+ TEST + '/' + MTD + '/hash_g_' + str(g) + '.csv')
     results_artif.to_csv('../results/'+TEST + '/' + MTD +'/artif_g_' + str(g) + '.csv')

stop = timeit.default_timer()

print('Time: ', stop - start)  

