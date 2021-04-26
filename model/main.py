import sys
import pickle
from test_sys import *
import timeit

VEC = sys.argv[1]
MTD = sys.argv[2]

with open(VEC, "rb") as fp:
     mfccs_vectors = pickle.load(fp)



start = timeit.default_timer()

results_PR_hash = test_memory(METHOD = MTD, TEST = "PR", mfccs_vectors=mfccs_vectors, U = 0, N = 100, g = 100, p_list = [0.01, 0.1, 0.5], n_list = [5, 10, 15])

results_PR_artif = test_memory(METHOD = MTD, TEST = "PR", U = 0, N = 100, g = 100, p_list = [0.01, 0.1, 0.5], n_list = [5, 10, 15])

stop = timeit.default_timer()

print('Time: ', stop - start)  

# print(results_PR_hash)
# print(results_PR_artif)