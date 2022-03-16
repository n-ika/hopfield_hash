import sys
import pickle
from test_sys import *
import argparse
import timeit



if __name__ == '__main__':

     parser = argparse.ArgumentParser()

     # Model configuration.
     parser.add_argument('-v', '--vec', type=str, default='./input/vectors.txt', help='feature vectors of audio files used in the network')
     parser.add_argument('-mtd', '--method', type=str, default='default', help='method according to which we are building the network; possible methods are `default`, `amit`, `tsodyks`')
     parser.add_argument('-t', '--test', type=str, default='errors', help='test with which we are evaluating memory of the network; possible tests are `errors`, `PR`, `distance`')
     parser.add_argument('-p', '--p', type=float, default=0.1, help='sparsity degree: how many neurons are set to 1 (i.e. 10%)')
     parser.add_argument('-N', '--N', type=int, default=100, help='number of neurons (size of synaptic weight matrix is N*N)')
     parser.add_argument('-n', '--n', type=int, default=15, help='number of memories that are stored in the system')
     parser.add_argument('-u', '--U', type=float, default=0, help='threshold for determining neuron activation')
     parser.add_argument('-g', '--g', type=int, default=100, help='parameter for adjusting the synaptic matrix and threshold (Amit method)')
     parser.add_argument('-d', '--d', type=float, default=0.01, help='amount of noise (%) to degenerate the test vectors')
     parser.add_argument('-s', '--seed', type=int, default=27, help='random seed for generating data')

     args = parser.parse_args()
     print(args)

     with open(args.vec, "rb") as fp:
          mfccs_vectors = pickle.load(fp)

     start = timeit.default_timer()

     results_artif = test_memory(METHOD = args.method, TEST = args.test, mfccs_vectors=None, U = args.U, N = args.N, n = args.n, g = args.g, p = args.p, d = args.d, SEED = args.seed)
     results_artif["g"] = args.g
     results_artif["U"] = args.U
     results_artif["d"] = args.d
     results_artif["n"] = args.n
     results_artif["seed"] = args.seed
     specs = "_".join([args.test, args.method, str(args.p)+"p", str(args.N)+"N", str(args.n)+"n", str(args.U)+"U", str(args.g)+"g", str(args.d)+"d", str(args.seed)+"s"])
     results_artif.to_csv('./output/results/' + specs + '.csv')

     stop = timeit.default_timer()
     print('Time: ', stop - start)
     print('DONE')




# import sys
# import pickle
# from test_sys import *
# import argparse
# import timeit



# if __name__ == '__main__':
     
#      parser = argparse.ArgumentParser()

#      # Model configuration.
#      parser.add_argument('-v', '--vec', type=str, default='./input/vectors.txt', help='feature vectors of audio files used in the network')
#      parser.add_argument('-mtd', '--method', type=str, default='default', help='method according to which we are building the network; possible methods are `default`, `amits`, `tsodyks`')
#      parser.add_argument('-t', '--test', type=str, default='errors', help='test with which we are evaluating memory of the network; possible tests are `errors`, `PR`, `distance`')
#      parser.add_argument('-p', '--p', type=float, default=0.1, help='sparsity degree: how many neurons are set to 1 (i.e. 10%)')
#      parser.add_argument('-N', '--N', type=int, default=100, help='number of neurons (size of synaptic weight matrix is N*N)')
#      parser.add_argument('-n', '--n', type=int, default=15, help='number of memories that are stored in the system')
#      parser.add_argument('-u', '--U', type=float, default=0, help='threshold for determining neuron activation')
#      parser.add_argument('-g', '--g', type=int, default=100, help='parameter for adjusting the synaptic matrix and threshold (Amit method)')

#      args = parser.parse_args()
#      print(args)
     
#      with open(args.vec, "rb") as fp:
#           mfccs_vectors = pickle.load(fp)

#      start = timeit.default_timer()

#      results_artif = test_memory(METHOD = args.method, TEST = args.test, U = args.U, N = args.N, n = args.n, g = args.g, p = args.p)
#      results_artif["g"] = args.g
#      results_artif["U"] = args.U
#      results_artif["n"] = args.n
#      specs = "_".join([args.test, args.method, str(args.N)+"N", str(args.n)+"n", str(args.U)+"U", str(args.g)+"g"])
#      results_artif.to_csv('./output/results/' + specs + '.csv')

#      stop = timeit.default_timer()
#      print('Time: ', stop - start)  

#      # results_hash = test_memory(METHOD = MTD, TEST = TEST, mfccs_vectors=mfccs_vectors, U = U, N = N, g = g, p_list = p_list)
# # n_list = [1, 2, int(N*0.05), int(N*0.1), int(N*0.15), int(N*0.5), N, N*2,  N*10]