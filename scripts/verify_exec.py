import numpy as np
import os
import sys
import pandas as pd

"""
check if all files exist for all params
they are named as: test_method_p_N_n_U_g_d_s
i.e.: errors_amit_0.2p_100N_64n_64.0U_8192g_1.0d_27s.csv
"""

path = sys.argv[1]
# test = sys.argv[2]
# method = sys.argv[3]

root_dir = os.path.abspath(path)

test = "errors"
method = "amit"

N_list=[100, 200, 500]
s_list=[27]
p_list=[0.05, 0.1, 0.2]
n_list=[16, 32, 64, 128, 256, 512]
d_list=[0.01, 0.1, 0.5, 1]
U_list=[0, 3, 3.1, 3.9, 4.1, 4.9, 4, 5, 16, 32, 64, 128]
g_list=[0, 1, 2, 4, 16, 32, 64, 128, 256, 512, 1024, 8192]


todo = {"N": [], "s": [], "p": [], "n": [], "d": [], "U": [], "g": []}

for N in N_list:
    dir = root_dir+"".join(["/",str(N),"N"])
    for s in s_list:
        for p in p_list:
            for n in n_list:
                for d in d_list:
                    for U in U_list:
                        for g in g_list:
                            csv = "_".join([test, method, "".join([str(p),"p"]), "".join([str(N),"N"]), "".join([str(n),"n"]), "".join([str(U),"U"]), "".join([str(g),"g"]), "".join([str(d),"d"]), "".join([str(s),"s"])])+".csv"
                            if not os.path.exists(dir + "/" + csv):
                                # params = {"N": N, "s": s, "p": p, "n": n, "d": d, "U": U, "g": g}
                                # todo[num] = params
                                # num += 1
                                todo["N"].append(N)
                                todo["s"].append(s)
                                todo["p"].append(p)
                                todo["n"].append(n)
                                todo["d"].append(d)
                                todo["U"].append(U)
                                todo["g"].append(g)

fin = pd.DataFrame(todo, columns=["N", "s", "p", "n", "d", "U", "g"])

fin.to_csv(root_dir+"/fix_params.csv")





    

