from ae import *
import sys

EP = sys.argv[1]

tr_data,test = make_data(size=1000)

for NUM in range(1):
    run_experiment(tr_data, test, EP=int(EP), NUM=NUM)