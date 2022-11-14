import sys

out = sys.argv[1]

f = open(out+"/results/testfile.txt", "x")
f.write("Hello, world!")
f.close()
