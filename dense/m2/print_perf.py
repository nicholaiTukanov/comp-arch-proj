import os
import subprocess
import statistics
import math

# matmul ops
def ops(m,n,k):
    return (2*m*n*k)

# computes ops/mem
# assume C+=AB
# naive implies 1 load/input and 1 store/output
# matmul bytes transferred = 2*sizeof(C) + sizeof(A) + sizeof(B) = 2mn + mk + kn
def get_naive_ai(m,n,k):
    return ops(m,n,k) / (2*m*n+m*k+k*n)

# runs a matmul problem (m,n,k) on some device dev
# dev is a container with 2 elements: name, peak_tp
# assume that executable will ONLY print execution time in milliseconds
def time_matmul(dev,m,n,k):
    cmd_str = f"./performance.x {m} {n} {k}"
    p = subprocess.Popen(list(cmd_str.split(" ")), stdout=subprocess.PIPE)
    try:
        t = float(str(p.stdout.readline().rstrip(),encoding="utf-8"))
        return ops(m,n,k) / (t)
    except:
        print(f"ERROR RUNNING EXECUTABLE. CHECK ./{dev}_performance.x")
        print(f"m = {m}, n = {n}, k = {k}")

# get performance for many problem sizes
def gather_performance(dev):

    # square
    Ps = [2**x for x in range(4, int(math.log2(4096)+1))]
    throughput = []
    for p in Ps:
        throughput.append(time_matmul(dev, p, p, 64))
        
    for tp in throughput:
        print(tp)

    # plot_performance(dev, type_, list(range(P_START, P_END+1, P_INC)), throughput)

gather_performance("gpu")

