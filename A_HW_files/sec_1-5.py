import time, timeit
import numpy as np

# Problem 1.26
def ABx(k, time_fn):
    """Compute ABx as (AB)x and A(Bx) and return the times for each.

    Parameter:
        k (int): A and B are matrices of size 2**k x 2**k and x has length k
    """

    A = np.random.rand(2**k, 2**k)
    B = np.random.rand(2**k, 2**k)
    x = np.random.rand(2**k, 1)

    start = time_fn()
    AB_x = (A@B)@x
    end = time_fn()
    t1 = end - start

    start = time_fn()
    A_Bx = A@(B@x)
    end = time_fn()
    t2 = end - start

    return (t1, t2)

def run_ABx():
    times = list()

    for k in range(1, 11 + 1):
        times.append(ABx(k, time.perf_counter_ns))
    
    for i, pair in enumerate(times):
        try:
            ratio = times[i][0]/times[i][1]
        except ZeroDivisionError:
            ratio = "n/a"

        print(f"k = {i+1}"
                f"\t(AB)x: {times[i][0]}\n"
                f"\tA(Bx): {times[i][1]}\n"
                f"  (AB)x/A(Bx): {ratio}\n")

# Problem 1.27
def fancy(n, time_fn):
    """Compute two expressions and return the times for each.
    
    Parameters:
        n (int): I is 2**n x 2**n and u, v, x have length k
    """

    In = np.identity(2**n)
    u = np.random.rand(2**n, 1)
    v = np.random.rand(2**n, 1)
    x = np.random.rand(2**n, 1)

    start = time_fn()
    one = (In + u@v.T)@x
    end = time_fn()
    t1 = end - start

    start = time_fn()
    two = x + u@(v.T@x)
    end = time_fn()
    t2 = end - start

    return (t1, t2)

def run_fancy():
    times = list()

    for n in range(1, 11 + 1):
        times.append(fancy(n, time.perf_counter_ns))
    
    for i, pair in enumerate(times):
        try:
            ratio = times[i][0]/times[i][1]
        except ZeroDivisionError:
            ratio = "n/a"

        print(f"n = {i+1}"
                f"\tone: {times[i][0]}\n"
                f"\ttwo: {times[i][1]}\n"
                f"    one/two: {ratio}\n")