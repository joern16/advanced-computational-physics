#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of pi using midpoint-rule quadrature.
...
License: MIT
"""

import time
import argparse
import numpy as np
from mpi4py import MPI


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=1000000, help="Number of points to sample at")
    parser.add_argument("-p", "--parallel", action="store_true", help="Use parallel computation")

    return parser.parse_args()


def integrant(x):
    """Integrant function for pi calculation."""
    return 4.0 * np.sqrt(1.0 - x**2)


def serial_integration(start, stop, N):
    """Perform serial midpoint-rule integration."""
    integral = 0.0
    for i in range(start, stop):
        integral += integrant((i + 0.5) / N) / N

    return integral


def parallel_integration(N):
    """Perform parallel midpoint-rule integration using MPI."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    block_size = N // size
    start = rank * block_size
    stop = start + block_size 

    local_integral = serial_integration(start, stop, N)

    total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)

    if rank == 0:
        return total_integral
    else:
        return None
    


if __name__ == "__main__":
    args = parse_arguments()
    N = args.N

    if not args.parallel:
        start_time = time.time()
        integral = serial_integration(0, N, N)
        end_time = time.time()
        print(f"Serial integration result: {integral}")
        print(f"Time taken: {end_time - start_time} seconds")

    else:
        start_time = time.time()
        integral = parallel_integration(N)
        end_time = time.time() 
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"Parallel integration result: {integral}")
            print(f"Time taken: {end_time - start_time} seconds")
