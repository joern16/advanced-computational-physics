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
import matplotlib.pyplot as plt



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, default=1000000, help="Number of points to sample at")
    parser.add_argument("-p", "--parallel", action="store_true", help="Use parallel computation")
    parser.add_argument("-c", "--compare", action="store_true", help="Compare serial and parallel results for different N values")
    parser.add_argument("-vc", "--visualize", action="store_true", help="Visualize the comparison results")

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
    stop = start + block_size if rank != size - 1 else N

    local_integral = serial_integration(start, stop, N)

    total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)

    if rank == 0:
        return total_integral
    else:
        return None
    


if __name__ == "__main__":
    args = parse_arguments()
    N = args.N

    # Compare serial and parallel implementations
    if args.compare or args.visualize:
        N_values = [10**3, 10**4, 10**5, 10**6, 10**7]

        serial_results = []
        parallel_results = []

        time_results_serial = []
        time_results_parallel = []

        for N in N_values:
            start_time = time.time()
            serial_results.append(serial_integration(0, N, N))
            time_results_serial.append(time.time() - start_time)

            start_time = time.time()
            parallel_results.append(parallel_integration(N))
            time_results_parallel.append(time.time() - start_time)

            # Output results or visualize
            if MPI.COMM_WORLD.Get_rank() == 0:
                if args.visualize:
                    # Visualization of time
                    plt.figure(figsize=(10, 6))
                    plt.loglog(N_values, time_results_serial, marker='x', label='Serial', color='blue')
                    plt.loglog(N_values, time_results_parallel, marker='x', label='Parallel', color='red')
                    plt.xlabel('Number of Points (N)')
                    plt.ylabel('Computational Time (seconds)')
                    plt.title('Comparison of Serial and Parallel Integration Times')
                    plt.legend()
                    plt.show()

                    # Visualization of results
                    plt.figure(figsize=(10, 6)) 
                    plt.semilogx(N_values, serial_results, marker='o', label='Serial', color='blue')
                    plt.semilogx(N_values, parallel_results, marker='o', label='Parallel', color='red')
                    plt.axhline(y=np.pi, color='green', linestyle='--', label='Actual π Value')
                    plt.xlabel('Number of Points (N)')
                    plt.ylabel('Integration Result')
                    plt.title('Comparison of Serial and Parallel Integration Results')
                    plt.legend()
                    plt.show()
                else:
                    for i in range(len(N_values)):
                        print(f"N={N_values[i]}: Serial Time={time_results_serial[i]:.6f}s, Parallel Time={time_results_parallel[i]:.6f}s")


    # Single run: either serial or parallel
    else:
        if not args.parallel:
            if MPI.COMM_WORLD.Get_rank() == 0:
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
                print(f"Parallel integration result for N={N}: {integral}")
                print(f"Time taken: {end_time - start_time} seconds")
