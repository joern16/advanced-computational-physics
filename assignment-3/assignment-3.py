#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of the Poisson and Laplace equations on grids.
...
License: MIT
"""

import numpy as np
from mpi4py import MPI


def optimal_omega(N):
    """
    Returns the optimal parameter for over-relaxation for an N x N square grid.
    """
    return 2.0 / (1.0 + np.sin(np.pi / N))

def solve_poisson_over_relaxation(phi, f, h, omega, tol=1e-5, max_iter=100000):
    """
    Solve the Poisson equation using over-relaxation.
    """
    N_y, N_x = phi.shape
    h = 1.0 / (N_y - 1)
    
    for iterations in range(max_iter):
        max_diff = 0.0
        
        # Lexicographical Gauss-Seidel update
        for i in range(1, N_y - 1):
            for j in range(1, N_x - 1):
                phi_old = phi[i, j]
                
                phi[i,j] = omega * (h**2 * f[i,j] + 0.25*(phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1])) + (1.0 - omega) * phi_old

                diff = abs(phi[i, j] - phi_old)
                if diff > max_diff:
                    max_diff = diff
                    
        if max_diff < tol:
            return phi, iterations + 1
            
    return phi, max_iter
    

def random_walk(i_start, j_start, N):
    """
    Perform a random walk on a square grid.
    """
    visits = np.zeros((N, N), dtype=np.int64)
    i, j = i_start, j_start
    while True:
        visits[i, j] += 1
        r = np.random.rand()
        if r < 0.25:
            i += 1
        elif r < 0.5:
            i -= 1
        elif r < 0.75:
            j += 1
        else:
            j -= 1
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            break
    return visits

def greens_function_parallel(i_start, j_start, N, N_walkers_per_core):
    """
    Evaluate the greens function at (i, j) using multiple random walks in parallel.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    visits = np.zeros((N, N), dtype=np.int64)

    for _ in range(N_walkers_per_core):
        visits += random_walk(i_start, j_start, N)

    total_visits = comm.Reduce(visits, op=MPI.SUM, root=0)

    if rank == 0:
        probability = total_visits / (N_walkers_per_core * comm.Get_size())
        return probability

    return None    
  
def solve_poisson_greens(phi, f, N, i, j, greens_ij, N_walkers_per_core):
    """
    Evaluate phi at (i, j) using a computed greens function.
    """
    h = 1.0 / (N - 1)
    phi[1:-1, 1:-1] = 0.0 # makes sure that phi only containes the boundary values
    
    phi_ij = 0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            phi_ij += greens_ij[i, j] * (h**2 * f[i, j] + phi[i, j])

    return phi_ij


if __name__ == "__main__":
    N = 100
    h = 1.0 / (N - 1)
    
   
