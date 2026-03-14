#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of the Poisson and Laplace equations on grids.
...
License: MIT
"""

import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt



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
    visits_sq = np.zeros((N, N), dtype=np.int64)

    for _ in range(N_walkers_per_core):
        visits = random_walk(i_start, j_start, N)
        visits += visits
        visits_sq += visits**2

    total_visits = comm.Reduce(visits, op=MPI.SUM, root=0)
    total_visits_sq = comm.Reduce(visits_sq, op=MPI.SUM, root=0)

    if rank == 0:
        greens_ij = total_visits / (N_walkers_per_core * comm.Get_size())
        variance = (total_visits_sq - total_visits**2 / (N_walkers_per_core * comm.Get_size())) / (N_walkers_per_core * comm.Get_size() - 1)
        return greens_ij, np.sqrt(np.sum(variance))

    return None, None    
  
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


def plot_greens_function(greens_ij, title="Green's Function", filename=None):
    """
    Plots the 2D Green's function as a heatmap.
    """
    plt.figure(figsize=(8, 6))
    
    # Assuming extent is from 0 to 1 meters as per assignment
    plt.imshow(greens_ij, origin='lower', cmap='viridis', extent=[0, 1, 0, 1])
    plt.colorbar(label="Green's Function Value")
    plt.title(title)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    if filename:
        plt.savefig(filename)
    plt.show()


def result_wrapper(x, y, phi, f, N_walkers_per_core, N, name="test"):
    """
    Obtains results for the poisson equation.
    """
    h = 1.0 / (N - 1)
    phi_sol =  solve_poisson_over_relaxation(phi, f, h, optimal_omega(N))[0]

    print("\n" + "="*60)
    print("                 Restuls: " + name)
    print("="*60)

    for _x, _y in x, y:
        i, j = _x // h, _y // h
        greens_ij, std_deviation = greens_function_parallel(i, j, N, N_walkers_per_core)
        phi_ij = solve_poisson_greens(phi, f, N, i, j, greens_ij, N_walkers_per_core)
        phi_ij_sol = phi_sol[i, j]

        print(f"Random walk phi({_x}, {_y})    : {phi_ij:.8f}")
        print(f"Standard deviation {_x}, {_y}  : {std_deviation:.8f}")
        print(f"Gauss-Seidel phi({_x}, {_y})   : {phi_ij_sol:.8f}" + "\n")

        plot_greens_function(greens_ij, title= name + f": greens function at ({_x}, {_y})", filename=f"greens_function_{name}_{_x}_{_y}.png")
   
    print("="*60)
    
    return phi_ij, std_deviation, phi_ij_sol

if __name__ == "__main__":
    N = 100
    N_walkers_per_core = 1000

    x, y = [0.50, 0.02, 0.02], [0.50, 0.02, 0.50]

    f = np.zeros((N, N), dtype=np.float64)
    phi = np.zeros((N, N), dtype=np.float64)

    results_wrapper(x, y, phi, f, N_walkers_per_core, N, name="no_potential")
    

    


    
   
