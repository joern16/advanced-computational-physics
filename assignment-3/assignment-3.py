#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of the Poisson and Laplace equations on grids.
...
License: MIT
"""
import time

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from parallel_statistics import ParallelMeanVariance

from numba import njit



def optimal_omega(N):
    """
    Returns the optimal parameter for over-relaxation for an N x N square grid.
    """
    return 2.0 / (1.0 + np.sin(np.pi / N))

def solve_poisson_over_relaxation(phi, f, omega, tol=1e-10, max_iter=10000):
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

                phi[i,j] = (omega * (h**2 * f[i,j] + 0.25*(phi[i+1,j] + phi[i-1,j]
                    + phi[i,j+1] + phi[i,j-1])) + (1.0 - omega) * phi_old)

                diff = abs(phi[i, j] - phi_old)
                max_diff = max(max_diff, diff)

        if max_diff < tol:
            return phi, iterations + 1

    return phi, max_iter

@njit
def random_walk(i_start, j_start, N):
    """
    Perform a random walk on a square grid.
    """
    visits = np.zeros((N, N), dtype=np.int64)
    i, j = i_start, j_start
    while True:
        visits[i, j] += 1
        if i == 0 or i == N - 1 or j == 0 or j == N - 1:
            break

        r = np.random.rand()
        if r < 0.25:
            i += 1
        elif r < 0.5:
            i -= 1
        elif r < 0.75:
            j += 1
        else:
            j -= 1

    return visits

def greens_function_parallel_std_dev_approx(i_start, j_start, N, N_walkers_per_core):
    """
    Evaluate the greens function at (i, j) using multiple random walks in parallel.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    visits = np.zeros((N, N), dtype=np.int64)
    variance = 0.0

    for _ in range(N_walkers_per_core):
        _visits = random_walk(i_start, j_start, N)
        _visits_sq = _visits**2
        visits += _visits
        variance += (_visits_sq.sum() - _visits.sum()**2) / (N * N)

    total_visits = comm.reduce(visits, op=MPI.SUM, root=0)
    total_variance = comm.reduce(variance, op=MPI.SUM, root=0)

    if rank == 0:
        greens_ij = total_visits / (N_walkers_per_core * size)

        return greens_ij, np.sqrt(abs(total_variance))

    return None, None

def greens_function_parallel(i_start, j_start, N, N_walkers_per_core):
    """
    Evaluate the greens function at (i, j) using multiple random walks in parallel.
    Uses parallel_statistics for accurate distributed variance and mean calculation.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    calc = ParallelMeanVariance(size=N*N)

    # Collect all walks of a batch in a 2D array. Batches save memory, otherwise OOM error
    batch_size = 1000
    for batch_start_idx in range(0, N_walkers_per_core, batch_size):
        batch_end_idx = min(batch_start_idx + batch_size, N_walkers_per_core)
        current_batch_size = batch_end_idx - batch_start_idx

        batch_visits = np.zeros((current_batch_size, N*N), dtype=np.int64)
        for w in range(current_batch_size):
            batch_visits[w, :] = random_walk(i_start, j_start, N).flatten()

        # Add batch data for each pixel at once using add_data
        for pixel_idx in range(N*N):
            calc.add_data(pixel_idx, batch_visits[:, pixel_idx])

    # Add variances and means among all ranks (gather returns to rank 0)
    _, mean, variance = calc.collect(comm=comm, mode="gather")

    if rank == 0:
        greens_ij = mean.reshape((N, N))

        # Calculate standard deviation as the square root of the sum of variances of all the pixels
        std_deviation = float(np.sqrt(np.sum(variance)))
        return greens_ij, std_deviation

    return None, None

def solve_poisson_greens(phi, f, N, greens_ij):
    """
    Evaluate phi at (i, j) using a computed greens function.
    """
    h = 1.0 / (N - 1)
    phi[1:-1, 1:-1] = 0.0 # makes sure that phi only containes the boundary values

    phi_ij = 0.0
    for i in range(N):
        for j in range(N):
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
    plt.xlabel('y (m)')
    plt.ylabel('x (m)')
    if filename:
        plt.savefig(filename)
    plt.show()


def result_wrapper(points_xy, phi, f, N_walkers_per_core, N, name="test", plot=False):
    """
    Obtains results for the poisson equation.
    """
    h = 1.0 / (N - 1)
    phi_sol, _ =  solve_poisson_over_relaxation(phi.copy(), f, optimal_omega(N))

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("\n" + "="*60)
        print("                 Restuls: " + name)
        print("="*60)

    for _x, _y in points_xy:
        i, j = int(_x // h), int(_y // h)
        start_time = time.time()
        greens_ij, std_deviation = greens_function_parallel(i, j, N, N_walkers_per_core)
        time_taken = time.time() - start_time

        if rank == 0:
            phi_ij = solve_poisson_greens(phi, f, N, greens_ij)
            phi_ij_sol = phi_sol[i, j]

            print(f"Time taken for greens function  : {time_taken:.8f}")
            print(f"Standard deviation ({_x}, {_y}) : {std_deviation:.8f}")
            print(f"Random walk phi({_x}, {_y})     : {phi_ij:.8f}")
            print(f"Gauss-Seidel phi({_x}, {_y})    : {phi_ij_sol:.8f}")
            print(f"Difference ({_x}, {_y})         : {abs(phi_ij_sol-phi_ij):.8f}" + "\n")

            if plot:
                plot_greens_function(greens_ij, title= name + f": greens function at ({_x}, {_y})",
                    filename=f"greens_function_{name}_{_x}_{_y}.png")

    if rank == 0:
        print("="*60)

if __name__ == "__main__":
    N = 200
    N_walkers_per_core = 1000

    # Evaluation points
    points_xy = [(0.50, 0.50), (0.02, 0.02), (0.02, 0.50)]

    # Construct phi for (a), (b) and (c)
    phi_a = np.full((N, N), 100.0,dtype=np.float64)

    phi_b = np.full((N, N), 100.0,dtype=np.float64)
    phi_b[0, :] = 100.0   # Bottom
    phi_b[-1, :] = 100.0  # Top
    phi_b[:, 0] = -100.0   # Left
    phi_b[:, -1] = -100.0  # Right

    phi_c = np.full((N, N), 100.0,dtype=np.float64)
    phi_c[0, :] = 0.0   # Bottom
    phi_c[-1, :] = 200.0  # Top
    phi_c[:, 0] = 200.0   # Left
    phi_c[:, -1] = -400.0  # Right

    # Construct f for 0, (a), (b) and (c)
    f_0 = np.full((N, N), 0.0,dtype=np.float64)
    f_a = np.full((N, N), 10.0,dtype=np.float64)
    f_b = np.linspace(1, 0, N)[:, None] + np.zeros((1, N))

    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    f_c = np.exp(-10.0 * np.sqrt((X - 0.5)**2 + (Y - 0.5)**2))


    # f = 0
    result_wrapper(points_xy, phi_a, f_0, N_walkers_per_core, N, name="phi_a_f_0", plot=True)
    result_wrapper(points_xy, phi_b, f_0, N_walkers_per_core, N, name="phi_b_f_0")
    result_wrapper(points_xy, phi_c, f_0, N_walkers_per_core, N, name="phi_c_f_0")
"""
    # f (a)
    result_wrapper(points_xy, phi_a, f_a, N_walkers_per_core, N, name="phi_a_f_a")
    result_wrapper(points_xy, phi_b, f_a, N_walkers_per_core, N, name="phi_b_f_a")
    result_wrapper(points_xy, phi_c, f_a, N_walkers_per_core, N, name="phi_c_f_a")

    # f (b)
    result_wrapper(points_xy, phi_a, f_b, N_walkers_per_core, N, name="phi_a_f_b")
    result_wrapper(points_xy, phi_b, f_b, N_walkers_per_core, N, name="phi_b_f_b")
    result_wrapper(points_xy, phi_c, f_b, N_walkers_per_core, N, name="phi_c_f_b")

    # f (c)
    result_wrapper(points_xy, phi_a, f_c, N_walkers_per_core, N, name="phi_a_f_c")
    result_wrapper(points_xy, phi_b, f_c, N_walkers_per_core, N, name="phi_b_f_c")
    result_wrapper(points_xy, phi_c, f_c, N_walkers_per_core, N, name="phi_c_f_c")
"""
