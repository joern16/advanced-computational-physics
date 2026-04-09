#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of the Ising and XY model.
...
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from mpi4py import MPI
from parallel_statistics import ParallelMeanVariance

from numba import njit

@njit
def ising_walker(L, T, J=1.0, H=0.0, burn_in=1000, steps=10000):
    """
    Perform Metropolis sampling at temperature T.
    """
    # Initialize lattice
    lattice = np.random.choice([-1, 1], size=(L, L))
    
    # Burn-in period
    for _ in range(burn_in):
        i, j = np.random.randint(L, size=2)
        energy_difference = 2 * lattice[i, j] * (J * (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j] + 
                                                 lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L]) + H)
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] *= -1

    # Calculate energy and magnetization
    energies = np.zeros(steps + 1)
    for i in range(L):
        for j in range(L):
            spin = lattice[i, j]
            # Only sum right and down to avoid double-counting pairs
            neighbor_sum = (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            energies[0] += -J * spin * neighbor_sum - H * spin

    magnetizations = np.zeros(steps + 1)
    magnetizations[0] = np.sum(lattice)

    # Sampling period
    for iteration in range(steps):
        i, j = np.random.randint(L, size=2)
        energy_difference = 2 * lattice[i, j] * (J * (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j] + 
                                                 lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L]) + H)
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] *= -1
            energies[iteration + 1] = energies[iteration] + energy_difference
            magnetizations[iteration + 1] = magnetizations[iteration] + 2 * lattice[i, j]
        else:
            energies[iteration + 1] = energies[iteration]
            magnetizations[iteration + 1] = magnetizations[iteration]
    
    return energies, magnetizations

def ising_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    energies, magnetizations = ising_walker(L, T, J, H, burn_in, steps_per_core)

    calc = ParallelMeanVariance(size=2)
    calc.add_data(0, energies)
    calc.add_data(1, magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")

    # Plot energies and magnetizations for single walker
    if plot_walk and rank == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(energies)
        plt.title("energies")
        plt.savefig("energies")

        plt.figure(figsize=(8, 6))
        plt.plot(magnetizations)
        plt.title("magnetizations")
        plt.savefig("magnetizations")

    # Return mean and variance (in units of kB)
    if rank == 0:
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)

    
if __name__ == "__main__":
    ising_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=10000, steps_per_core=100000, plot_walk=True)
