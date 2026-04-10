#!/opt/software/anaconda/python-3.10.9/bin/python

"""
Numerical evaluation of the Ising and XY model.
...
License: MIT
"""
import time

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
    lattice = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            lattice[i, j] = 1 if np.random.rand() < 0.5 else -1
    
    # Burn-in period
    for _ in range(burn_in):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
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
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
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
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    energies, magnetizations = ising_walker(L, T, J, H, burn_in, steps_per_core)
    calc.add_data(0, energies)
    calc.add_data(1, magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")

    time_taken = time.time() - start_time

    # Plot energies and magnetizations for single walker
    if plot_walk and rank == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(energies)
        plt.title("Ising - Energies")
        plt.savefig("ising_energies.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(magnetizations)
        plt.title("Ising - Magnetizations")
        plt.savefig("ising_magnetizations.png")
        plt.close()

    # Return mean and variance (in units of kB)
    if rank == 0:
        print("\n" + "="*60)
        print("                Ising Model Results for T = " + str(T))
        print("="*60)
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)
        print("Time taken: ", time_taken)
        print("="*60)

@njit
def xy_walker(L, T, J=1.0, H=0.0, burn_in=1000, steps=10000):
    """
    Perform Metropolis sampling for the XY model at temperature T.
    """
    # Initialize lattice
    lattice = np.empty((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            lattice[i, j] = np.random.rand() * 2 * np.pi
    
    # Burn-in period
    for _ in range(burn_in):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        theta_old = lattice[i, j]
        theta_new = np.random.rand() * 2 * np.pi
        
        # Calculate local energy difference
        theta_up = lattice[(i - 1) % L, j]
        theta_down = lattice[(i + 1) % L, j]
        theta_left = lattice[i, (j - 1) % L]
        theta_right = lattice[i, (j + 1) % L]
        
        E_old_local = -J * (np.cos(theta_old - theta_up) + np.cos(theta_old - theta_down) + 
                            np.cos(theta_old - theta_left) + np.cos(theta_old - theta_right)) - H * np.cos(theta_old)
        E_new_local = -J * (np.cos(theta_new - theta_up) + np.cos(theta_new - theta_down) + 
                            np.cos(theta_new - theta_left) + np.cos(theta_new - theta_right)) - H * np.cos(theta_new)
        
        energy_difference = E_new_local - E_old_local
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] = theta_new

    # Calculate initial energy and magnetization
    energies = np.zeros(steps + 1)
    Mx = 0.0
    My = 0.0
    for i in range(L):
        for j in range(L):
            theta = lattice[i, j]
            # Only sum right and down to avoid double-counting pairs
            theta_right = lattice[i, (j + 1) % L]
            theta_down = lattice[(i + 1) % L, j]
            energies[0] += -J * np.cos(theta - theta_right) - J * np.cos(theta - theta_down) - H * np.cos(theta)
            Mx += np.cos(theta)
            My += np.sin(theta)

    magnetizations = np.zeros(steps + 1)
    magnetizations[0] = np.sqrt(Mx**2 + My**2)

    # Sampling period
    for iteration in range(steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        theta_old = lattice[i, j]
        theta_new = np.random.rand() * 2 * np.pi
        
        # Calculate local energy difference
        theta_up = lattice[(i - 1) % L, j]
        theta_down = lattice[(i + 1) % L, j]
        theta_left = lattice[i, (j - 1) % L]
        theta_right = lattice[i, (j + 1) % L]
        
        E_old_local = -J * (np.cos(theta_old - theta_up) + np.cos(theta_old - theta_down) + 
                            np.cos(theta_old - theta_left) + np.cos(theta_old - theta_right)) - H * np.cos(theta_old)
        E_new_local = -J * (np.cos(theta_new - theta_up) + np.cos(theta_new - theta_down) + 
                            np.cos(theta_new - theta_left) + np.cos(theta_new - theta_right)) - H * np.cos(theta_new)
        
        energy_difference = E_new_local - E_old_local
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] = theta_new
            energies[iteration + 1] = energies[iteration] + energy_difference
            Mx = Mx - np.cos(theta_old) + np.cos(theta_new)
            My = My - np.sin(theta_old) + np.sin(theta_new)
            magnetizations[iteration + 1] = np.sqrt(Mx**2 + My**2)
        else:
            energies[iteration + 1] = energies[iteration]
            magnetizations[iteration + 1] = magnetizations[iteration]
    
    return energies, magnetizations

def xy_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility for the XY model.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    energies, magnetizations = xy_walker(L, T, J, H, burn_in, steps_per_core)
    calc.add_data(0, energies)
    calc.add_data(1, magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")

    time_taken = time.time() - start_time

    # Plot energies and magnetizations for single walker
    if plot_walk and rank == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(energies)
        plt.title("XY Model - Energies")
        plt.savefig("xy_energies.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(magnetizations)
        plt.title("XY Model - Magnetizations")
        plt.savefig("xy_magnetizations.png")
        plt.close()

    # Return mean and variance (in units of kB)
    if rank == 0:
        print("\n" + "="*60)
        print("                 XY Model Results for T = " + str(T))
        print("="*60)
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)
        print("Time taken: ", time_taken)
        print("="*60)

if __name__ == "__main__":
    ising_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=10000, steps_per_core=100000, plot_walk=True)
    xy_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=10000, steps_per_core=100000, plot_walk=True)
