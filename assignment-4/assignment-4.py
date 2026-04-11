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
def ising_init(L, T, J=1.0, H=0.0, burn_in=1000):
    """
    Initialize lattice and perform burn-in for the Ising model.
    """
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

    # Calculate initial energy and magnetization
    current_energy = 0.0
    for i in range(L):
        for j in range(L):
            spin = lattice[i, j]
            # Only sum right and down to avoid double-counting pairs
            neighbor_sum = (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            current_energy += -J * spin * neighbor_sum - H * spin

    current_magnetization = float(np.sum(lattice))
    return lattice, current_energy, current_magnetization

@njit
def ising_run_batch(lattice, current_energy, current_magnetization, L, T, J=1.0, H=0.0, steps=10000):
    """
    Run a batch of Metropolis steps for the Ising model.
    """
    energies = np.zeros(steps)
    magnetizations = np.zeros(steps)
    
    for iteration in range(steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        energy_difference = 2 * lattice[i, j] * (J * (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j] + 
                                                 lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L]) + H)
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] *= -1
            current_energy += energy_difference
            current_magnetization += 2 * lattice[i, j]
            
        energies[iteration] = current_energy
        magnetizations[iteration] = current_magnetization
        
    return current_energy, current_magnetization, energies, magnetizations

def ising_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    lattice, current_energy, current_magnetization = ising_init(L, T, J, H, burn_in)
    
    batch_size = 1000000
    all_energies = []
    all_magnetizations = []
    
    calc.add_data(0, np.array([current_energy]))
    calc.add_data(1, np.array([current_magnetization]))
    if plot_walk and rank == 0:
        all_energies.append(np.array([current_energy]))
        all_magnetizations.append(np.array([current_magnetization]))
        
    for start_step in range(0, steps_per_core, batch_size):
        steps = min(batch_size, steps_per_core - start_step)
        current_energy, current_magnetization, energies, magnetizations = ising_run_batch(
            lattice, current_energy, current_magnetization, L, T, J, H, steps)
        calc.add_data(0, energies)
        calc.add_data(1, magnetizations)
        
        if plot_walk and rank == 0:
            all_energies.append(energies)
            all_magnetizations.append(magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")

    time_taken = time.time() - start_time

    # Plot energies and magnetizations for single walker
    if plot_walk and rank == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(np.concatenate(all_energies))
        plt.title("Ising - Energies")
        plt.savefig("ising_energies.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(np.concatenate(all_magnetizations))
        plt.title("Ising - Magnetizations")
        plt.savefig("ising_magnetizations.png")
        plt.close()

    # Return mean and variance (in units of kB)
    if rank == 0:
        print("\n" + "="*60)
        print("           Ising Model Results for T = " + str(T) + " and L = " + str(L))
        print("="*60)
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)
        print("Time taken: ", time_taken)
        print("="*60)

@njit
def xy_init(L, T, J=1.0, H=0.0, burn_in=1000):
    """
    Initialize lattice and perform burn-in for the XY model.
    """
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
    current_energy = 0.0
    Mx = 0.0
    My = 0.0
    for i in range(L):
        for j in range(L):
            theta = lattice[i, j]
            # Only sum right and down to avoid double-counting pairs
            theta_right = lattice[i, (j + 1) % L]
            theta_down = lattice[(i + 1) % L, j]
            current_energy += -J * np.cos(theta - theta_right) - J * np.cos(theta - theta_down) - H * np.cos(theta)
            Mx += np.cos(theta)
            My += np.sin(theta)

    return lattice, current_energy, Mx, My

@njit
def xy_run_batch(lattice, current_energy, Mx, My, L, T, J=1.0, H=0.0, steps=10000):
    """
    Run a batch of Metropolis steps for the XY model.
    """
    energies = np.zeros(steps)
    magnetizations = np.zeros(steps)
    
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
            current_energy += energy_difference
            Mx = Mx - np.cos(theta_old) + np.cos(theta_new)
            My = My - np.sin(theta_old) + np.sin(theta_new)
            
        energies[iteration] = current_energy
        magnetizations[iteration] = np.sqrt(Mx**2 + My**2)
        
    return current_energy, Mx, My, energies, magnetizations

def xy_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility for the XY model.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    lattice, current_energy, Mx, My = xy_init(L, T, J, H, burn_in)
    
    batch_size = 1000000
    all_energies = []
    all_magnetizations = []
    
    calc.add_data(0, np.array([current_energy]))
    calc.add_data(1, np.array([np.sqrt(Mx**2 + My**2)]))
    if plot_walk and rank == 0:
        all_energies.append(np.array([current_energy]))
        all_magnetizations.append(np.array([np.sqrt(Mx**2 + My**2)]))

    for start_step in range(0, steps_per_core, batch_size):
        steps = min(batch_size, steps_per_core - start_step)
        current_energy, Mx, My, energies, magnetizations = xy_run_batch(
            lattice, current_energy, Mx, My, L, T, J, H, steps)
        calc.add_data(0, energies)
        calc.add_data(1, magnetizations)
        
        if plot_walk and rank == 0:
            all_energies.append(energies)
            all_magnetizations.append(magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")

    time_taken = time.time() - start_time

    # Plot energies and magnetizations for single walker
    if plot_walk and rank == 0:
        plt.figure(figsize=(8, 6))
        plt.plot(np.concatenate(all_energies))
        plt.title("XY Model - Energies")
        plt.savefig("xy_energies.png")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(np.concatenate(all_magnetizations))
        plt.title("XY Model - Magnetizations")
        plt.savefig("xy_magnetizations.png")
        plt.close()

    # Return mean and variance (in units of kB)
    if rank == 0:
        print("\n" + "="*60)
        print("            XY Model Results for T = " + str(T) + " and L = " + str(L))
        print("="*60)
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)
        print("Time taken: ", time_taken)
        print("="*60)

if __name__ == "__main__":
    """
    for T in np.linspace(1.0,3.0,num=100):
        ising_wrapper(L=16, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        ising_wrapper(L=32, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        ising_wrapper(L=64, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        ising_wrapper(L=128, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        ising_wrapper(L=256, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)

    for T in np.linspace(0.01,2.0,num=100):
        xy_wrapper(L=16, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        xy_wrapper(L=32, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        xy_wrapper(L=64, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        xy_wrapper(L=128, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)
        xy_wrapper(L=256, T=T, J=1.0, H=0.0, burn_in=10**5, steps_per_core=10**9, plot_walk=False)  
    """
    ising_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=1*(10**5), steps_per_core=1*(10**9), plot_walk=False)
    xy_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=1*(10**5), steps_per_core=1*(10**9), plot_walk=False)


