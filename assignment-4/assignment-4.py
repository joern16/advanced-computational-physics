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
def ising_init(L, T, J=1.0, H=0.0, burn_in=1000, use_wolff=False):
    """
    Initialize lattice and perform burn-in for the Ising model.
    """
    lattice = np.empty((L, L), dtype=np.int64)
    for i in range(L):
        for j in range(L):
            lattice[i, j] = 1 if np.random.rand() < 0.5 else -1
            
    if use_wolff:
        p_add = 1.0 - np.exp(-2.0 * J / T)
        max_size = L * L
        stack_i = np.zeros(max_size, dtype=np.int32)
        stack_j = np.zeros(max_size, dtype=np.int32)
        for _ in range(burn_in):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            
            s_old = lattice[i, j]
            s_new = -s_old
            
            lattice[i, j] = s_new
            
            stack_i[0] = i
            stack_j[0] = j
            stack_ptr = 1
            
            while stack_ptr > 0:
                stack_ptr -= 1
                curr_i = stack_i[stack_ptr]
                curr_j = stack_j[stack_ptr]
                
                for k in range(4):
                    if k == 0:
                        ni, nj = (curr_i + 1) % L, curr_j
                    elif k == 1:
                        ni, nj = (curr_i - 1) % L, curr_j
                    elif k == 2:
                        ni, nj = curr_i, (curr_j + 1) % L
                    else:
                        ni, nj = curr_i, (curr_j - 1) % L
                        
                    if lattice[ni, nj] == s_old:
                        if np.random.rand() < p_add:
                            lattice[ni, nj] = s_new
                            stack_i[stack_ptr] = ni
                            stack_j[stack_ptr] = nj
                            stack_ptr += 1
    else:
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

@njit
def ising_run_batch_wolff(lattice, current_energy, current_magnetization, L, T, J=1.0, H=0.0, steps=10000):
    """
    Run a batch of Wolff cluster steps for the Ising model.
    """
    energies = np.zeros(steps)
    magnetizations = np.zeros(steps)
    
    p_add = 1.0 - np.exp(-2.0 * J / T)
    
    max_size = L * L
    stack_i = np.zeros(max_size, dtype=np.int32)
    stack_j = np.zeros(max_size, dtype=np.int32)
    cluster_i = np.zeros(max_size, dtype=np.int32)
    cluster_j = np.zeros(max_size, dtype=np.int32)
    in_cluster = np.zeros((L, L), dtype=np.bool_)
    
    for iteration in range(steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        s_old = lattice[i, j]
        s_new = -s_old
        
        lattice[i, j] = s_new
        in_cluster[i, j] = True
        
        stack_i[0] = i
        stack_j[0] = j
        stack_ptr = 1
        
        cluster_i[0] = i
        cluster_j[0] = j
        cluster_ptr = 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            curr_i = stack_i[stack_ptr]
            curr_j = stack_j[stack_ptr]
            
            for k in range(4):
                if k == 0:
                    ni, nj = (curr_i + 1) % L, curr_j
                elif k == 1:
                    ni, nj = (curr_i - 1) % L, curr_j
                elif k == 2:
                    ni, nj = curr_i, (curr_j + 1) % L
                else:
                    ni, nj = curr_i, (curr_j - 1) % L
                    
                if lattice[ni, nj] == s_old:
                    if np.random.rand() < p_add:
                        lattice[ni, nj] = s_new
                        in_cluster[ni, nj] = True
                        
                        stack_i[stack_ptr] = ni
                        stack_j[stack_ptr] = nj
                        stack_ptr += 1
                        
                        cluster_i[cluster_ptr] = ni
                        cluster_j[cluster_ptr] = nj
                        cluster_ptr += 1
                        
        delta_E = 0.0
        for c in range(cluster_ptr):
            ci = cluster_i[c]
            cj = cluster_j[c]
            
            for k in range(4):
                if k == 0:
                    ni, nj = (ci + 1) % L, cj
                elif k == 1:
                    ni, nj = (ci - 1) % L, cj
                elif k == 2:
                    ni, nj = ci, (cj + 1) % L
                else:
                    ni, nj = ci, (cj - 1) % L
                    
                if not in_cluster[ni, nj]:
                    delta_E += 2.0 * J * s_old * lattice[ni, nj]
                    
            in_cluster[ci, cj] = False
            
        delta_E += cluster_ptr * 2.0 * H * s_old
        
        current_energy += delta_E
        current_magnetization += cluster_ptr * 2 * s_new
        
        energies[iteration] = current_energy
        magnetizations[iteration] = current_magnetization
        
    return current_energy, current_magnetization, energies, magnetizations

def ising_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False, method="metropolis"):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    use_wolff = (method == "wolff")
    lattice, current_energy, current_magnetization = ising_init(L, T, J, H, burn_in, use_wolff)
    
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
        
        if method == "metropolis":
            current_energy, current_magnetization, energies, magnetizations = ising_run_batch(
                lattice, current_energy, current_magnetization, L, T, J, H, steps)
        elif method == "wolff":
            current_energy, current_magnetization, energies, magnetizations = ising_run_batch_wolff(
                lattice, current_energy, current_magnetization, L, T, J, H, steps)
        else:
            raise ValueError("Unknown method")
            
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
def xy_init(L, T, J=1.0, H=0.0, burn_in=1000, use_wolff=False):
    """
    Initialize lattice and perform burn-in for the XY model.
    """
    lattice = np.empty((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            lattice[i, j] = np.random.rand() * 2 * np.pi
    
    if use_wolff:
        max_size = L * L
        stack_i = np.zeros(max_size, dtype=np.int32)
        stack_j = np.zeros(max_size, dtype=np.int32)
        cluster_i = np.zeros(max_size, dtype=np.int32)
        cluster_j = np.zeros(max_size, dtype=np.int32)
        in_cluster = np.zeros((L, L), dtype=np.bool_)
        
        for _ in range(burn_in):
            i = np.random.randint(0, L)
            j = np.random.randint(0, L)
            
            phi = np.random.rand() * np.pi
            
            theta_old = lattice[i, j]
            theta_new = (2.0 * phi - theta_old) % (2.0 * np.pi)
            
            lattice[i, j] = theta_new
            in_cluster[i, j] = True
            
            stack_i[0] = i
            stack_j[0] = j
            stack_ptr = 1
            
            cluster_i[0] = i
            cluster_j[0] = j
            cluster_ptr = 1
            
            while stack_ptr > 0:
                stack_ptr -= 1
                curr_i = stack_i[stack_ptr]
                curr_j = stack_j[stack_ptr]
                curr_theta_old = (2.0 * phi - lattice[curr_i, curr_j]) % (2.0 * np.pi)
                
                for k in range(4):
                    if k == 0:
                        ni, nj = (curr_i + 1) % L, curr_j
                    elif k == 1:
                        ni, nj = (curr_i - 1) % L, curr_j
                    elif k == 2:
                        ni, nj = curr_i, (curr_j + 1) % L
                    else:
                        ni, nj = curr_i, (curr_j - 1) % L
                        
                    if not in_cluster[ni, nj]:
                        nj_theta_old = lattice[ni, nj]
                        
                        r_proj_i = np.sin(curr_theta_old - phi)
                        r_proj_j = np.sin(nj_theta_old - phi)
                        
                        if r_proj_i * r_proj_j > 0:
                            p_add = 1.0 - np.exp(-2.0 * J * r_proj_i * r_proj_j / T)
                            if np.random.rand() < p_add:
                                lattice[ni, nj] = (2.0 * phi - lattice[ni, nj]) % (2.0 * np.pi)
                                in_cluster[ni, nj] = True
                                
                                stack_i[stack_ptr] = ni
                                stack_j[stack_ptr] = nj
                                stack_ptr += 1
                                
                                cluster_i[cluster_ptr] = ni
                                cluster_j[cluster_ptr] = nj
                                cluster_ptr += 1
                                
            for c in range(cluster_ptr):
                in_cluster[cluster_i[c], cluster_j[c]] = False
    else:
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

@njit
def xy_run_batch_wolff(lattice, current_energy, Mx, My, L, T, J=1.0, H=0.0, steps=10000):
    """
    Run a batch of Wolff cluster steps for the XY model.
    """
    energies = np.zeros(steps)
    magnetizations = np.zeros(steps)
    
    max_size = L * L
    stack_i = np.zeros(max_size, dtype=np.int32)
    stack_j = np.zeros(max_size, dtype=np.int32)
    cluster_i = np.zeros(max_size, dtype=np.int32)
    cluster_j = np.zeros(max_size, dtype=np.int32)
    in_cluster = np.zeros((L, L), dtype=np.bool_)
    
    for iteration in range(steps):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        
        phi = np.random.rand() * np.pi
        
        theta_old = lattice[i, j]
        theta_new = (2.0 * phi - theta_old) % (2.0 * np.pi)
        
        lattice[i, j] = theta_new
        in_cluster[i, j] = True
        
        stack_i[0] = i
        stack_j[0] = j
        stack_ptr = 1
        
        cluster_i[0] = i
        cluster_j[0] = j
        cluster_ptr = 1
        
        while stack_ptr > 0:
            stack_ptr -= 1
            curr_i = stack_i[stack_ptr]
            curr_j = stack_j[stack_ptr]
            
            curr_theta_old = (2.0 * phi - lattice[curr_i, curr_j]) % (2.0 * np.pi)
            
            for k in range(4):
                if k == 0:
                    ni, nj = (curr_i + 1) % L, curr_j
                elif k == 1:
                    ni, nj = (curr_i - 1) % L, curr_j
                elif k == 2:
                    ni, nj = curr_i, (curr_j + 1) % L
                else:
                    ni, nj = curr_i, (curr_j - 1) % L
                    
                if not in_cluster[ni, nj]:
                    nj_theta_old = lattice[ni, nj]
                    
                    r_proj_i = np.sin(curr_theta_old - phi)
                    r_proj_j = np.sin(nj_theta_old - phi)
                    
                    if r_proj_i * r_proj_j > 0:
                        p_add = 1.0 - np.exp(-2.0 * J * r_proj_i * r_proj_j / T)
                        if np.random.rand() < p_add:
                            lattice[ni, nj] = (2.0 * phi - lattice[ni, nj]) % (2.0 * np.pi)
                            in_cluster[ni, nj] = True
                            
                            stack_i[stack_ptr] = ni
                            stack_j[stack_ptr] = nj
                            stack_ptr += 1
                            
                            cluster_i[cluster_ptr] = ni
                            cluster_j[cluster_ptr] = nj
                            cluster_ptr += 1
                            
        delta_E = 0.0
        for c in range(cluster_ptr):
            ci = cluster_i[c]
            cj = cluster_j[c]
            
            theta_new = lattice[ci, cj]
            theta_old = (2.0 * phi - theta_new) % (2.0 * np.pi)
            
            for k in range(4):
                if k == 0:
                    ni, nj = (ci + 1) % L, cj
                elif k == 1:
                    ni, nj = (ci - 1) % L, cj
                elif k == 2:
                    ni, nj = ci, (cj + 1) % L
                else:
                    ni, nj = ci, (cj - 1) % L
                    
                if not in_cluster[ni, nj]:
                    nj_theta = lattice[ni, nj]
                    E_old = -J * np.cos(theta_old - nj_theta)
                    E_new = -J * np.cos(theta_new - nj_theta)
                    delta_E += E_new - E_old
            
            E_H_old = -H * np.cos(theta_old)
            E_H_new = -H * np.cos(theta_new)
            delta_E += E_H_new - E_H_old
            
            Mx = Mx - np.cos(theta_old) + np.cos(theta_new)
            My = My - np.sin(theta_old) + np.sin(theta_new)
            
            in_cluster[ci, cj] = False
            
        current_energy += delta_E
        
        energies[iteration] = current_energy
        magnetizations[iteration] = np.sqrt(Mx**2 + My**2)
        
    return current_energy, Mx, My, energies, magnetizations

@njit
def xy_spin_correlation(lattice, L):
    max_r = L // 2
    correlations = np.zeros(max_r + 1)
    
    for r in range(max_r + 1):
        corr_sum = 0.0
        for i in range(L):
            for j in range(L):
                corr_sum += np.cos(lattice[i, j] - lattice[(i + r) % L, j])
                corr_sum += np.cos(lattice[i, j] - lattice[i, (j + r) % L])
        correlations[r] = corr_sum / (2.0 * L * L)
        
    return correlations

def xy_wrapper(L, T, J=1.0, H=0.0, burn_in=1000, steps_per_core=10000, plot_walk=False, method="metropolis"):
    """
    Run Metropolis sampling in parallel and calculate mean energy, mean magnetization, heat capacity, and magnetic susceptibility for the XY model.
    """
    start_time = time.time()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    calc = ParallelMeanVariance(size=2)

    use_wolff = (method == "wolff")
    lattice, current_energy, Mx, My = xy_init(L, T, J, H, burn_in, use_wolff)
    
    batch_size = 1000000
    all_energies = []
    all_magnetizations = []
    
    calc.add_data(0, np.array([current_energy]))
    calc.add_data(1, np.array([np.sqrt(Mx**2 + My**2)]))
    if plot_walk and rank == 0:
        all_energies.append(np.array([current_energy]))
        all_magnetizations.append(np.array([np.sqrt(Mx**2 + My**2)]))

    sum_correlations = np.zeros(L // 2 + 1)
    num_samples = 0

    for start_step in range(0, steps_per_core, batch_size):
        steps = min(batch_size, steps_per_core - start_step)
        
        if method == "metropolis":
            current_energy, Mx, My, energies, magnetizations = xy_run_batch(
                lattice, current_energy, Mx, My, L, T, J, H, steps)
        elif method == "wolff":
            current_energy, Mx, My, energies, magnetizations = xy_run_batch_wolff(
                lattice, current_energy, Mx, My, L, T, J, H, steps)
        else:
            raise ValueError("Unknown method")
            
        calc.add_data(0, energies)
        calc.add_data(1, magnetizations)
        
        sum_correlations += xy_spin_correlation(lattice, L)
        num_samples += 1
        
        if plot_walk and rank == 0:
            all_energies.append(energies)
            all_magnetizations.append(magnetizations)

    _, mean, var = calc.collect(comm=comm, mode="gather")
    
    local_mean_corr = sum_correlations / num_samples
    global_mean_corr = np.zeros_like(local_mean_corr)
    comm.Reduce(local_mean_corr, global_mean_corr, op=MPI.SUM, root=0)

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
        global_mean_corr /= comm.Get_size()
        
        print("\n" + "="*60)
        print("            XY Model Results for T = " + str(T) + " and L = " + str(L))
        print("="*60)
        print("Mean energy: ", mean[0])
        print("Mean magnetization: ", mean[1])
        print("Heat capacity: ", var[0] / (T**2))
        print("Magnetic susceptibility: ", var[1] / T)
        print("Time taken: ", time_taken)
        print("-" * 60)
        print("Spin-correlation C(r) vs r/L:")
        print("  r/L   | C(r)")
        print("  ----------------")
        for r in range(L // 2 + 1):
            print(f"  {r/L:<5.3f} | {global_mean_corr[r]:.5f}")
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
    ising_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=1*(10**5), steps_per_core=1*(10**9), plot_walk=False, method="wolff")
    xy_wrapper(L=16, T=1.0, J=1.0, H=0.0, burn_in=1*(10**5), steps_per_core=1*(10**9), plot_walk=False, method="wolff")


