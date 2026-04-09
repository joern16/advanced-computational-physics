import numpy as np
import matplotlib.pyplot as plt

def ising_lattice_init(L, random=True):
    """
    Initializes an L x L lattice.
    """
    if random:
        return np.random.choice([-1, 1], size=(L, L))
    return np.ones((L, L), dtype=int)

def ising_energy(lattice, J=1.0, H=0.0):
    """
    Calculate total energy of ising lattice with periodic BC.
    """
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            spin = lattice[i, j]
            # Only sum right and down to avoid double-counting pairs
            neighbor_sum = (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            energy += -J * spin * neighbor_sum - H * spin
    return energy

def ising_walker(L, T, J=1.0, H=0.0, burn_in=1000, steps=10000):
    """
    Perform Metropolis sampling at temperature T.
    """
    lattice = ising_lattice_init(L)
    
    # Burn-in period
    for _ in range(burn_in):
        i, j = np.random.randint(L, size=2)
        energy_difference = 2 * lattice[i, j] * (J * (lattice[(i + 1) % L, j] + lattice[(i - 1) % L, j] + 
                                                 lattice[i, (j + 1) % L] + lattice[i, (j - 1) % L]) + H)
        
        if energy_difference <= 0 or np.random.rand() < np.exp(-energy_difference / T):
            lattice[i, j] *= -1

    # Sampling period
    energies = np.zeros(steps + 1)
    magnetizations = np.zeros(steps + 1)

    energies[0] = ising_energy(lattice, J, H)
    magnetizations[0] = np.sum(lattice)

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


L = 16
T = 1.0

energies, magnetizations = ising_walker(L, T, steps=100000, burn_in=10000)

print(energies)
plt.figure(figsize=(8, 6))
plt.plot(energies)
plt.title("energies")
plt.savefig("energies")
plt.show()

print(magnetizations)
plt.figure(figsize=(8, 6))
plt.plot(magnetizations)
plt.title("magnetizations")
plt.savefig("magnetizations")
plt.show()
