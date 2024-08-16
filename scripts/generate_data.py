import numpy as np
from scipy.linalg import eigh
from scipy.integrate import simps
import os

# Define parameters for the potential well
L = 1.0  # Width of the potential well
N = 100  # Number of grid points
num_profiles = 100  # Number of potential profiles to generate

# Define the potential well
def potential_well(x, profile):
    return np.interp(x, np.linspace(0, L, len(profile)), profile)

# Construct the Hamiltonian matrix
def construct_hamiltonian(x, dx, V):
    H = -0.5 * (np.diag(np.ones(N-1), -1) - 2 * np.diag(np.ones(N)) + np.diag(np.ones(N-1), 1))
    H = H / (dx**2)
    H += np.diag(V)
    return H

# Compute the eigenvalues and eigenstates of the Hamiltonian
def solve_schrodinger(hamiltonian):
    eigenvalues, eigenstates = eigh(hamiltonian)
    return eigenvalues, eigenstates

# Compute the density matrix from the eigenstate
def density_matrix(state):
    return np.outer(state, np.conj(state))

# Compute the Von Neumann entropy
def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 0]  # Remove zero eigenvalues
    entropy = -np.sum(eigvals * np.log(eigvals))
    return entropy

def generate_data():
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    entanglement_values = []
    potential_profiles = []

    for _ in range(num_profiles):
        profile = np.random.uniform(-1, 1, size=N)  # Random potential profile
        V = potential_well(x, profile)
        H = construct_hamiltonian(x, dx, V)
        eigenvalues, eigenstates = solve_schrodinger(H)
        ground_state = eigenstates[:, 0]
        density_rho = density_matrix(ground_state)
        entropy = von_neumann_entropy(density_rho)
        entanglement_values.append(entropy)
        potential_profiles.append(profile)

    # Save data
    np.save('data/potential_profiles.npy', np.array(potential_profiles))
    np.save('data/entanglement_values.npy', np.array(entanglement_values))

if __name__ == "__main__":
    generate_data()
