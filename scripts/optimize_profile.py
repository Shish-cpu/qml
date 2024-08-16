import numpy as np
import joblib
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.linalg import eigh

# Load the trained model
model = joblib.load('models/rf_regressor.pkl')

# Define parameters for the potential well
L = 1.0
N = 100

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
    eigvals = eigvals[eigvals > 0]
    entropy = -np.sum(eigvals * np.log(eigvals))
    return entropy

# Objective function to maximize
def objective(profile):
    x = np.linspace(0, L, N)
    V = potential_well(x, profile)
    dx = x[1] - x[0]
    H = construct_hamiltonian(x, dx, V)
    eigenvalues, eigenstates = solve_schrodinger(H)
    ground_state = eigenstates[:, 0]
    density_rho = density_matrix(ground_state)
    entropy = von_neumann_entropy(density_rho)
    return -entropy  # Negative because we want to maximize entropy

# Optimization function
def optimize_profile():
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]

    # Initial guess for parameters
    initial_profile = np.random.uniform(-1, 1, size=N)

    result = minimize(objective, initial_profile, method='L-BFGS-B', bounds=[(-1, 1)]*N)

    optimized_profile = result.x

    # Plot the optimized potential profile
    plt.plot(x, optimized_profile)
    plt.title("Optimized Potential Profile")
    plt.xlabel("x")
    plt.ylabel("Potential V(x)")
    plt.savefig('optimized_potential_profile.png')

    print("Optimized Profile:", optimized_profile)
    print("Maximum Entanglement:", -result.fun)

if __name__ == "__main__":
    optimize_profile()
