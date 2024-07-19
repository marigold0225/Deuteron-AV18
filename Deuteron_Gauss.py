import numpy as np
import matplotlib.pyplot as plt


def g11(r, E):
    V0 = 72.194
    ra = 1.484
    r = r / ra
    Vr = -V0 * np.exp(-(r**2))
    g11 = (Vr - E) / a
    return g11


def calculate_wave_function(r, E):
    G11 = g11(r, E)
    # Initialize arrays
    u = np.zeros(n + 1)
    k = np.zeros(n + 1)
    u[1] = 0.1
    # Calculate the solution using the loop
    for i in range(1, n):
        k[i - 1] = 1 - G11[i - 1] * d2
        k[i] = 1 - G11[i] * d2
        k[i + 1] = 1 - G11[i + 1] * d2
        A1 = (12 - 10 * k[i]) * u[i] - k[i - 1] * u[i - 1]
        u[i + 1] = A1 / k[i + 1]
    # Calculate the normalization constant (integral of u^2)
    norm_constant = np.sum(u**2) * delta
    # Normalize the wave function
    u /= np.sqrt(norm_constant)
    return u


r0 = 0.0
delta = 0.01
n = 2500
a = 41.47
d2 = delta**2 / 12
planck = 197.33
# Initialize arrays using NumPy
r = np.linspace(r0, r0 + delta * n, n + 1)


def find_energy_eigenvalue(E_min, E_max, tolerance, max_iterations):
    for _ in range(max_iterations):
        # Choose the mid-point energy value
        E_mid = (E_min + E_max) / 2

        # Calculate the wave function using the calculate_wave_function function
        u = calculate_wave_function(r, E_mid)

        # Check if the last value of u is within the tolerance
        if abs(u[-1]) < tolerance:
            return E_mid

        # Update the search range based on the sign of the last value of u
        if u[-1] > 0:
            E_min = E_mid
        else:
            E_max = E_mid

    return None  # Return None if the solution is not found within max_iterations


# Search for the energy eigenvalue within the given range
E_min = -3.0
E_max = 3.0
tolerance = 1e-5
max_iterations = 100

if __name__ == "__main__":
    found_energy = find_energy_eigenvalue(E_min, E_max, tolerance, max_iterations)

    if found_energy is not None:
        print("Found energy eigenvalue E:", found_energy)
        u = calculate_wave_function(r, found_energy)
        plt.plot(r, u, label="u(r)")
        plt.xlabel("r")
        plt.ylabel("u(r)")
        plt.xlim(0, 25)
        plt.ylim(0, 0.6)
        plt.title("Deuteron Wave Function with Gauss potential")
        plt.legend()
        plt.grid()
        plt.show()
        plt.savefig("Fig-Gauss")
    else:
        print("Energy eigenvalue E not found within the given range.")
