import numpy as np
from scipy.optimize import fsolve
from Algorithm2_1 import get_admitance
from Algo_GS import process_data_file



# Define the sV2stem of nonlinear equations
def equation_old(vars):
    # Split the real and imaginary parts
    Y = get_admitance()
    power_matrix = process_data_file('data.txt')
    V2_real, V2_imag, V3_real, V3_imag, V4_real, V4_imag = vars
    V2 = V2_real + 1j * V2_imag
    V3 = V3_real + 1j * V3_imag
    V4 = V4_real + 1j * V4_imag

    # Example equations
    eq2 = np.conj(V2)*(Y[1][0]*V1 + Y[1][1]*V2 + Y[1][2]*V3 + Y[1][3]*V4) + (0.45 - 0.15j) # V1 * V2 = 1 + j
    eq3 = np.conj(V3)*(Y[2][0]*V1 + Y[2][1]*V2 + Y[2][2]*V3 + Y[2][3]*V4) + (0.51 - 0.25j) # V1 * V3 = 3 + j
    eq4 = np.conj(V4)*(Y[3][0]*V1 + Y[3][1]*V2 + Y[3][2]*V3 + Y[3][3]*V4) + (0.60 - 0.30j)  # V1 * V4 = 4 + j

    # Return real and imaginary parts as separate equations
    return [eq2.real, eq2.imag, eq3.real, eq3.imag, eq4.real, eq4.imag]

def equations(vars):

    n = Y.shape[0]  # Number of buses
    V1 = power_matrix[0]  # Known voltage at bus 1 (slack bus)

    # Split the real and imaginary parts of the unknown voltages
    voltages = [V1]  # Start with V1
    equations = []

    for i in range(0, len(vars), 2):
        real = vars[i]
        imag = vars[i + 1]
        voltages.append(real + 1j * imag)

    # Construct the equations
    for i in range(1, n):  # Skip the slack bus (V1)
        Vi = voltages[i]
        power = power_matrix[i]  # Power at bus i (adjust index for 0-based)
        equation = np.conj(Vi) * sum(Y[i][j] * voltages[j] for j in range(n)) - np.conj(power)
        equations.append(equation.real)  # Real part
        equations.append(equation.imag)
    return np.array(equations).reshape(-1)


# Get the admittance matrix and power matrix
Y = get_admitance()
power_matrix = process_data_file('data.txt')

initial_guess = np.array([1, 0] * (Y.shape[0]-1) ) # Initial guess for unknowns
solution = fsolve(equations, initial_guess)

# Combine the real and imaginary parts into complex numbers
n = len(solution) // 2
V_solution = [1.05 + 0j] + [solution[2*i] + 1j * solution[2*i + 1] for i in range(n)]

# Print the solutions
for i, V in enumerate(V_solution, start=1):
    print(f"V{i} = {np.round(V, 4)}")
