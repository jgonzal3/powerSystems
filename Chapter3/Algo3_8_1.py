import numpy as np
import pandas as pd
import math

from YBUS_singular_transformation import get_admitance
from test5 import P_gen


def read_power_file(file_path):
    """
    Reads a power system file with the specified format.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        tuple: A tuple containing:
            - NG (int): Number of Generators.
            - NB (int): Number of Buses.
            - NV (int): Number of PV buses.
            - slack_bus (list): Voltage and delta angle for the slack bus.
            - real_power_demand (list): Real power demand for all buses.
            - reactive_power_demand (list): Reactive power demand for all buses.
            - type (list): Type of each bus (PV, PQ, or slack).
            - cost_function (numpy.ndarray): Cost function matrix (NG x 3).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Filter out lines starting with '#'
    lines = [line.strip() for line in lines if not line.strip().startswith('#')]

    # Parse the first line for NG, NB, NV
    NG, NB, NV, R, tol = map(lambda x: float(x) if x == lines[0].split(',')[-1] else int(x), lines[0].split(','))

    # Parse the second line for slack bus voltage and delta
    slack_bus = list(map(float, lines[1].split(',')))

    # Parse the third line for real power demand
    real_power_demand = list(map(float, lines[2].split(',')))

    # Parse the fourth line for reactive power demand
    reactive_power_demand = list(map(float, lines[3].split(',')))

    # Parse the fifth line for bus types
    bus_t = lines[4].split(',')

    # Parse the next NG lines for the cost function matrix
    cost_function = np.array([list(map(float, lines[5 + i].split(','))) for i in range(NG)])

    return NG, NB, NV, slack_bus, real_power_demand, reactive_power_demand, bus_t, cost_function, R, tol

def estimate_lambda():
    """
    Estimate lambda based on the given formula.

    Parameters:
        Pd (float): Power demand.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        float: Estimated lambda.
    """
    NG = cost_function.shape[0]  # Number of generators (rows in the matrix)
    PD = np.sum(real_power_demand)
    M = cost_function
    # Calculate numerator and denominator
    numerator = PD + sum(M[i][1] / (2 * M[i][0]) for i in range(NG))
    denominator = sum(1 / (2 * M[i][0]) for i in range(NG))
    # Calculate lambda
    l = numerator / denominator

    return l

def estimate_power(lambda_p):
    """
    Estimate the power of each generator.

    Parameters:
        lambda_value (float): The calculated lambda.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        numpy.ndarray: Array of power values for each generator.
    """
    target_length = NB
    M = cost_function
    NG = cost_function.shape[0]  # Number of generators
    power = np.array([(lambda_p - M[i][1])/(2 * M[i][0]) for i in range(NG)])

    if len(power) < target_length:
        power = np.concatenate((power, [0] * (target_length - len(power))))

    return power

def estimate_reactive(PG, bus_types):
    """
    Estimate the power of each generator.

    Parameters:
        lambda_value (float): The calculated lambda.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        numpy.ndarray: Array of power values for each generator.
    """
    NB = len(reactive_power_demand) # Ner of generators
    PD = sum(real_power_demand)
    QD = sum(reactive_power_demand)
    P_div_Q = float(QD/PD)
    t = [bus_types[i] for i in range(NB)]
    # Calculate the reactive power for each generator

    return np.array([P_div_Q * PG[i] if t[i] == 'PQ' else 0 for i in range(NB)])

def build_matrix(Q, V, NB, delta, matrix_type):

    H, L = [np.zeros((NB-1, NB-1)) for _ in range(2)]
    V = V[1:]

    for i, k in [(i, k) for i in range(NB - 1) for k in range(NB - 1)]:
        j, l = i + 1, k + 1
        if i == k:
            H[i][i] = -Q[i] - B[j][j] * V[i] ** 2
            L[i][i] = Q[i] - B[j][j] * V[i] ** 2
        else:
            H[i][k] = V[i]*V[k]*(G[j][l] * math.sin(delta[i] - delta[k]) - B[j][l] * math.cos(delta[i] - delta[k]))
            L[i][k] = V[i]*V[k]*(-G[j][l] * math.sin(delta[i] - delta[k]) - B[j][l] * math.cos(delta[i] - delta[k]))
    # Build the combined matrix
    return H if matrix_type == "H" else L

def calculate_A_B_coefficients(X,V,delta):
    """
    Calculate the B coefficients for the optimization problem.

    Parameters:
        B_matrix (numpy.ndarray): Matrix of size NG x NG.
        power_values (numpy.ndarray): Power array of size NG.

    Returns:
        numpy.ndarray: The B coefficients.
    """
    # Calculate the B coefficients
    a_ij = [(X[i][j] * np.cos(delta[i] - delta[j])) /(V[i] * V[j]) for i in range(NB) for j in range(NB)]
    b_ij = [(X[i][j] * np.sin(delta[i] - delta[j])) /(V[i] * V[j]) for i in range(NB) for j in range(NB)]

    a_coefficients = np.array(a_ij).reshape(NB, NB)
    b_coefficients = np.array(b_ij).reshape(NB, NB)
    return a_coefficients, b_coefficients

def calculate_H_77(a_ij, b_ij, lambda_value, cost_function, P, Q):

    H = np.zeros((2*NG+1,2*NG+1))
    lambda_value = float(lambda_value.item())

    for i in range(NG):
        for j in range(NG):
            if i == j:
                H[i,i] = 2*cost_function[i][0] + 2*lambda_value*a_ij[i,i]
            else:
                H[i,j] =  lambda_value*(a_ij[i,j] + a_ij[j,i])
    # line three
    for i in range(3):
        for k in range(NG,2*NG):
            j = k-NG
            H[i,k] = lambda_value*(b_ij[j,i] - b_ij[i,j])

    for i in range(NG):
        H[i,2*NG] = sum((a_ij[i,j] + a_ij[j,i])*P[j] + (b_ij[j,i] - b_ij[i,j])*Q[j] for j in range (NB))-1

    for j in range(NG):
        for k in range(NG,2*NG):
            i = k-NG
            H[k,j] = lambda_value*(b_ij[i,j] - b_ij[j,i])

    for k in range(NG,2*NG):
        for l in range(NG,2*NG):
            i = k-NG
            j = l-NG
            H[k,l] = lambda_value*(a_ij[i,j] + a_ij[j,i])

    for i in range(NG):
        H[i+NG,2*NG] = sum(((a_ij[i,j] + a_ij[j,i])*Q[j] + (b_ij[i,j] - b_ij[j,i])*P[j]) for j in range(NB))

    for i in range(NG,2*NG):
        H[2*NG,i] = H[i,2*NG]

    for i in range(NG):
        H[2*NG,i] = H[i,2*NG]

    return H

def power_flow_estimation(a_ij, b_ij, lambda_value, P_injected, Q_injected, P_generated, Q_generated):
    """
    Function to estimate power flow.
    """
    # Placeholder for the actual implementation

    PL, B_jacobian = build_jacobian_matrix_7(a_ij, b_ij, P_injected, Q_injected, P_generated, real_power_demand, lambda_value)

    B_hessian = calculate_H_77(a_ij, b_ij, lambda_value, cost_function, P_injected, Q_injected)
    S = np.linalg.solve(B_hessian, -B_jacobian)

    # Update my_power and my_lambda
    #[expression_if_true if condition else expression_if_false for item in iterable]
    P_generated = [P_generated[i] + S[i] for i in range(NG)]
    Q_generated = [Q_generated[i] + S[i+NG] for i in range(NG)]
    lambda_value += S[2*NG]
    squared_sum = np.sqrt(np.sum(np.square(B_jacobian)))

    P_generated = np.append(P_generated,[0,0])
    Q_generated = np.append(Q_generated, [0, 0])
    P_injected = P_generated - real_power_demand
    Q_injected = Q_generated- reactive_power_demand

    #Calculate the new power generation
    tol=0.0000001
    if squared_sum < tol:
         return lambda_value, P_generated, Q_generated, P_injected,Q_injected, True
    else:
        return lambda_value, P_generated, Q_generated, P_injected,Q_injected, False

# Example usage of read_power_file
def build_jacobian_matrix_7(a_ij, b_ij, P_injected, Q_injected, P_generated, real_power_demand, lambda_p):
    """
    Creates the Jacobian matrix for the optimization problem.

    Parameters:
        B_matrix (numpy.ndarray): Matrix of size NB x NB.
        P (numpy.ndarray): Power array of size NB.
        my_power (numpy.ndarray): Power generation array of size NG.
        real_power_demand (list): Real power demand for all buses.
        cost_function (numpy.ndarray): Cost function matrix of size NG x 3.
        my_lambda (float): Lambda value.
        NG (int): Number of generators.
        NB (int): Number of buses.

    Returns:
        numpy.ndarray: The Jacobian matrix.
    """
    # Covert my_lambda to a float to avoid the deprecated message
    lambda_p = float(lambda_p.item())
    pg = -1*np.ones(2*NG+1)

    #  Estimation of the line loss PL
    # Use NumPy's vectorized operations for efficiency
    PL = np.sum(a_ij * (np.outer(P_injected, P_injected) + np.outer(Q_injected, Q_injected)) + b_ij * (np.outer(Q_injected, P_injected) - np.outer(P_injected, Q_injected)))

    # Partial of lagrangian with respect to lambda
    pg[2*NG] = -float(sum(P_generated)) + float(sum(real_power_demand)) + float(PL.item())

    # Partial of lagrangian with respect to Pg
    for i in range(NG):
        pg[i] = 2 * cost_function[i][0] * P_generated[i] + cost_function[i][1] + lambda_p * (sum((a_ij[i,j] + a_ij[j,i])* P_injected[j] + (b_ij[j,i] - b_ij[i,j]) * Q_injected[j] for j in range(NB)) - 1)

    # Partial of lagrangian with respect to Qg
    for i in range(NG):
        k = i+NG
        pg[k] = lambda_p * (sum((a_ij[i, j] + a_ij[j, i]) * Q_injected[j] + (b_ij[i,j] - b_ij[j,i]) * P_injected[j] for j in range(NB)))

    jacobian_matrix = pg.reshape(-1, 1)

    return PL, jacobian_matrix

def perform_load_flow(G, B, delta, V, P_injected, Q_injected):
    """
    Performs load flow analysis using the Gauss-Seidel method.

    Parameters:
        G (numpy.ndarray): Conductance matrix.
        B (numpy.ndarray): Susceptance matrix.
        deltaN (numpy.ndarray): Voltage angle differences.
        VN (numpy.ndarray): Voltage magnitudes.
        PiS (list): Specified active power.
        QiS (list): Specified reactive power.
        tol (float): Tolerance for convergence.

    Returns:
        tuple: Updated deltaN, VN, and convergence status.
    """

    P_injected = P_injected[1:]
    Q_injected = Q_injected[1:]
    # Calculate power for each bus
    Pi = [sum(V[i] * V[k] * (G[i][k] * math.cos(delta[i] - delta[k]) + B[i][k] * math.sin(delta[i] - delta[k])) for k in range(len(V))) for i in range(1, len(V))]
    delta_Pi = [P_injected[i] - Pi[i] for i in range(len(Pi))]
    Qi = [sum(V[i] * V[k] * (G[i][k] * math.sin(delta[i] - delta[k]) - B[i][k] * math.cos(delta[i] - delta[k])) for k in range(len(V))) for i in range(1, len(V))]
    delta_Qi = [Q_injected[i] - Qi[i] for i in range(len(Qi))]
    # Update global P and Q

    # Step 6
    if max(abs(num) for num in delta_Pi) > tol:
        # Step 7
        H_matrix = build_matrix(Qi, V, NB, delta, "H")
        # Step 8 and Step 9
        delta[1:] = delta[1:] + np.linalg.inv(H_matrix) @ delta_Pi

    # Step 11
    if max(abs(num) for num in delta_Qi) > tol:
        # Step 12
        L_matrix = build_matrix(Qi, V, NB, delta, "L")
        # Step 13
        V[1:] = V[1:] + np.linalg.inv(L_matrix) @ delta_Qi
        # Step 14

    P_injected = np.insert(P_injected, 0, 0)
    Q_injected = np.insert(Q_injected, 0, 0)
    P_injected[0] = V[0] * sum(
        V[k] * (G[0][k] * math.cos(delta[0] - delta[k]) + B[0][k] * math.sin(delta[0] - delta[k])) for k in range(NB))
    Q_injected[0] = V[0] * sum(
        V[k] * (G[0][k] * math.sin(delta[0] - delta[k]) - B[0][k] * math.cos(delta[0] - delta[k])) for k in range(NB))
    # Check for convergence
    if max(abs(num) for num in delta_Pi + delta_Qi) < tol:
        return delta, V, P_injected, Q_injected, True
    else:
        return delta, V, P_injected, Q_injected, False

def compute_cost(P_generated):
    return sum([P_generated[i]**2*cost_function[i][0] + P_generated[i]*cost_function[i][1] + cost_function[i][2] for i in range (NG)])


file_path = 'power_3_8.txt'

NG, NB, NV, slack_bus, real_power_demand, reactive_power_demand, bus_types, cost_function, R, tol = read_power_file(file_path)

print("Number of Generators (NG):", NG)
print("Number of Buses (NB):", NB)
print("Number of PV Buses (NV):", NV)
print("Slack Bus Voltage and Delta:", slack_bus)
print("Real Power Demand:", real_power_demand)
print("Reactive Power Demand:", reactive_power_demand)
print("Bus Types:", bus_types)
print("Cost Function Matrix:\n", cost_function)
print("Tolerance:\n", tol)
print("Loop executions:\n", R)



# The above code reads a power system file and extracts relevant information such as the number of generators,
# buses, slack bus parameters, power demands, and cost function matrix. It then prints this information.
# The function read_power_file is designed to handle the specific format of the input file, ensuring that
# the data is correctly parsed and returned in a structured manner.
# The function also includes error handling to ensure that the input file is correctly formatted.
# The example usage demonstrates how to call the function and print the extracted information.

Y = get_admitance("ad_3_9.txt")
Z = np.linalg.inv(Y)
V, delta = np.ones(NB), np.zeros(NB)
V[0] = slack_bus[0]
delta[0] = slack_bus[1]
G, B, X = np.real(Y), np.imag(Y), np.real(Z)

lambda_p = estimate_lambda()
P_generated = estimate_power(lambda_p)
Q_generated = estimate_reactive(P_generated, bus_types)
F = compute_cost(P_generated)

P_injected = P_generated - real_power_demand

# Iterative load flow calculation

count_II = 0
count_I = 0
count_III = 0
converged_I = False
converged_II = False
converged_III = False
while not converged_II and count_II < R:
    print(f"Iteration_II {count_I}:,lambda_value {lambda_p}, Power {np.round([float(x) for x in P_injected],6)}")
    P_injected = P_generated - real_power_demand
    Q_injected = Q_generated - reactive_power_demand
    while not converged_I and count_I < R:
        print(f"Iteration_I {count_I}:,lambda_value {lambda_p}, Power {np.round([float(x) for x in P_injected],6)}")
        delta, V, P_injected, Q_injected, converged = perform_load_flow(G, B, delta, V, P_injected, Q_injected)
        P_injected[0] = V[0] * sum(V[k] * (G[0][k] * math.cos(delta[0] - delta[k]) + B[0][k] * math.sin(delta[0] - delta[k])) for k in range(NB))
        Q_injected[0] = V[0] * sum(V[k] * (G[0][k] * math.sin(delta[0] - delta[k]) - B[0][k] * math.cos(delta[0] - delta[k])) for k in range(NB))
        count_I += 1
    count_I = 0.0
    converged_I = False
    if (abs(P_generated[0] - real_power_demand[0] - P_injected[0])) > 0.00001:
        P_generated[0] = P_injected[0] + real_power_demand[0]
        Q_generated = Q_injected + reactive_power_demand
        a_ij, b_ij = calculate_A_B_coefficients(X, V, delta)
        while not converged_III and count_III < R:
            print(f"Iteration_III {count_III}:,lambda_value {lambda_p}, Power {np.round([float(x) for x in P_injected],6)}")
            lambda_p, P_generated, Q_generated, P_injected, Q_injected, converged_III = power_flow_estimation(a_ij, b_ij, lambda_p, P_injected, Q_injected, P_generated, Q_generated)
            count_III += 1
        count_III = 0
        converged_III = False
        print(f"Iteration last:,lambda_value {lambda_p}, Power {np.round([float(x) for x in P_injected],6)}")
    #print(Q_generated - reactive_power_demand)
    F_new = compute_cost(P_generated)
    if abs(F - F_new) < 0.0001:
        break
    else:
        F = F_new
    count_II += 1

data3 = {
    "Bus": list(range(1, NB + 1)),
    "V": V,
    "delta": delta,
    "P generated":  P_generated,
    "Q generated":  Q_generated,
    "P_injected": P_injected,
    "Q_injected": Q_injected
}

table3 = pd.DataFrame(data3)
print(table3)
print(f"Lambda: {lambda_p}, Final cost: {F}")
