import numpy as np
import pandas as pd
import math

from YBUS_singular_transformation import get_admitance

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

def estimate_lambda(Pd, M):
    """
    Estimate lambda based on the given formula.

    Parameters:
        Pd (float): Power demand.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        float: Estimated lambda.
    """
    NG = M.shape[0]  # Number of generators (rows in the matrix)
    PD = np.sum(Pd)
    # Calculate numerator and denominator
    numerator = PD + sum(M[i][1] / (2 * M[i][0]) for i in range(NG))
    denominator = sum(1 / (2 * M[i][0]) for i in range(NG))
    # Calculate lambda
    l = numerator / denominator

    return l

def estimate_power(lambda_value, M, target_length=5):
    """
    Estimate the power of each generator.

    Parameters:
        lambda_value (float): The calculated lambda.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        numpy.ndarray: Array of power values for each generator.
    """
    NG = M.shape[0]  # Number of generators
    power = np.array([(lambda_value - M[i][1])/(2 * M[i][0]) for i in range(NG)])

    if len(power) < target_length:
        power = np.concatenate((power, [0] * (target_length - len(power))))

    return power

def estimate_reactive(reactive_power_demand, real_power_demand, power_generation, bus_types):
    """
    Estimate the power of each generator.

    Parameters:
        lambda_value (float): The calculated lambda.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        numpy.ndarray: Array of power values for each generator.
    """
    NB = len(reactive_power_demand) # Ner of generators
    P = sum(real_power_demand)
    Q = sum(reactive_power_demand)
    P_div_Q = float(Q/P)
    t = [bus_types[i] for i in range(NB)]
    # Calculate the reactive power for each generator

    return np.array([P_div_Q * power_generation[i] if t[i] == 'PQ' else 0 for i in range(NB)])

def perform_load_flow(G, B, delta, V, PiG, QiG):
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
    global P, Q  # Declare P and Q as global variables

    # Calculate power for each bus
    Pi = [sum(V[i] * V[k] * (G[i][k] * math.cos(delta[i] - delta[k]) + B[i][k] * math.sin(delta[i] - delta[k])) for k in range(len(V))) for i in range(1, len(V))]
    deltaPiG = [PiG[i] - Pi[i] for i in range(len(Pi))]
    Qi = [sum(V[i] * V[k] * (G[i][k] * math.sin(delta[i] - delta[k]) - B[i][k] * math.cos(delta[i] - delta[k])) for k in range(len(V))) for i in range(1, len(V))]
    deltaQiG = [QiG[i] - Qi[i] for i in range(len(Qi))]
    # Update global P and Q
    P, Q = np.copy(Pi), np.copy(Qi)

    # Step 6
    if max(abs(num) for num in deltaPiG) > tol:
        # Step 7
        H_matrix = build_matrix(Qi, V, NB, delta, "H")
        # Step 8 and Step 9
        delta[1:] = delta[1:] + np.linalg.inv(H_matrix) @ deltaPiG

    # Step 11
    if max(abs(num) for num in deltaQiG) > tol:
        # Step 12
        L_matrix = build_matrix(Qi, V, NB, delta, "L")
        # Step 13
        V[1:] = V[1:] + np.linalg.inv(L_matrix) @ deltaQiG
        # Step 14

    # Check for convergence
    if max(abs(num) for num in deltaPiG + deltaQiG) < tol:
        return delta, V, P, Q, True
    else:
        return delta, V, P, Q, False

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

def calculate_H(a_ij, b_ij, lambda_value, cost_function, P, Q):

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

def power_flow_estimation(a_ij, b_ij, lambda_value, P_inj, Q_inj, PG, QG):
    """
    Function to estimate power flow.
    """
    # Placeholder for the actual implementation

    PL, B_jacobian = build_jacobian_matrix(a_ij, b_ij, P_inj, Q_inj, PG, real_power_demand, lambda_value)

    B_hessian = calculate_H(a_ij, b_ij, lambda_value, cost_function, P_inj, Q_inj)
    S = np.linalg.solve(B_hessian, -B_jacobian)
    loss = 0.0

    # Update my_power and my_lambda
    #[expression_if_true if condition else expression_if_false for item in iterable]
    PG = [PG[i] + S[i] for i in range(NG)]
    QG = [QG[i] + S[i+NG] for i in range(NG)]
    lambda_value += S[2*NG]
    squared_sum = np.sqrt(np.sum(np.square(B_jacobian)))

    P_inj = np.append(PG,[0,0]) - real_power_demand
    Q_inj = np.append(QG,[0,0]) - reactive_power_demand

    #Calculate the new power generation
    if squared_sum < tol:
         return lambda_value, PG, QG, P_inj,Q_inj, True
    else:
        return lambda_value, PG, QG, P_inj,Q_inj, False

# Example usage of read_power_file
def build_jacobian_matrix(a_ij, b_ij, P, Q, my_power, real_power_demand, my_lambda):
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
    my_lambda = float(my_lambda.item())
    pg = -1*np.ones(2*NG+1)

    #  Estimation of the line loss PL
    # Use NumPy's vectorized operations for efficiency
    PL = np.sum(a_ij * (np.outer(P, P) + np.outer(Q, Q)) + b_ij * (np.outer(Q, P) - np.outer(P, Q)))

    # Partial of lagrangian with respect to lambda
    pg[2*NG] = -float(sum(my_power)) + float(sum(real_power_demand)) + float(PL.item())

    # Partial of lagrangian with respect to Pg
    for i in range(NG):
        pg[i] = 2 * cost_function[i][0] * my_power[i] + cost_function[i][1] + my_lambda * (sum((a_ij[i,j] + a_ij[j,i])* P[j] + (b_ij[j,i] - b_ij[i,j]) * Q[j] for j in range(NB)) - 1)

    # Partial of lagrangian with respect to Qg
    for i in range(NG):
        k = i+NG
        pg[k] = my_lambda * (sum((a_ij[i, j] + a_ij[j, i]) * Q[j] + (b_ij[i,j] - b_ij[j,i]) * P[j] for j in range(NB)))

    jacobian_matrix = pg.reshape(-1, 1)

    return PL, jacobian_matrix

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

my_lambda = estimate_lambda(real_power_demand, cost_function)
my_power = estimate_power(my_lambda, cost_function)
my_reactive = estimate_reactive(reactive_power_demand, real_power_demand, my_power, bus_types)

PiG = my_power - real_power_demand
QiG = my_reactive - reactive_power_demand

# Iterative load flow calculation
count = 0
converged = False
while not converged and count < R:
    print(f"Iteration {count}:,delta {np.round([float(x) for x in delta],4)}, Voltage {np.round([float(x) for x in V],4)}")
    delta, V, P, Q, converged = perform_load_flow(G, B, delta, V, PiG[1:], QiG[1:])
    count += 1

# Final power calculations for slack bus
P1 = V[0] * sum(V[k] * (G[0][k] * math.cos(delta[0] - delta[k]) + B[0][k] * math.sin(delta[0] - delta[k])) for k in range(NB))
Q1 = V[0] * sum(V[k] * (G[0][k] * math.sin(delta[0] - delta[k]) - B[0][k] * math.cos(delta[0] - delta[k])) for k in range(NB))

V_com = [V[i]*np.cos(delta[i]) + V[i]*np.sin(delta[i])*1j for i in range(NB)]
P = np.insert(P, 0, P1)
Q = np.insert(Q, 0, Q1)

phi = [np.arctan(Q[i] / P[i]) if P[i] != 0 else 0 for i in range(len(P))]
theta = delta - phi
a_ij, b_ij = calculate_A_B_coefficients(X,V,delta)

# Create the data for the table
data = {
    "Bus": list(range(1, NB + 1)),
    "V": V,
    "delta": delta,
    "P": P,
    "Q": Q,
}

# Create and display the DataFrame
table = pd.DataFrame(data)

data2 = {
    "Bus": list(range(1, NB + 1)),
    "Delta": delta,
    "Phi": phi,
    "Theta": theta,
}
table2 = pd.DataFrame(data2)
print(table2)

PG = my_power
QG = my_reactive
P_inj = PG- real_power_demand
Q_inj = QG - reactive_power_demand

R = 10
count = 0
converged = False
while not converged and count < R:
    print(f"Iteration {count}:,lambda_value {my_lambda}, Power {PG}")
    my_lambda, PG, QG, P_inj, Q_inj, converged = power_flow_estimation(a_ij, b_ij, my_lambda, P_inj, Q_inj, PG, QG)
    count += 1

data3 = {
    "Bus": list(range(1, NB + 1)),
    "V": V,
    "delta": delta,
    "P":  np.append(PG,[0,0]),
    "Q":  np.append(QG,[0,0]),
    "P_injected": P_inj,
    "Q_injected": Q_inj,
}

table3 = pd.DataFrame(data3)
print(table3)
