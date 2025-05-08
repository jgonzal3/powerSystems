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
    Pd = sum(Pd)  # Total power demand
    # Calculate numerator and denominator
    numerator = Pd + sum(M[i][1] / (2 * M[i][0]) for i in range(NG))
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
        return delta, V, True
    else:
        return delta, V, False

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


def build_hessian_matrix(B_ij_matrix, my_lambda, cost_function, P):
    """
    Builds the Hessian matrix using the provided parameters.

    Parameters:
        B_ij_matrix (numpy.ndarray): Matrix of size NG x NG.
        my_lambda (float): Lambda value.
        cost_function (numpy.ndarray): Cost function matrix of size NG x 3.
        my_power (numpy.ndarray): Power array of size NG.
        NG (int): Number of generators.

    Returns:
        numpy.ndarray: The resulting Hessian matrix.
    """
    # Build g1 matrix
    g1 = np.zeros((NG, NG))
    g2= np.zeros(NG)
    for i in range(NG):
        for j in range(NG):
            if i == j:
                g1[i][j] = 2 * cost_function[i][0] + 2 * float(my_lambda.item())* B_ij_matrix[i][j]
            else:
                g1[i][j] = 2 * float(my_lambda.item())* B_ij_matrix[i][j]

    for i in range(NG):
        g2[i] = sum(2 * B_ij_matrix[i][j] * P[j] for j in range(NB)) - 1

    g3 = g2.reshape(-1, 1)
    g3 = (np.insert(g3, NG, 0)).reshape(-1,1)

    A = np.vstack((g1, g2))
    B = np.hstack((A, g3))

    return B

def build_jacobian_matrix(B_matrix, P, my_power, real_power_demand, cost_function, my_lambda):
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
    # Calculate PL
    PL = sum(B_matrix[i][j] * P[i] * P[j] for i in range(NB) for j in range(NB))

    # Initialize pg array
    pg = np.zeros(NG + 1)
    for i in range(NG):
        pg[i] = (2 * cost_function[i][0] * my_power[i] + cost_function[i][1] +
                 float(my_lambda.item() * (sum(2 * B_matrix[j][i] * P[j] for j in range(NB)) - 1)))

    # Notice that the different types of power used to calculate PL and pg
    # With PL, I used the power calculated, P. But to estimate the jacobian of lambda, I used
    # both powers, specifically to estimate the generator losses. I cannot use the estimated.
    # Add the last element to pg
    pg[NG] = -float(sum(my_power)) + float(sum(real_power_demand) ) + float(PL.item())

    # Reshape pg into a column vector
    jacobian_matrix = pg.reshape(-1, 1)

    return PL, jacobian_matrix

def calculate_B_coefficients(X,V,phi,theta):
    """
    Calculate the B coefficients for the optimization problem.

    Parameters:
        B_matrix (numpy.ndarray): Matrix of size NG x NG.
        power_values (numpy.ndarray): Power array of size NG.

    Returns:
        numpy.ndarray: The B coefficients.
    """
    # Calculate the B coefficients
    B_ij = [(X[i][j] * np.cos(theta[i] - theta[j])) / (V[i] * V[j] * np.cos(phi[i]) * np.cos(phi[j])) for i in range(NB)
            for j in range(NB)]

    B_coefficients = np.array(B_ij).reshape(NB, NB)

    return B_coefficients

def power_flow_estimation(B_matrix, lambda_value, cost_function, P, power_values, real_power_demand):
    """
    Function to estimate power flow.
    """
    # Placeholder for the actual implementation

    B_hessian = build_hessian_matrix(B_matrix, lambda_value, cost_function, P)
    loss, B_jacobian = build_jacobian_matrix(B_matrix, P, power_values, real_power_demand, cost_function, lambda_value)
    S = np.linalg.solve(B_hessian, -B_jacobian)

    # Update my_power and my_lambda
    #[expression_if_true if condition else expression_if_false for item in iterable]
    power_values = [power_values[i] + S[i] for i in range(NG)]
    lambda_value += S[NG]
    squared_sum = np.sqrt(np.sum(np.square(B_jacobian)))

    #Calculate the new power generation
    if squared_sum < tol:
         return lambda_value, power_values, loss, True
    else:
        return lambda_value, power_values, loss, False


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
    delta, V, converged = perform_load_flow(G, B, delta, V, PiG[1:], QiG[1:])
    count += 1

# Final power calculations for slack bus
P1 = V[0] * sum(V[k] * (G[0][k] * math.cos(delta[0] - delta[k]) + B[0][k] * math.sin(delta[0] - delta[k])) for k in range(NB))
Q1 = V[0] * sum(V[k] * (G[0][k] * math.sin(delta[0] - delta[k]) - B[0][k] * math.cos(delta[0] - delta[k])) for k in range(NB))

V_com = [V[i]*np.cos(delta[i]) + V[i]*np.sin(delta[i])*1j for i in range(NB)]
P = np.insert(P, 0, P1)
Q = np.insert(Q, 0, Q1)

phi = [np.arctan(Q[i] / P[i]) if P[i] != 0 else 0 for i in range(len(P))]
theta = delta - phi
B_matrix = calculate_B_coefficients(X,V,phi,theta)

# Create the data for the table
data = {
    "Bus": list(range(1, NB + 1)),
    "V": V,
    "delta": delta,
    "P": P,
    "Q": Q
}

# Create and display the DataFrame
table = pd.DataFrame(data)
print(table)

data2 = {
    "Bus": list(range(1, NB + 1)),
    "Delta": delta,
    "Phi": phi,
    "Theta": theta,
}
table2 = pd.DataFrame(data2)
print(table2)

R = 10
count = 0
converged = False
while not converged and count < R:
    print(f"Iteration {count}:,lambda_value {my_lambda}, Power {my_power}")
    my_lambda, my_power, loss, converged = power_flow_estimation(B_matrix, my_lambda, cost_function, P, my_power, real_power_demand)
    count += 1

print (my_lambda)

print(f"Lambda value {my_lambda}, Loss {loss}")
data3 = {
    "Bus": list(range(1, NB+1)),
    "Pg": np.append(my_power,[0,0]),
    "Qg": my_reactive,
    "Pd": real_power_demand,
    "Qd": reactive_power_demand
}
table3 = pd.DataFrame(data3)
print(table3)
