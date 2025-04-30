import pandas as pd
import numpy as np
from tabulate import tabulate

def calculate_H(M, Bij,lambda_value, P_gen):
    n = len(Bij)
    H = np.zeros((n+1, n+1))
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][i] = 2*M[i][0] + 2 * lambda_value*Bij[i][i]
            else:
                H[i][j] = 2*lambda_value*Bij[i][j]

    for i in range(n):
        H[i][n] = H[n][i] = sum(2 * Bij[i][j] * P_gen[j] for j in range(n)) - 1

    return H

def calculate_J(M, Bij, lambda_value, P_gen, PD):

    n = len(Bij)
    J = []
    for i in range(n):
        b = 2*M[i][0]*P_gen[i]+M[i][1] + lambda_value*(sum(2*Bij[i][j]*P_gen[j] for j in range(n) )-  1)
        J.append(b)
    P_loss = PL(Bij, P_gen)
    J.append(PD + P_loss - sum(P_gen))

    return np.array(J)


def read_power_file(file_path):
    """
    Reads the file and creates a matrix of size (# of lines x 3).
    Ensures each line has exactly 3 comma-separated elements and the first element is not zero.
    Reads the last line for alfa, epsilon, and iterations.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        tuple: A numpy.ndarray matrix of size (# of lines x 3) and an array [alfa, epsilon, iterations].
    """
    matrix = []
    alfa_epsilon_iterations = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines[:-2], start=1):  # Process all lines except the last
            elements = line.strip().split(',')
            if len(elements) != 5:
                raise ValueError(f"Line {line_number} does not have exactly 5 elements: {line.strip()}")
            if float(elements[0]) == 0:
                raise ValueError(f"Line {line_number} has the first element as zero: {line.strip()}")
            matrix.append(list(map(float, elements[:-2])))

        n2 = len(matrix)*len(matrix)
        n = len(matrix)

        before_last_line = lines[-2].strip().split(',')
        if len(before_last_line) != n2:
            raise ValueError(f"Last line does not have exactly {n} elements: {lines[-1].strip()}")
        kron_array = list(map(float, before_last_line))
        kron_array = np.array(kron_array).reshape(n,n)

        # Process the last line for alfa, epsilon, and iterations
        last_line = lines[-1].strip().split(',')
        if len(last_line) != 3:
            raise ValueError(f"Last line does not have exactly 3 elements: {lines[-1].strip()}")
        alfa_epsilon_iterations = list(map(float, last_line))

    return np.array(matrix), np.array(kron_array), np.array(alfa_epsilon_iterations)

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

    # Calculate numerator and denominator
    numerator = Pd + sum(M[i][1] / (2 * M[i][0]) for i in range(NG))
    denominator = sum(1 / (2 * M[i][0]) for i in range(NG))
    # Calculate lambda
    l = numerator / denominator
    return l

def estimate_power(lambda_value, M):
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
    return power

def PL(kron_array, power_values):
    """
    Calculate the total power loss.

    Parameters:
        kron_array (numpy.ndarray): Array of size (NG x 3).
        power_values (numpy.ndarray): Array of power values for each generator.

    Returns:
        float: Total power loss.
    """

    n = len(kron_array)
    total_power_loss = sum(
        power_values[i]*power_values[j]*kron_array[i][j] for i in range(n) for j in range(n))

    # Kron formula
    #total_power += sum(power_values[i]*kron_array[i][0] for i in range(len(power_values))) + kron_array[0][0]

    return total_power_loss

def optimal_PG(lambda_value, kron_array, power_values):
    """
    Calculate the optimal power generation.

    Parameters:
        lambda_value (float): The calculated lambda.
        kron_array (numpy.ndarray): Array of size (NG x 3).
        power_values (numpy.ndarray): Array of power values for each generator.

    Returns:
        float: Optimal power generation.
    """
    pass

def perform_load_flow(PD,M,kron_array, param, lambda_value,power_values):

    H = calculate_H(M, kron_array, lambda_value, power_values)
    J = calculate_J(M, kron_array, lambda_value, power_values, PD)
    deltas = np.linalg.solve(H,-J)

    save_P = np.copy(power_values)

    n = len(M)
    for i in range(n):
        power_values[i] = power_values[i] + deltas[i]

    lambda_value = lambda_value + deltas[n]
    convergence = (sum(deltas[i]**2 for i in range(n)) + deltas[n]**2)
    PL_loss_new = PL(kron_array, power_values)
    delta_P = abs(-PD - PL_loss_new + np.sum(power_values))
    epsilon = param[1]

    # Check for convergence
    if abs(delta_P) < epsilon:
        return lambda_value, power_values, PL_loss_new, True
    else:
        return lambda_value, power_values,  PL_loss_new, False

def adjust_power_values(array1, P_min, P_max):
    """
    Adjusts power values in the first array based on P_min and P_max constraints
    and updates the violation information.

    Parameters:
        array1 (numpy.ndarray): 2xn array where the first n elements are power values
                                and the last n elements are violation flags (0 or 1).
        P_min (numpy.ndarray): Array of minimum power values (n elements).
        P_max (numpy.ndarray): Array of maximum power values (n elements).

    Returns:
        numpy.ndarray: Updated array1 with adjusted power values and violation flags.
    """
    n = len(P_min)
    power_values = array1[:n]
    violations = array1[n:]

    for i in range(n):
        if power_values[i] < P_min[i]:
            power_values[i] = P_min[i]
            violations[i] = 1
        elif power_values[i] > P_max[i]:
            power_values[i] = P_max[i]
            violations[i] = 1
        else:
            violations[i] = 0 if violations[i] != 1 else violations[i]

    return np.concatenate((power_values, violations))

count = 0
converged = False

#Pd = 180
# Example usage
file_path = 'power4.txt'
M, kron_array, param  = read_power_file(file_path)
PD = 160
#PD = 150
R=param[2]
n = len(M)

P_min = np.array([5, 15, 50])  # Minimum power values
P_max = np.array([150, 100, 250])  # Maximum power values

violations = np.zeros(n)
lambda_value = estimate_lambda(PD, M)
power_opt = estimate_power(lambda_value, M)
loss = 0.0


while not converged and count < R:
    print(f"Iteration {count}:,lambda_value {lambda_value}, Power {[float(x) for x in power_opt]}, Loss {loss}")
    lambda_value, power_opt, loss, converged= perform_load_flow(PD,M,kron_array, param, lambda_value, power_opt)
    count += 1

data = {
    'Iteration': count,
    'Lambda': [lambda_value],
    'P1': [power_opt[0]],
    'P2': [power_opt[1]],
    'P3': [power_opt[2]],

}

# Create the DataFrame
df = pd.DataFrame(data, index=range(1, len(data['Lambda']) + 1))

# Display the table
print(tabulate(df, headers='keys', tablefmt='pipe'))

#adjusted_array = adjust_power_values(power_opt, P_min, P_max)
#print(adjusted_array)
