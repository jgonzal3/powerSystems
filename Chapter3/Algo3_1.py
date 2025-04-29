import pandas as pd
import numpy as np
from tabulate import tabulate

def kron_to_matrix(kron_array):
    """
    Converts a Kron array to a matrix of size (NG x 3).

    Parameters:
        kron_array (numpy.ndarray): Array of size (NG x 3).

    Returns:
        numpy.ndarray: Matrix of size (NG x 3).
    """
    kron_matrix = np.array([[kron_array[0], kron_array[2],0],[kron_array[2],kron_array[1],0],[0,0,kron_array[1]]])
    return kron_matrix


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
            if len(elements) != 3:
                raise ValueError(f"Line {line_number} does not have exactly 3 elements: {line.strip()}")
            if float(elements[0]) == 0:
                raise ValueError(f"Line {line_number} has the first element as zero: {line.strip()}")
            matrix.append(list(map(float, elements)))

        before_last_line = lines[-2].strip().split(',')
        if len(before_last_line) != 3:
            raise ValueError(f"Last line does not have exactly 3 elements: {lines[-1].strip()}")
        kron_array = list(map(float, before_last_line))

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
    total_power_loss = kron_array[0] * power_values[0]**2 + kron_array[1] * power_values[1]**2 + 2*kron_array[2]*power_values[0]*power_values[1]
    return total_power_loss

def perform_load_flow(PD,M,kron_array, param, lambda_value,power_values):
    epsilon = param[1]
    alfa = param[0]
    km = kron_to_matrix(kron_array)

    optimal_power = [
        (lambda_value * (1 - 2.0 * power_values[j] * km[i][j]) - M[i][1]) /
        (2 * (M[i][0] + lambda_value * km[i][i]))
        for i in range(len(power_values))
        for j in range(len(power_values))
        if i != j
    ]

    # Reshape the optimal power array to match the number of generators
    PL_new = PL(kron_array, np.array(optimal_power))
    PD = PD + PL_new - sum(optimal_power)
    lambda_value = lambda_value + PD * alfa

    # Check for convergence
    if abs(PD) < epsilon:
        return lambda_value, optimal_power, PL_new, True
    else:
        return lambda_value, optimal_power,  PL_new, False

count = 0
converged = False
# Example usage
file_path = 'power.txt'
M, kron_array, param  = read_power_file(file_path)
PD = 150
R=param[2]

# Initialise the algorithm
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
    'P2': [power_opt[1]]
}

# Create the DataFrame
df = pd.DataFrame(data, index=range(1, len(data['Lambda']) + 1))

# Display the table
print(tabulate(df, headers='keys', tablefmt='pipe'))
