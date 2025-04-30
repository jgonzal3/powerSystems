import pandas as pd
import numpy as np
from tabulate import tabulate

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

    total_power_loss = sum(
        power_values[i]*power_values[j]*kron_array[i][j] for i in range(len(power_values)) for j in range(len(power_values)))

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
    epsilon = param[1]
    alfa = param[0]
    #print("Matrix M:", M, "\nKron array:", kron_array, "\nParameters:", param)
    Bij = kron_array

    optimal_power = [
        sum(
            (lambda_value * (1 - 2.0 * power_values[j] * Bij[i][j]) - M[i][1]) /
            (2 * (M[i][0] + lambda_value * Bij[i][i]))
            for j in range(len(power_values)) if i != j
        )
        for i in range(len(power_values))
    ]

    # Reshape the optimal power array to match the number of generators
    PL_loss_new = PL(kron_array, np.array(optimal_power))
    PD = PD + PL_loss_new - sum(optimal_power)
    lambda_value = lambda_value + PD * alfa

    # Check for convergence
    print(epsilon)
    if abs(PD) < epsilon:
        return lambda_value, optimal_power, PL_loss_new, True
    else:
        return lambda_value, optimal_power,  PL_loss_new, False

count = 0
converged = False

#Pd = 180
# Example usage
file_path = 'power4.txt'
M, kron_array, param  = read_power_file(file_path)
PD = 160
PD = 150
R=param[2]

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
