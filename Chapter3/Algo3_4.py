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

def calculate_PL(B, power_values):
    """
    Calculate the total power loss.

    Parameters:
        kron_array (numpy.ndarray): Array of size (NG x 3).
        power_values (numpy.ndarray): Array of power values for each generator.

    Returns:
        float: Total power loss.
    """

    n = len(B)
    total_power_loss = sum(
        power_values[i]*power_values[j]*B[i][j] for i in range(n) for j in range(n))

    return total_power_loss

def calculate_K(B,power_values):
    return [2 * sum(B[i][j] * power_values[j] for j in range(len(B))) for i in range(len(B))]

def calculate_X(M,power_values):
    return [2*M[i][0]*power_values[i] + M[i][1] for i in range(len(M))]

def calculate_Y(B,lambda_value):
   return [2*M[i][0] + 2*lambda_value*B[i][i] for i in range(len(M))]

def perform_load_flow(PD,M,Bij, param, lambda_value,power_values):
    """
    Performs a single iteration of the load flow optimization process.

    Parameters:
        PD (float): Power demand.
        M (numpy.ndarray): Matrix of generator parameters of size (NG x 3),
                           where each row contains [a, b, c] coefficients for a generator.
        Bij (numpy.ndarray): Kron's loss coefficient matrix of size (NG x NG).
        param (numpy.ndarray): Array containing [alpha, epsilon, max_iterations].
        lambda_value (float): Current value of lambda used in the optimization process.
        power_values (numpy.ndarray): Array of current power values for each generator.

    Returns:
        tuple: A tuple containing:
            - lambda_value (float): Updated lambda value.
            - power_values (numpy.ndarray): Updated power values for each generator.
            - PL_loss_new (float): Total power loss after the iteration.
            - converged (bool): True if the solution has converged, False otherwise.

    Raises:
        ValueError: If there are issues with matrix dimensions or convergence criteria.
    """

    K = calculate_K(Bij,power_values)
    PL = calculate_PL(Bij,power_values)
    PD = PD + PL - sum(power_values)
    X = calculate_X(M, power_opt)
    Y = calculate_Y(Bij,lambda_value)
    lambda_new = (PD + sum(((1-K[j])*X[j])/Y[j] for j in range(len(Bij))))/(sum( ((1 - K[j])**2)/(Y[j]) for j in range(len(Bij))))
    delta_P = [((1-K[i])*lambda_new - X[i])/Y[i] for i in range(len(Bij))]

    epsilon = param[1]

    # Check for convergence
    if abs(lambda_value - lambda_new) < epsilon:
        return lambda_value, power_values, PL ,True
    else:
        power_values = power_values + delta_P
        return lambda_new, power_values,  PL, False


count = 0
converged = False

#Pd = 180
# Example usage
file_path = 'power5.txt'
M, kron_array, param  = read_power_file(file_path)
PD = 160
PD = 150
R=param[2]
n = len(M)

P_min = np.array([5, 15, 50])  # Minimum power values
P_max = np.array([150, 100, 250])  # Maximum power values

lambda_value = estimate_lambda(PD, M)
power_opt = estimate_power(lambda_value, M)
loss = 0.0


while not converged and count < R:
    print(f"Iteration {count}:,lambda_value {lambda_value}, Power {[float(x) for x in power_opt]}, Loss {loss}")
    lambda_value, power_opt, loss, converged= perform_load_flow(PD,M,kron_array, param, lambda_value, power_opt)
    count += 1

delta_P = PD + loss - np.sum([power_opt])
data = {
    'Iteration': count,
    'Lambda': [lambda_value],
    'P1': [power_opt[0]],
    'P2': [power_opt[1]],
    'DeltaP': delta_P,
    #'P3': [power_opt[2]],

}

# Create the DataFrame
df = pd.DataFrame(data, index=range(1, len(data['Lambda']) + 1))

# Display the table
print(tabulate(df, headers='keys', tablefmt='pipe'))

#adjusted_array = adjust_power_values(power_opt, P_min, P_max)
#print(adjusted_array)
