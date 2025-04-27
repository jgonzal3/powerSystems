import pandas as pd
import numpy as np

def read_power_file(file_path):
    """
    Reads the file and creates a matrix of size (# of lines x 3).
    Ensures each line has exactly 3 comma-separated elements and the first element is not zero.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        numpy.ndarray: Matrix of size (# of lines x 3).
    """
    matrix = []
    with open(file_path, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            elements = line.strip().split(',')
            if len(elements) != 3:
                raise ValueError(f"Line {line_number} does not have exactly 3 elements: {line.strip()}")
            if float(elements[0]) == 0:
                raise ValueError(f"Line {line_number} has the first element as zero: {line.strip()}")
            matrix.append(list(map(float, elements)))
    return np.array(matrix)

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

#Pd = 180
# Example usage
file_path = 'power.txt'
l = []
g = []
demand = [80, 120, 180]
try:
    for Pd in demand:
        M = read_power_file(file_path)
        lambda_value = estimate_lambda(Pd, M)
        l.append(lambda_value)
        power_values = estimate_power(lambda_value, M)
        g.append(power_values)
except ValueError as e:
    print(e)

# Estimate power for each generator
# Convert g (list of generator arrays) to a DataFrame
data = {
    'Lambda': l,
    'P1': [gen[0] for gen in g],
    'P2': [gen[1] for gen in g]
}

# Create the DataFrame
df = pd.DataFrame(data, index=range(1, len(data['Lambda']) + 1))

# Display the table
print(df)

      Lambda         P1          P2
1  23.636364  16.363636   63.636364
2  25.818182  38.181818   81.818182
3  29.090909  70.909091  109.090909
