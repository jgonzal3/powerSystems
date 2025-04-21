import re
import numpy as np

# Read the file and initialize the matrix with error handling
def initialize_matrix(file_path):
    try:
        with open(file_path, 'r') as file:
            # Skip lines starting with #
            first_line = None
            for line in file:
                line = line.strip()
                if not line.startswith('#'):
                    first_line = line
                    break

            # Check if a valid first line was found
            if first_line is None:
                raise ValueError("File does not contain valid data.")

            # Read the first line and extract NL and NB
            NL, NB = map(int, first_line.split())

            # Initialize the Zij matrix with zeros
            Zij = [[0 for _ in range(NB)] for _ in range(NB)]
        return NL, NB, Zij

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        raise
    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

# Read the next NL lines with error handling
def read_next_lines(file_path, NL):
    try:
        with open(file_path, 'r') as file:
            # Skip lines starting with #
            lines = []
            for line in file:
                line = line.strip()
                if not line.startswith('#'):
                    lines.append(line)
            next_lines = lines[1:]
        return next_lines

    except ValueError as e:
        print(f"Error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
# Identify and process the type (1 to 4) using match-case
import re

def process_type_with_match(lines, Zij):
    Mij = np.zeros((NB + 1, NB + 1))
    for line in lines:
        entries = line.split(',')
        if len(entries) >= 4:
            type_value = int(entries[3])
            if int(entries[4]) != 0:
                Ypq = np.linalg.inv( [[float(entries[2]), float(entries[5])],[float(entries[5]), float(entries[6])]])
                Mij = process_type_with_coupling(Ypq, Zij,entries)
            Zb = float(entries[2])  # Ensure Zb is numeric
            match type_value:
                case 1:
                    if re.match(r'R-\d+', entries[1]):  # Validate format
                        index = int(entries[1].split('-')[1])-1 # Convert index to integer
                        for i in range(index):
                            Zij[index][i] = 0.0
                            Zij[i][index] = 0.0
                        Zij[index][index] = Zb
                case 2:
                    if re.match(r'\d-\d+', entries[1]):  # Validate format
                        index = int(entries[1].split('-')[1]) -1  # Convert index to integer
                        old_index = int(entries[1].split('-')[0])-1  # Convert index to integer
                        for i in range(index):
                            Zij[index][i] = Zij[old_index][i]
                            Zij[i][index] = Zij[old_index][i]
                        Zij[index][index] = Zb + Zij[old_index][old_index]
                case 3:
                    print(f"Processing Type 3: {line}")
                    Pij = [[0 for _ in range(NB)] for _ in range(NB)]
                    if re.match(r'\d-\d+', entries[1]):  # Validate format
                        old_bus_k = int(entries[1].split('-')[0]) - 1  # Convert index to integer
                        P = Zb + Zij[old_bus_k][old_bus_k]
                        for i in range(NB):
                            for j in range(NB):
                                Pij[i][j] = Zij[i][j] - (1 / P) * Zij[i][old_bus_k]*Zij[old_bus_k][j]
                        for i in range(len(Pij)):
                            for j in range(len(Pij[i])):
                                Zij[i][j] = Pij[i][j]
                case 4:
                    Qij = [[0 for _ in range(NB)] for _ in range(NB)]
                    if re.match(r'\d-\d+', entries[1]):  # Validate format
                        old_bus_k = int(entries[1].split('-')[0])-1  # Convert index to integer
                        old_bus_l = int(entries[1].split('-')[1])-1  # Convert index to integer
                        P = Zb + Zij[old_bus_k][old_bus_k] + Zij[old_bus_l][old_bus_l] - Zij[old_bus_k][old_bus_l] - Zij[old_bus_l][old_bus_k]
                        for i in range(NB):
                             for j in range(NB):
                                 first  = (1/P)*(Zij[i][old_bus_k]- Zij[i][old_bus_l])* (Zij[old_bus_k][j] -Zij[old_bus_l][j])
                                 Qij[i][j] = Zij[i][j] - first
                        for i in range(len(Qij)):
                            for j in range(len(Qij[i])):
                                Zij[i][j] = Qij[i][j]
                case _:
                    print(f"Invalid Type: {line}")
    return Mij if 'Mij' in locals() else Zij

def augment_matrix(Zij):
    NB = len(Zij)  # Size of the Zij matrix
    # Create an augmented matrix Mij with one extra row and column
    Mij = np.zeros((NB + 1, NB + 1))
    # Copy Zij into Mij
    for i in range(NB):
        for j in range(NB):
            Mij[i][j] = Zij[i][j]
    return Mij

def process_type_with_coupling(Ypq, Zij,entries):
    p, q = (int(x) - 1 for x in entries[7].split('-'))
    r, s = (int(x) - 1 for x in entries[8].split('-'))
    Mij = augment_matrix(Zij)
    l = NB
    for k in range(NB):
        Mij[l][k] = Zij[p][k] - Zij[q][k]  + (Ypq[0][1]/Ypq[1][1])*(Zij[r][k] - Zij[s][k])
        Mij[k][l] = Mij[l][k]
    Mij[l][l] = Mij[p][l] - Mij[q][l]  + (1 + Ypq[0][1]*(Mij[r][l] - Mij[s][l]))/Ypq[1][1]
    return Mij

def eliminate_node(M,k):
    k = k-1
    Qij = np.zeros((NB, NB))
    for i in range(NB):
        for j in range(NB):
            Qij[i][j] = M[i][j] - M[i][k]*M[k][j]/M[k][k]
    return Qij


# Example usage
if __name__ == '__main__':
    file_path = 'input.txt'  # Replace with your file path
    try:
        NL, NB, Zij = initialize_matrix(file_path)
        next_lines = read_next_lines(file_path, NL+1)
        print(f"NL: {NL}, NB: {NB}")
        Solution = process_type_with_match(next_lines, Zij)
        Solution1 =  eliminate_node(Solution, 4)
        print(f"Solution1: {Solution1}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except ValueError as e:
        print(f"Value error: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
