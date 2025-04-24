import numpy as np
import math
from Algorithm2_1 import get_admitance
import pandas as pd

def build_combined_matrix(H, J, N, L):
    """
    Constructs a combined matrix from H, J, N, and L.

    Parameters:
        H (numpy.ndarray): Top-left block matrix.
        N (numpy.ndarray): Top-right block matrix.
        J (numpy.ndarray): Bottom-left block matrix.
        L (numpy.ndarray): Bottom-right block matrix.

    Returns:
        numpy.ndarray: Combined matrix of size (2NB-1) x (2NB-1).
    """
    # Stack H and N horizontally
    top = np.hstack((H, N))

    # Stack J and L horizontally
    bottom = np.hstack((J, L))

    # Stack the top and bottom vertically
    combined_matrix = np.vstack((top, bottom))

    return combined_matrix



def get_line_data(file_path):
    """
    Reads the file and extracts data for slack bus, PQ buses, and PV buses.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
        tuple: Contains:
            - slack_voltage (float): Voltage of the slack bus.
            - slack_angle (float): Angle of the slack bus.
            - PiS (list): Array of active power for PQ and PV buses.
            - QiS (list): Array of reactive power for PV buses.
            - ViS (list): Array of voltage magnitudes for PV buses.
    """
    PiS = []
    QiS = []
    ViS = []

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Read the first line (NB and NV)
        first_line = lines[0].strip().split(',')
        NB = int(first_line[0])  # Number of buses
        NV = int(first_line[1])  # Number of PV buses
        R = int(first_line[2])  # Number of iterations
        tol = float(first_line[3])  # Tolerance

        # Read the second line (slack bus)
        slack_line = lines[1].strip().split(',')
        slack_voltage = float(slack_line[1])  # Voltage of slack bus
        slack_angle = float(slack_line[2])  # Angle of slack bus

        # Process the remaining lines
        for line in lines[2:]:
            fields = line.strip().split(',')
            bus_type = fields[-1].strip()  # Last field indicates bus type (PQ or PV)

            if bus_type in ['PQ', 'PV']:
                PiS.append(complex(fields[1]))  # Active power (2nd field)

            if bus_type == 'PV':
                QiS.append(float(fields[2]))  # Reactive power (3rd field)
                ViS.append(float(fields[3]))  # Voltage magnitude (4th field)

            if bus_type in ['PQ']:
                QiS.append(complex(fields[1]))  # Active power (2nd field)

    return slack_voltage, slack_angle, np.real(PiS),np.imag(PiS),ViS,R, tol, NB

def build_new_matrix(P,Q,VN,NB, deltaN):

    H, J, N, L = [np.zeros((NB-1, NB-1)) for _ in range(4)]
    VN = VN[1:]

    for i, k in [(i, k) for i in range(NB - 1) for k in range(NB - 1)]:
        j, l = i + 1, k + 1
        if i == k:
            H[i][i] = -Q[i] - B[j][j] * VN[i] ** 2
            J[i][i] = P[i] - G[j][j] * VN[i] ** 2
            N[i][i] = P[i] + G[j][j] * VN[i] ** 2
            L[i][i] = Q[i] - B[j][j] * VN[i] ** 2
        else:
            H[i][k] = VN[i]*VN[k]*(G[j][l] * math.sin(deltaN[i] - deltaN[k]) - B[j][l] * math.cos(deltaN[i] - deltaN[k]))
            J[i][k] = VN[i]*VN[k]*(-G[j][l] * math.cos(deltaN[i] + deltaN[k]) - B[j][l] * math.sin(deltaN[i] - deltaN[k]))
            N[i][k] = VN[i]*VN[k]*(G[j][l] * math.cos(deltaN[i] - deltaN[k]) + B[j][l] * math.sin(deltaN[i] - deltaN[k]))
            L[i][k] = VN[i]*VN[k]*(-G[j][l] * math.sin(deltaN[i] - deltaN[k]) - B[j][l] * math.cos(deltaN[i] - deltaN[k]))
    # Build the combined matrix
    return build_combined_matrix(H, J, N, L)


def perform_load_flow(G,B, deltaN, VN):
    """
    Performs load flow analysis using the Gauss-Seidel method.

    Parameters:
        G (numpy.ndarray): Conductance matrix.
        B (numpy.ndarray): Susceptance matrix.
        R (int): Number of iterations.
        tol (float): Tolerance for convergence.
    """

    # Initialize power vector
    Pi = []
    Qi = []

    global P, Q  # Declare P and Q as global variables
    Pi = [sum(
        VN[i] * VN[k] * (G[i][k] * math.cos(deltaN[i] - deltaN[k]) + B[i][k] * math.sin(deltaN[i] - deltaN[k])) for k in
        range(len(VN))) for i in range(1, len(VN))]
    Qi = [sum(
        VN[i] * VN[k] * (G[i][k] * math.sin(deltaN[i] - deltaN[k]) - B[i][k] * math.cos(deltaN[i] - deltaN[k])) for k in
        range(len(VN))) for i in range(1, len(VN))]

    # Calculate power for each bus
    deltaPiS = [PiS[i]-Pi[i] for i in range(len(Pi))]
    deltaQiS = [QiS[i]-Qi[i] for i in range(len(Qi))]
    column_vector = np.array(deltaPiS + deltaQiS).reshape(-1, 1)

    P,Q = np.copy(Pi), np.copy(Qi)
    combined_matrix = build_new_matrix(Pi, Qi, V, NB, deltaN)
    inverse_matrix = np.linalg.inv(combined_matrix)
    solution = inverse_matrix@column_vector

    # Solve the system of equations
    return 0 if max([abs(num) for num in deltaPiS+deltaQiS]) < tol else solution


file_path = 'data_2.txt'
slack_voltage, slack_angle, PiS, QiS, ViS, R, tol, NB = get_line_data(file_path)
print("Slack Voltage:", slack_voltage)
print("Slack Angle:", slack_angle)
print("PiS:", PiS)
print("QiS:", QiS)
print("ViS:", ViS)
print("R:", R)
print("Tolerance:", tol)
print("NB:", NB)

# Initialize voltage vector
V = np.ones(NB)  # Initial guess
V[0] = slack_voltage  # Slack bus voltage
# Initialize delta vector
delta = np.zeros(NB)
delta_old = np.copy(delta)
V_old = np.copy(V)

Y = get_admitance()
G = np.real(Y)  # Conductance matrix
B = np.imag(Y)  # Susceptance matrix
# Initialize power vector
P = []
Q = []

count = 0
result = perform_load_flow(G,B, delta, V)

while max(abs(num) for num in result) >= tol and count < R:
    delta = delta_old[1:] + np.transpose(result[:NB-1])
    V = V_old[1:] + np.transpose(result[NB-1:])
    V = np.insert(V,0,slack_voltage)
    delta = np.insert(delta,0,slack_angle)
    result = perform_load_flow(G, B, delta, V)
    delta_old = np.copy(delta)
    V_old = np.copy(V)
    count += 1
    print("Iteration:", count, "V:", V, "delta:", delta)

    # Exit the loop if the condition is not met
if count >= R:
    print("Maximum iterations reached.")
elif max(abs(num) for num in result) < tol:
    print("Convergence achieved.")

i=0
P1 = V[i]*sum([abs(V[k])*(G[i][k] * math.cos(delta[i] - delta[k]) + B[i][k] * math.sin(delta[i] - delta[k])) for k in range(NB)])
Q1 = V[i]*sum([abs(V[k])*(G[i][k] * math.sin(delta[i] - delta[k]) - B[i][k] * math.cos(delta[i] - delta[k])) for k in range(NB)])

# Create the data for the table
data = {
    "Bus": list(range(1, NB + 1)),
    "PS": np.insert(PiS,0,0),
    "QS": np.insert(QiS,0,0),
    "P": np.insert(P,0,P1),
    "Q": np.insert(Q,0,Q1),
    "V": V,
    "Angle": delta
}
# Create a DataFrame
table = pd.DataFrame(data)
# Corrected styling for the table
print(table)


