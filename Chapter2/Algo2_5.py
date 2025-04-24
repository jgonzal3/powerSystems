import numpy as np
import math
import pandas as pd
from Algorithm2_1 import get_admitance

def get_line_data(file_path):
    """
    Reads the file and extracts data for slack bus, PQ buses, and PV buses.

    Parameters:
        file_path (str): Path to the input file.

    Returns:
    tuple: A tuple containing the following elements:
        - slack_voltage (float): Voltage of the slack bus.
        - slack_angle (float): Angle of the slack bus in radians.
        - PiS (numpy.ndarray): Array of real parts of active power for PQ and PV buses.
        - QiS (numpy.ndarray): Array of imaginary parts of active power for PQ buses.
        - ViS (list): Array of voltage magnitudes for PV buses.
        - R (int): Number of iterations for the load flow calculation.
        - tol (float): Tolerance for convergence.
        - NB (int): Total number of buses in the system.
        - is_pq (numpy.ndarray): Boolean array indicating whether each bus is of type PQ (True) or PV (False).
        - V_max (float): Maximum allowable voltage magnitude.
        - V_min (float): Minimum allowable voltage magnitude.
        - Q_max (float): Maximum allowable reactive power.
        - Q_min (float): Minimum allowable reactive power.
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
        V_max = float(first_line[4]) # V max
        V_min = float(first_line[5]) # V min
        Q_max = float(first_line[6]) # Q max
        Q_min = float(first_line[7]) # Q min

        # Read the second line (slack bus)
        slack_line = lines[1].strip().split(',')
        slack_voltage = float(slack_line[1])  # Voltage of slack bus
        slack_angle = float(slack_line[2])  # Angle of slack bus

        bus_types = [line.strip().split(',')[-1].strip() for line in lines[2:]]  # Extract bus types
        is_pq = np.array([bus_type == 'PQ' for bus_type in bus_types])  # True for PQ, False for PV

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

    return slack_voltage, slack_angle, np.real(PiS),np.imag(PiS),ViS,R, tol, NB, is_pq, V_max, V_min,Q_max, Q_min

def build_matrix(Q, VN, NB, deltaN, matrix_type):

    H, L = [np.zeros((NB-1, NB-1)) for _ in range(2)]
    VN = VN[1:]

    for i, k in [(i, k) for i in range(NB - 1) for k in range(NB - 1)]:
        j, l = i + 1, k + 1
        if i == k:
            H[i][i] = -Q[i] - B[j][j] * VN[i] ** 2
            L[i][i] = Q[i] - B[j][j] * VN[i] ** 2
        else:
            H[i][k] = VN[i]*VN[k]*(G[j][l] * math.sin(deltaN[i] - deltaN[k]) - B[j][l] * math.cos(deltaN[i] - deltaN[k]))
            L[i][k] = VN[i]*VN[k]*(-G[j][l] * math.sin(deltaN[i] - deltaN[k]) - B[j][l] * math.cos(deltaN[i] - deltaN[k]))
    # Build the combined matrix
    return H if matrix_type == "H" else L

def perform_load_flow(G, B, deltaN, VN, PiS, QiS):
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
    Pi = [sum(VN[i] * VN[k] * (G[i][k] * math.cos(deltaN[i] - deltaN[k]) + B[i][k] * math.sin(deltaN[i] - deltaN[k])) for k in range(len(VN))) for i in range(1, len(VN))]
    deltaPiS = [PiS[i] - Pi[i] for i in range(len(Pi))]
    Qi = [sum(VN[i] * VN[k] * (G[i][k] * math.sin(deltaN[i] - deltaN[k]) - B[i][k] * math.cos(deltaN[i] - deltaN[k])) for k in range(len(VN))) for i in range(1, len(VN))]
    deltaQiS = [QiS[i] - Qi[i] for i in range(len(Qi))]
    # Update global P and Q
    P, Q = np.copy(Pi), np.copy(Qi)

    # Step 6
    if max(abs(num) for num in deltaPiS) > tol:
        # Step 7
        H_matrix = build_matrix(Qi, VN, NB, deltaN, "H")
        # Step 8 and Step 9
        deltaN[1:] = deltaN[1:] + np.linalg.inv(H_matrix) @ deltaPiS

    # Step 11
    if max(abs(num) for num in deltaQiS) > tol:
        # Step 12
        L_matrix = build_matrix(Qi, VN, NB, deltaN, "L")
        # Step 13
        VN[1:] = VN[1:] + np.linalg.inv(L_matrix) @ deltaQiS
        # Step 14

    # Iterate through each bus
    for i in range(len(is_pq)):
        if is_pq[i]:  # Check if the bus is PQ
            if VN[i+1] <= V_min:
                VN[i+1] = V_min
            elif VN[i+1] > V_max:
                VN[i+1] = V_max
        else:
            print("Bus is PV")
            if Qi[i] <= Q_min:
                Qi[i] = Q_min
            elif VN[i] > Q_max:
                Qi[i] = Q_max


    # Check for convergence
    if max(abs(num) for num in deltaPiS + deltaQiS) < tol:
        return deltaN, VN, True
    else:
        return deltaN, VN, False

# Main execution
file_path = 'data_3.txt'
slack_voltage, slack_angle, PiS, QiS, ViS, R, tol, NB, is_pq, V_max, V_min,Q_max, Q_min = get_line_data(file_path)

# Initialize voltage and angle vectors
V = np.ones(NB)
V[0] = slack_voltage
delta = np.zeros(NB)
delta[0] = slack_angle

Y = get_admitance()
G, B = np.real(Y), np.imag(Y)

# Iterative load flow calculation
count = 0
converged = False
while not converged and count < R:
    delta, V, converged = perform_load_flow(G, B, delta, V, PiS, QiS)
    count += 1

# Final power calculations for slack bus
P1 = V[0] * sum(V[k] * (G[0][k] * math.cos(delta[0] - delta[k]) + B[0][k] * math.sin(delta[0] - delta[k])) for k in range(NB))
Q1 = V[0] * sum(V[k] * (G[0][k] * math.sin(delta[0] - delta[k]) - B[0][k] * math.cos(delta[0] - delta[k])) for k in range(NB))

# Create the data for the table
data = {
    "Bus": list(range(1, NB + 1)),
    "PS": np.insert(PiS, 0, 0),
    "QS": np.insert(QiS, 0, 0),
    "P": np.insert(P, 0, P1),
    "Q": np.insert(Q, 0, Q1),
    "V": V,
    "Angle": delta
}

# Create and display the DataFrame
table = pd.DataFrame(data)
print(table)
