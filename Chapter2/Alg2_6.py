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

    Returns:
        tuple: Updated deltaN, VN, and convergence status.
    """
    global P, Q  # Declare P and Q as global variables

    # Calculate power for each bus
    B_prime = -B[1:, 1:]
    Pi = [sum(
        VN[i] * VN[k] * (G[i][k] * math.cos(deltaN[i] - deltaN[k]) + B[i][k] * math.sin(deltaN[i] - deltaN[k])) for k in
        range(len(VN))) for i in range(1, len(VN))]
    deltaPiS = [PiS[i] - Pi[i] for i in range(len(Pi))]
    deltaN[1:] = deltaN[1:] + np.linalg.inv(B_prime) @ (deltaPiS / VN[1:])
    Qi = [sum(
        VN[i] * VN[k] * (G[i][k] * math.sin(deltaN[i] - deltaN[k]) - B[i][k] * math.cos(deltaN[i] - deltaN[k])) for k in
        range(len(VN))) for i in range(1, len(VN))]
    deltaQiS = [QiS[i] - Qi[i] for i in range(len(Qi))]
    VN[1:] = VN[1:] + np.linalg.inv(B_prime) @ (deltaQiS / VN[1:])
    # Update global P and Q
    P, Q = np.copy(Pi), np.copy(Qi)

    # Check for convergence
    if max(abs(num) for num in deltaPiS + deltaQiS) < tol:
        return deltaN, VN, True
    else:
        return deltaN, VN, False

def calculate_line_flows(V, delta, Y):
    """
    Calculate the line flows Sij and Sji.

    Parameters:
        V (numpy.ndarray): Voltage magnitudes at each bus.
        delta (numpy.ndarray): Voltage angles at each bus (in radians).
        Y (numpy.ndarray): Admittance matrix.

    Returns:
        tuple: Two matrices Sij and Sji representing the line flows.
    """
    NB = len(V)  # Number of buses
    Sij = np.zeros((NB, NB), dtype=complex)
    Sji = np.zeros((NB, NB), dtype=complex)

    # Convert V and delta to complex voltage phasors
    V_phasor = V *(np.cos(delta) +  1j * np.sin(delta))

    for i in range(NB):
        for j in range(NB):
            if i != j and Y[i, j] != 0:  # Only consider connected buses
                I_ij = Y[i, j] * (V_phasor[i] - V_phasor[j]) # Current from i to j
                Sij[i, j] = V_phasor[i] * np.conj(I_ij)       # Power flow from i to j
                Sji[j, i] = V_phasor[j] * np.conj(-I_ij)      # Power flow from j to i

    return Sij, Sji

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
    print("Iteration:", count + 1, "V:", V, "delta:", delta)
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

Sij, Sji = calculate_line_flows(V, delta, Y)

# Print the results

data = []
for i in range(Sij.shape[0]):
    for j in range(Sij.shape[1]):
        if i != j and Y[i, j] != 0:  # Only include connected buses
            data.append({
                "From Bus": i + 1,
                "To Bus": j + 1,
                "Sij (Power from i to j)": Sij[i, j],
            })

# Create the DataFrame
line_flows_table = pd.DataFrame(data)

# Display the table
print(line_flows_table)
