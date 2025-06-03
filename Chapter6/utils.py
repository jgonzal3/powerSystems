import numpy as np
import cmath
import math

from YBUS_singlular_transformation_skip_comments import get_admitance


def Pi_injected(V, Y, i):
    '''
    Calculate the active power injected at bus i.
    :param V:
    :param Y:
    :param i:
    :return:
    '''
    return (np.sum(Y[i, :] * V) * np.conj(V[i])).real

def P_injected(V, Y):
    '''
    Calculate the active power injected at bus i.
    :param V:
    :param Y:
    :param i:
    :return:
    '''
    n = len(V)
    P = np.zeros(n)
    for i in range(n):
        P[i] = (np.sum(Y[i, :] * V) * np.conj(V[i])).real
    return P

def Qi_injected(V, Y, i):
    '''

    :param V:
    :param Y:
    :param i:
    :return:
    '''
    return (np.conj(np.sum(Y[i, :] * V) * np.conj(V[i])).imag)

def Si_injected(V, Y, i):
    '''

    :param V:
    :param Y:
    :param i:
    :return:
    '''
    return (np.conj(np.sum(Y[i, :] * V) * np.conj(V[i])))

def S_injected(V, Y):
    '''

    :param V:
    :param Y:
    :param i:
    :return:
    '''
    n = len(V)
    S = np.zeros(n, dtype=complex)
    for i in range(n):
        S[i] = (np.sum(Y[i, :] * V) * np.conj(V[i]))
    return np.conj(S)


def Pi_injected_polar(V_magnitude, V_angle, Y, i):
    """
    Calculate power (active and reactive) given admittance and voltage in polar coordinates.

    Parameters:
        V_magnitude (numpy.ndarray): Voltage magnitudes.
        V_angle (numpy.ndarray): Voltage angles in radians.
        Y (numpy.ndarray): Admittance.
        i (int): Index of the bus.

    Returns:
        float: Active power (P) as scalar.
    """
    n = len(V_magnitude)
    Y_angle = np.angle(Y)
    Y_magnitude = np.abs(Y)
    P = 0.0

    for k in range(n):
        angle_diff = V_angle[i] - V_angle[k] - Y_angle[i, k]
        P += V_magnitude[i] * V_magnitude[k] * Y_magnitude[i, k] * np.cos(angle_diff)

    return P


def P_injected_polar(V_magnitude, V_angle, Y):
    """
    Calculate power (active and reactive) given admittance and voltage in polar coordinates.

    Parameters:
        V_magnitude (numpy.ndarray): Voltage magnitudes.
        V_angle (numpy.ndarray): Voltage angles in radians.
        Y (numpy.ndarray): Admittance.
        i (int, optional): Index of the bus. If None, calculate for all buses.

    Returns:
        tuple: Active power (P) as numpy arrays or scalars.
    """
    n = len(V_magnitude)
    Y_angle = np.angle(Y)
    Y_magnitude = np.abs(Y)
    P = np.zeros(n)

    for i in range(n):
        for k in range(n):
            angle_diff = V_angle[i] - V_angle[k] - Y_angle[i, k]
            P[i] += V_magnitude[i] * V_magnitude[k] * Y_magnitude[i, k] * np.cos(angle_diff)

    return P

def Qi_injected_polar(V_magnitude, V_angle, Y, i):
    """
    Calculate power (active and reactive) given admittance and voltage in polar coordinates.

    Parameters:
        V_magnitude (numpy.ndarray): Voltage magnitudes.
        V_angle (numpy.ndarray): Voltage angles in radians.
        Y (numpy.ndarray): Admittance.
        i (int, optional): Index of the bus. If None, calculate for all buses.

    Returns:
        tuple: Active power (P) as numpy arrays or scalars.
    """
    n = len(V_magnitude)
    Y_angle = np.angle(Y)
    Y_magnitude = np.abs(Y)
    Q = 0.0

    for k in range(n):
        angle_diff = V_angle[i] - V_angle[k] - Y_angle[i, k]
        Q -= V_magnitude[i] * V_magnitude[k] * Y_magnitude[i, k] * np.sin(angle_diff)

    return Q

def Q_injected_polar(V_magnitude, V_angle, Y):
    """
    Calculate power (active and reactive) given admittance and voltage in polar coordinates.

    Parameters:
        V_magnitude (numpy.ndarray): Voltage magnitudes.
        V_angle (numpy.ndarray): Voltage angles in radians.
        Y (numpy.ndarray): Admittance.
        i (int, optional): Index of the bus. If None, calculate for all buses.

    Returns:
        tuple: Active power (P) as numpy arrays or scalars.
    """
    n = len(V_magnitude)
    Y_angle = np.angle(Y)
    Y_magnitude = np.abs(Y)
    Q = np.zeros(n)

    for i in range(n):
        for k in range(n):
            angle_diff = V_angle[i] - V_angle[k] - Y_angle[i, k]
            Q[i] -= V_magnitude[i] * V_magnitude[k] * Y_magnitude[i, k] * np.sin(angle_diff)
    return Q

def estimate_lambda(cost_function, PD):
    """
    Estimate lambda based on the given formula.

    Parameters:
        Pd (float): Power demand.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        float: Estimated lambda.
    """
    NG = cost_function.shape[0]  # Number of generators (rows in the matrix)
    P_demand = np.sum(PD)  # Total power demand
    # Calculate numerator and denominator
    numerator = P_demand + sum(cost_function[i][1] / (2 * cost_function[i][0]) for i in range(NG))
    denominator = sum(1 / (2 * cost_function[i][0]) for i in range(NG))
    # Calculate lambda
    l = numerator / denominator

    return l

def estimate_power(cost_function, lambda_p, NB):
    """
    Estimate the power of each generator.

    Parameters:
        lambda_value (float): The calculated lambda.
        M (numpy.ndarray): Matrix of size (NG x 3).

    Returns:
        numpy.ndarray: Array of power values for each generator.
    """
    NG = cost_function.shape[0]  # Number of generators
    power = np.array([(lambda_p - cost_function[i][1])/(2 * cost_function[i][0]) for i in range(NG)])

    if len(power) < NB:
        power = np.concatenate((power, [0] * (NB - len(power))))

    return power


def Qi_only(voltage, Y,  bus, bus_types):
    """
    Estimate power for PV buses.

    Parameters:
        v_magnitude (numpy.ndarray): Voltage magnitudes for all buses.
        v_angle (numpy.ndarray): Voltage angles in radians for all buses.
        bus_types (list): List of bus types ('PV', 'PQ', or 'SL').
        Y (numpy.ndarray): Admittance matrix.
        bus_type (str): Type of bus to estimate power for ('PV', 'PQ', or 'SL').

    Returns:
        numpy.ndarray: Estimated power for PV buses.
    """
    p_indices = [i for i, bus_type in enumerate(bus_types) if bus_type == bus]
    power = np.zeros_like(v_magnitude)

    for i in p_indices:
        power[i] =Qi_injected(voltage, Y, i)

    return power

def insert_slack_power(S, V, Y, bus_types):
    """
    Estimate power for PV buses.

    Parameters:
        v_magnitude (numpy.ndarray): Voltage magnitudes for all buses.
        v_angle (numpy.ndarray): Voltage angles in radians for all buses.
        bus_types (list): List of bus types ('PV', 'PQ', or 'SL').

    Returns:
        numpy.ndarray: Estimated power for PV buses.
    """
    slack_idx = [i for i, bus_type in enumerate(bus_types) if bus_type == 'SL']
    index_val = slack_idx[0] if len(slack_idx) > 0 else None
    S[index_val] = Si_injected(V,Y, slack_idx)

    return S


def total_power_injected(v_magnitude,V_angle, G, B, NB):
    """
    Calculate the total power injected into the system.

    Parameters:
        V (numpy.ndarray): Voltage vector.
        Y (numpy.ndarray): Admittance matrix.

    Returns:
        numpy.ndarray: Total power injected into the system.
    """
    G = np.real(Y)
    B = np.imag(Y)
    s = np.array((NB, NB), dtype=complex)

    Pi = [sum(
        v_magnitude[i] * v_magnitude[k] * (
                    G[i][k] * np.cos(V_angle[i] - V_angle[k]) + B[i][k] * np.sin(V_angle[i] - V_angle[k])) for k in
        range(NB)) for i in range(NB)]

    Qi = [sum(
        v_magnitude[i] * v_magnitude[k] * (
                    G[i][k] * np.sin(V_angle[i] - V_angle[k]) - B[i][k] * np.cos(V_angle[i] - V_angle[k])) for k in
        range(NB)) for i in range(NB)]

    return np.array([Pi[i] + 1j * Qi[i] for i in range(NB)])

def power_Reset(P_gen, P_min, P_max):
    """
    Resets the power values in P_gen based on the minimum and maximum limits.

    Parameters:
        P_gen (numpy.ndarray): 2xn array where the first n elements are power values
                               and the last n elements are violation flags (0 or 1).
        P_min (numpy.ndarray): Array of minimum power values (n elements).
        P_max (numpy.ndarray): Array of maximum power values (n elements).

    Returns:
        numpy.ndarray: Updated P_gen with adjusted power values and violation flags.
    """
    n = len(P_min)
    power_values = P_gen[:n]
    violations = P_gen[n:]

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


def parse_value(val):
    """
    Parses a value that can be either a complex number or a dict with 'mod' and 'angle' (degrees).
    Returns a complex number.
    """
    if isinstance(val, complex):
        return val
    elif isinstance(val, dict) and 'mod' in val and 'angle' in val:
        # angle in degrees
        angle_rad = math.radians(val['angle'])
        return cmath.rect(val['mod'], angle_rad)
    else:
        raise ValueError("Invalid parameter format")

def calculate_received_power(V_input, param):
    """
    V_input: dict with keys 'Vs' and 'Vr', each either complex or {'mod': float, 'angle': float (deg)}
    param: dict with keys 'A', 'B', 'C', 'D', each either complex or {'mod': float, 'angle': float (deg)}
    Returns: (Pr, Qr)
    """
    # Parse Vs and Vr
    Vs = parse_value(V_input['Vs'])
    Vr = parse_value(V_input['Vr'])

    # Parse A and B
    A = parse_value(param['A'])
    B = parse_value(param['B'])
    # C and D are not used in this calculation

    # Calculate magnitudes and angles
    abs_Vs = abs(Vs)
    abs_Vr = abs(Vr)
    abs_B = abs(B)
    abs_A = abs(A)
    beta = cmath.phase(B)
    alfa = cmath.phase(A)
    delta = cmath.phase(Vs)

    # Calculate Pr and Qr
    cos_diff_v = math.cos(beta - delta)
    cos_diff = math.cos(beta - alfa)
    sin_diff_v = math.sin(beta - delta)
    sin_diff = math.sin(beta - alfa)

    Pr = (abs_Vr * abs_Vs / abs_B) * cos_diff_v - (abs_A / abs_B) * (abs_Vr ** 2) * cos_diff
    Qr = (abs_Vr * abs_Vs / abs_B) * sin_diff_v - (abs_A / abs_B) * (abs_Vr ** 2) * sin_diff

    return Pr, Qr

def calculate_received_power_simplified(V_input, param):
    """
    V_input: dict with keys 'Vs' and 'Vr', each either complex or {'mod': float, 'angle': float (deg)}
    param: dict with keys 'A', 'B', 'C', 'D', each either complex or {'mod': float, 'angle': float (deg)}
    Returns: (Pr, Qr)
    """
    # Parse Vs and Vr
    Vs = parse_value(V_input['Vs'])
    Vr = parse_value(V_input['Vr'])

    # Parse A and B
    B = parse_value(param['B'])
    # C and D are not used in this calculation

    # Calculate magnitudes and angles
    abs_Vs = abs(Vs)
    abs_Vr = abs(Vr)
    X = np.imag(B)
    delta = np.angle(Vs)

    # Calculate Pr and Qr


    Pr = (abs_Vr * abs_Vs / X) * np.sin(delta)
    Qr = (abs_Vr/ X) *(abs_Vs - abs_Vr)

    return Pr, Qr



def calculate_sending_power(V_input, param):
    """
    V_input: dict with keys 'Vs' and 'Vr', each either complex or {'mod': float, 'angle': float (deg)}
    param: dict with keys 'A', 'B', 'C', 'D', each either complex or {'mod': float, 'angle': float (deg)}
    Returns: (Ps, Qs)
    """
    Vs = parse_value(V_input['Vs'])
    Vr = parse_value(V_input['Vr'])
    B = parse_value(param['B'])
    D = parse_value(param['D'])

    abs_Vs = abs(Vs)
    abs_Vr = abs(Vr)
    abs_B = abs(B)
    abs_D = abs(D)
    delta = cmath.phase(Vs)

    # If values are complex, use angles of B (beta) and D (alfa), A and C are zero
    # If values are mod/angle, use their values directly (already handled by parse_value)
    beta = cmath.phase(B)
    alfa = cmath.phase(D)

    cos_diff = math.cos(beta - alfa)
    sin_diff = math.sin(beta - alfa)
    cos_sum = math.cos(beta + delta)
    sin_sum = math.sin(beta + delta)

    Ps = (abs_D / abs_B) * (abs_Vs ** 2) * cos_diff - (abs_Vr * abs_Vs / abs_B) * cos_sum
    Qs = (abs_D / abs_B) * (abs_Vs ** 2) * sin_diff - (abs_Vr * abs_Vs / abs_B) * sin_sum

    return Ps, Qs


V_input = {
    'Vs': {'mod': 1, 'angle': 14.5},
    'Vr': {'mod': 1, 'angle': 0},
}
param = {
    'A': {'mod': 0.0, 'angle': 0},
    'B': complex(0,0.05),
    'C': {'mod': 0, 'angle': 0},
    'D': {'mod': 1, 'angle': 0}
}
#Pr, Qr = calculate_received_power_simplified(V_input, param)

#voltage = np.array([1.04, 1.0190901999936934 + 1j*0.046361087607556736,  1.0, 1.0], dtype=complex)

#Y = get_admitance("new_book_6_3_b.txt")
#print(Si_injected(voltage,Y,1))

# Example usage
#v_magnitude = np.array([1.04, 1.0201442, 1.0, 1.0])  # Voltage magnitudes
#v_angle = np.array([0, 0.04546128, 0.0, 0.0])  # Voltage angles in radians

#P = Pi_injected_polar(v_magnitude, v_angle, Y,1)
#Q = Qi_injected_polar(v_magnitude, v_angle, Y,1)

#print("Active Power (P):", P)
#print("Reactive Power (Q):", Q)


#P = Pi_injected_polar(v_magnitude, v_angle, Y, 1)
#Q = Qi_injected_polar(v_magnitude, v_angle, Y,1)

#print("Active Power (P):", P)
#print("Reactive Power (Q):", Q)

#S = S_injected(voltage,Y)
#print("Complex Power (S):", S)

#print("Complex Power (S):", total_power_injected(v_magnitude,v_angle, np.real(Y), np.real(Y), 4))

#array = ["PV", "PV", "PQ", "SL"]
##print(Qi_only(voltage,Y, 'PV', array))

#S = np.zeros(4, dtype=complex)

#print(insert_slack_power(S, voltage, Y, array))
