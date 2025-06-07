import numpy as np

from YBUS_singlular_transformation_skip_comments import get_admitance

def print_admitance_matrix_polar(Y_p):
    for i in range(Y_p.shape[0]):
        for j in range(Y_p.shape[1]):
            print(f"Y_p[{i},{j}] = (magnitude: {Y_p[i, j][0]}, angle: {Y_p[i, j][1]} degrees)")

def print_admitance_matrix(Y):
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            print(f"Y[{i},{j}] = {Y[i, j]}")

def admitance_rect_to_polar(Y_rect, angle_in_degrees=True):
    dec_places = 2
    NB = Y_rect.shape[0]
    Y_polar = np.empty((NB, NB), dtype=object)
    for i in range(NB):
        for j in range(NB):
            mag = float(np.round(np.abs(Y_rect[i, j]),dec_places))
            ang = float(np.round(np.angle(Y_rect[i, j], deg=angle_in_degrees),dec_places))
            Y_polar[i, j] = (mag, ang)
    return Y_polar

def read_power_system_file(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    idx = 0

    # NG, NN, NB, NQ, NV, R, tol
    NG, NB, NE, NQ, NV, R, tol, deltaV_max = lines[idx].split(',')
    NG = int(NG)
    NB = int(NB)
    NE = int(NE)
    NQ = int(NQ)
    NV = int(NV)
    R = int(R)
    tol = float(tol)
    deltaV_max = float(deltaV_max)
    idx += 1

    assert (NB == NQ + NV + 1), f"NB ({NB}) must equal NQ ({NQ}) + NV ({NV} + slack bus)"

    second_line = lines[idx]
    node_array = list(map(int, second_line.split(',')))
    idx += 1

    # V and delta for slack bus
    V_slack, delta_slack = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Real bus power for PQ and PV buses
    Pid_Qid = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Reactive bus power for PQ buses
    Q_PQ = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Real bus power for PQ and PV buses
    p_generated = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Reactive bus power for PQ buses
    q_generated = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Voltage magnitudes for PV buses (only if NV > 0)
    voltage_limits_PV = []
    if NV > 0:
        if idx < len(lines):
            entries = [x for x in lines[idx].split(',') if x.strip() != ""]
            if len(entries) < NV:
                raise ValueError(f"Expected {NV} PV voltage limits, but got {len(entries)}: {entries}")
            voltage_limits_PV = [float(x) for x in entries[:NV]]
            idx += 1

    # Maximum voltage for each bus (must be NB entries)
    max_voltage_buses = []
    if idx < len(lines):
        entries = [x for x in lines[idx].split(',') if x.strip() != ""]
        if len(entries) != NB:
            raise ValueError(f"Expected {NB} maximum voltage entries, but got {len(entries)}: {entries}")
        max_voltage_buses = [float(x) for x in entries]
        # Always set the max voltage of the first element equal to the slack value
        max_voltage_buses[0] = V_slack
        idx += 1

    min_voltage_buses = []
    if idx < len(lines):
        entries = [x for x in lines[idx].split(',') if x.strip() != ""]
        if len(entries) != NB:
            raise ValueError(f"Expected {NB} minum voltage entries, but got {len(entries)}: {entries}")
        min_voltage_buses = [float(x) for x in entries]
        # Always set the max voltage of the first element equal to the slack value
        min_voltage_buses[0] = V_slack
        idx += 1

    # Type
    types = [x.strip() for x in lines[idx].split(',')]
    idx += 1
    # Check bus types: 1 SL, NQ PQ, NV PV
    sl_count = types.count("SL")
    pq_count = types.count("PQ")
    pv_count = types.count("PV")
    errors = []
    if sl_count != 1:
        errors.append(f"Expected 1 'SL' bus, found {sl_count}.")
    if pq_count != NQ:
        errors.append(f"Expected {NQ} 'PQ' buses, found {pq_count}.")
    if pv_count != NV:
        errors.append(f"Expected {NV} 'PV' buses, found {pv_count}.")
    if errors:
        raise Exception("Bus type mismatch:\n" + "\n".join(errors))

    # Cost function for Generator NG
    cost_function = []
    for _ in range(NG):
        cost_function.append([float(x) for x in lines[idx].split(',')])
        idx += 1

    return {
        'NG': NG,
        'NB': NB,
        'NE': NE,
        'NQ': NQ,
        'NV': NV,
        'R': R,
        'tol': tol,
        'v_slack': V_slack,
        'angle_slack': delta_slack,
        'demand_real_bus_power_PQ_PV': Pid_Qid,
        'demand_reactive_bus_power_PQ': Q_PQ,
        'v_magnitudes_PV': voltage_limits_PV,
        'v_magnitude_max_limits_PQ': max_voltage_buses,
        'v_magnitude_min_limits_PQ': min_voltage_buses,
        'p_generated': p_generated,
        'q_generated': q_generated,
        'types': types,
        'cost_function': cost_function,
        "deltaV_max": deltaV_max,
        "node_ids": [node-1 for node in node_array]  # Convert to zero-based indexing
    }

def isConnecte(adj_matrix, i, m):
    # This function should check if buses i and j are connected.
    return adj_matrix[i-1, m-1] == 1

def dp_ddelta(i, k, v_next, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to V for bus i.
    """
    result = 0.0
    delta_i = np.deg2rad(delta_next[i])
    delta_k = np.deg2rad(delta_next[k])
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_next[i])
    v_k = np.abs(v_next[k])
    if i != k:
        result = - v_i * v_k*Y_polar[i, k][0] * np.sin(theta_i_k + delta_k - delta_i)
    else:
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_next[m])
                result += v_i*v_m * Y_polar[i, m][0] * np.sin(theta_i_m + delta_m - delta_i)
    return result

def dp_dv(i, k, v_next, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to V for bus i.
    """
    result = 0.0
    delta_i = np.deg2rad(delta_next[i])
    delta_k = np.deg2rad(delta_next[k])
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_next[i])
    v_k = np.abs(v_next[k])
    if i != k:
        result =  v_i * Y_polar[i, k][0] * np.cos(theta_i_k + delta_k - delta_i)
    else:
        result = 2 * v_i * Y_polar[i, i][0] * np.cos(theta_i_k)
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_next[m])
                result += v_m * Y_polar[i, m][0] * np.cos(theta_i_m + delta_m - delta_i)
    return result

def dq_ddelta(i, k, v_next, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to V for bus i.
    """
    result = 0.0
    delta_i = np.deg2rad(delta_next[i])
    delta_k = np.deg2rad(delta_next[k])
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_next[i])
    v_k = np.abs(v_next[k])
    if i != k:
        result = v_i * v_k*Y_polar[i, k][0] * np.cos(theta_i_k + delta_k - delta_i)
    else:
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_next[m])
                result -= v_i*v_m * Y_polar[i, m][0] * np.cos(theta_i_m + delta_m - delta_i)
    return -result

def dq_dv(i, k, v_next, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to V for bus i.
    """
    result = 0.0
    delta_i = np.deg2rad(delta_next[i])
    delta_k = np.deg2rad(delta_next[k])
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_next[i])
    v_k = np.abs(v_next[k])
    if i != k:
        result = - v_i * Y_polar[i, k][0] * np.sin(theta_i_k + delta_k - delta_i)
    else:
        result = 2 * v_i * Y_polar[i, i][0] * np.sin(theta_i_k)
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_next[m])
                result += v_m * Y_polar[i, m][0] * np.sin(theta_i_m + delta_m - delta_i)
    return -result

def merge_lists_as_column(list1, list2):
    """
    Merges two lists into a single column vector (as a 2D numpy array).
    The elements of list1 come first, followed by the elements of list2.
    """
    col1 = np.array(list1).reshape(-1, 1)
    col2 = np.array(list2).reshape(-1, 1)
    merged_col = np.vstack((col1, col2))
    return merged_col

def estimate_jacobian(v_current, delta_current, Y_polar, pq_indices, pv_indices):
    """
    Estimate the Jacobian matrix for the power flow equations.
    """
    row_total = len(pq_indices) + len(pv_indices)
    col_total = len(pq_indices) + len(pv_indices)
    H = np.zeros((row_total, col_total), dtype=float)
    qp_indices = (np.setdiff1d(node_ids, pv_indices))[1:]

    row = 0
    for i in pq_indices:
        for j in pq_indices:
            H[i - 1, j - 1] = dp_ddelta(i, j, v_current, delta_current, Y_polar)
        row += 1

    for i in qp_indices:
        for j in pq_indices:
            H[row, j - 1] = dq_ddelta(i, j, v_current, delta_current, Y_polar)
        row += 1

    for j in range(len(pq_indices) - 1, len(pq_indices)):
        for i in pq_indices:
            H[i - 1, j + 1] = dp_dv(i, j, v_current, delta_current, Y_polar)

    for i in pv_indices:
        for j in pv_indices:
            H[i, j] = dq_dv(i - 1, j - 1, v_current, delta_current, Y_polar)

    return H

A, Y = get_admitance("new_book_6_9.txt")
B = np.imag(Y)
G = np.real(Y)
Y_polar = admitance_rect_to_polar(Y)
#print_admitance_matrix_polar(Y_polar)
data = read_power_system_file("problem_6_9.txt")
NB = data["NB"]
NQ = data["NQ"]
NV = data["NV"]
node_ids = data["node_ids"]

p_demand = data["demand_real_bus_power_PQ_PV"]
q_demand = data["demand_reactive_bus_power_PQ"]
p_generated = data["p_generated"]
q_generated = data["q_generated"]


v_current = np.array([1.04+1j*0.0, 1.0+1j*0.0, 1.04+1j*0.0], dtype=complex)
v_next = v_current.copy()
delta_current = np.array([0]*len(v_current))
delta_next = delta_current.copy()
Pi = 0.0
Qi = 0.0

p_current_calculated = np.zeros(NB)
q_current_calculated = np.zeros(NB)

p_specified = np.array(p_generated.copy()) - np.array(p_demand.copy())
q_specified = np.array(q_generated.copy()) - np.array(q_demand.copy())

pq_indices = np.where(np.isin(np.array(data["types"]), ['PQ', 'PV']))[0].ravel()
pv_indices = np.where(np.array(data["types"]) == 'PV')[0].ravel()

for i in pq_indices:
    for k in range(3):
        delta_i = np.deg2rad(delta_current[i])
        delta_k = np.deg2rad(delta_current[k])
        theta_i_k = np.deg2rad(Y_polar[i, k][1])
        if i not in pv_indices:
            Qi -= np.abs(v_current[i])*np.abs(v_current[k]) * Y_polar[i, k][0]*np.sin(theta_i_k + delta_k - delta_i)
        Pi += np.abs(v_current[i])*np.abs(v_current[k]) * Y_polar[i, k][0]*np.cos(theta_i_k + delta_k - delta_i)
    p_current_calculated[i]= Pi
    q_current_calculated[i]= Qi
    Pi = 0.0
    Qi = 0.0

delta_p_current = p_specified - p_current_calculated
delta_q_current = q_specified - q_current_calculated
residuals = merge_lists_as_column(delta_p_current[1:],delta_q_current[1:2])

H1 = estimate_jacobian(v_current, delta_current, Y_polar, pq_indices, pv_indices)

print("H matrix inverse:", np.linalg.inv(H1)@residuals)