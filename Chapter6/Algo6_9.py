import numpy as np
import pandas as pd

from YBUS_singlular_transformation_skip_comments import get_admitance

def print_admitance_matrix_polar(Y_p):
    """
    Print the admittance matrix in polar form (magnitude and angle in degrees).
    """
    for i in range(Y_p.shape[0]):
        for j in range(Y_p.shape[1]):
            print(f"Y_p[{i},{j}] = (magnitude: {Y_p[i, j][0]}, angle: {Y_p[i, j][1]} degrees)")

def print_admitance_matrix(Y):
    """
    Print the admittance matrix in rectangular form.
    """
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            print(f"Y[{i},{j}] = {Y[i, j]}")

def admitance_rect_to_polar(Y_rect, angle_in_degrees=True):
    """
    Convert a rectangular admittance matrix to polar form.
    Returns a matrix of (magnitude, angle) tuples.
    """
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
    """
    Read and parse the power system data file.
    Returns a dictionary with all relevant system parameters.
    """
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]

    idx = 0

    # Parse system parameters
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

    # Slack bus voltage and angle
    V_slack, delta_slack = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Real and reactive power demands and generations
    Pid_Qid = [float(x) for x in lines[idx].split(',')]
    idx += 1
    Q_PQ = [float(x) for x in lines[idx].split(',')]
    idx += 1
    p_generated = [float(x) for x in lines[idx].split(',')]
    idx += 1
    q_generated = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Voltage limits for PV buses
    voltage_limits_PV = []
    if NV > 0:
        if idx < len(lines):
            entries = [x for x in lines[idx].split(',') if x.strip() != ""]
            if len(entries) < NV:
                raise ValueError(f"Expected {NV} PV voltage limits, but got {len(entries)}: {entries}")
            voltage_limits_PV = [float(x) for x in entries[:NV]]
            idx += 1

    # Maximum and minimum voltage for each bus
    max_voltage_buses = []
    if idx < len(lines):
        entries = [x for x in lines[idx].split(',') if x.strip() != ""]
        if len(entries) != NB:
            raise ValueError(f"Expected {NB} maximum voltage entries, but got {len(entries)}: {entries}")
        max_voltage_buses = [float(x) for x in entries]
        max_voltage_buses[0] = V_slack
        idx += 1

    min_voltage_buses = []
    if idx < len(lines):
        entries = [x for x in lines[idx].split(',') if x.strip() != ""]
        if len(entries) != NB:
            raise ValueError(f"Expected {NB} minum voltage entries, but got {len(entries)}: {entries}")
        min_voltage_buses = [float(x) for x in entries]
        min_voltage_buses[0] = V_slack
        idx += 1

    # Bus types
    types = [x.strip() for x in lines[idx].split(',')]
    idx += 1
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

    # Generator cost function
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
    """
    Check if buses i and m are connected in the adjacency matrix.
    """
    return adj_matrix[i-1, m-1] == 1

def dp_ddelta(i, k, v_current, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to delta for bus i.
    """
    result = 0.0
    delta_i = delta_next[i]
    delta_k = delta_next[k]
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_current[i])
    v_k = np.abs(v_current[k])
    if i != k:
        result = - v_i * v_k*Y_polar[i, k][0] * np.sin(theta_i_k + delta_k - delta_i)
    else:
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_current[m])
                result += v_i*v_m * Y_polar[i, m][0] * np.sin(theta_i_m + delta_m - delta_i)
    return result

def dp_dv(i, k, v_current, delta_next, Y_polar):
    """
    Calculate the partial derivative of P with respect to V for bus i.
    """
    result = 0.0
    delta_i = delta_next[i]
    delta_k = delta_next[k]
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_current[i])
    v_k = np.abs(v_current[k])
    if i != k:
        result =  v_i * Y_polar[i, k][0] * np.cos(theta_i_k + delta_k - delta_i)
    else:
        result = 2 * v_i * Y_polar[i, i][0] * np.cos(theta_i_k)
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_current[m])
                result += v_m * Y_polar[i, m][0] * np.cos(theta_i_m + delta_m - delta_i)
    return result

def dq_ddelta(i, k, v_current, delta_next, Y_polar):
    """
    Calculate the partial derivative of Q with respect to delta for bus i.
    """
    result = 0.0
    delta_i = delta_next[i]
    delta_k = delta_next[k]
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_current[i])
    v_k = np.abs(v_current[k])
    if i != k:
        result = v_i * v_k*Y_polar[i, k][0] * np.cos(theta_i_k + delta_k - delta_i)
    else:
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_current[m])
                result -= v_i*v_m * Y_polar[i, m][0] * np.cos(theta_i_m + delta_m - delta_i)
    return -result

def dq_dv(i, k, v_current, delta_next, Y_polar):
    """
    Calculate the partial derivative of Q with respect to V for bus i.
    """
    result = 0.0
    delta_i = delta_next[i]
    delta_k = delta_next[k]
    theta_i_k = np.deg2rad(Y_polar[i, k][1])
    v_i = np.abs(v_current[i])
    v_k = np.abs(v_current[k])
    if i != k:
        result = - v_i * Y_polar[i, k][0] * np.sin(theta_i_k + delta_k - delta_i)
    else:
        result = 2 * v_i * Y_polar[i, i][0] * np.sin(theta_i_k)
        for m in range(NB):
            if m != i:
                theta_i_m = np.deg2rad(Y_polar[i, m][1])
                delta_m = np.deg2rad(delta_next[m])
                v_m = np.abs(v_current[m])
                result += v_m * Y_polar[i, m][0] * np.sin(theta_i_m + delta_m - delta_i)
    return -result

def merge_lists_as_column(list1, list2):
    """
    Merge two lists into a single column vector (2D numpy array).
    """
    col1 = np.array(list1).reshape(-1, 1)
    col2 = np.array(list2).reshape(-1, 1)
    merged_col = np.vstack((col1, col2))
    return merged_col

def estimate_jacobian(v_current, delta_current, Y_polar, pq_indices, pv_indices):
    """
    Estimate the Jacobian matrix for the power flow equations.
    """
    row_pq = len(pq_indices)
    col_pq = len(pq_indices)
    col_pv = len(pv_indices)
    qp_indices = (np.setdiff1d(node_ids, pv_indices))[1:]
    row_qp = len(qp_indices)

    A = np.zeros((row_pq, col_pq), dtype=float)
    B = np.zeros((row_qp, col_pq), dtype=float)
    C = np.zeros((row_pq, col_pv), dtype=float)
    D = np.zeros((row_qp, row_qp), dtype=float)

    for row, i in enumerate(pq_indices):
        for col, j in enumerate(pq_indices):
            A[row, col] = dp_ddelta(i, j, v_current, delta_current, Y_polar)

    for row, i in enumerate(qp_indices):
        for col,j in enumerate(pq_indices):
            B[row, col] = dq_ddelta(i, j, v_current, delta_current, Y_polar)

    for row, i in enumerate(pq_indices):
        for col,j in enumerate(qp_indices):
            C[row, col] = dp_dv(i, j, v_current, delta_current, Y_polar)

    for row, i in enumerate(qp_indices):
        for col, j in enumerate(qp_indices):
            D[row,col] = dq_dv(i, j, v_current, delta_current, Y_polar)

    H = np.hstack((np.vstack((A, B)), np.vstack((C, D))))
    return H

def calculate_jacobian(v_current, delta_current, Y_polar, pq_indices, qp_indices):
    """
    Calculate the Jacobian matrix for the power flow equations.
    """
    A = np.array([[dp_ddelta(i, j, v_current, delta_current, Y_polar) for j in pq_indices] for i in pq_indices])
    B = np.array([[dq_ddelta(i, j, v_current, delta_current, Y_polar) for j in pq_indices] for i in qp_indices])
    C = np.array([[dp_dv(i, j, v_current, delta_current, Y_polar) for j in qp_indices] for i in pq_indices])
    D = np.array([[dq_dv(i, j, v_current, delta_current, Y_polar) for j in qp_indices] for i in qp_indices])

    H = np.hstack((np.vstack((A, B)), np.vstack((C, D))))
    return H

def calculate_Q_at_PV(q_current_calculated, pv_indices, delta_next, v_next, Y_polar):
    """
    Calculate Q at PV buses and update q_current_calculated.
    """
    Qi = 0.0
    for i in pv_indices:
        for k in range(NB):
            delta_i = delta_next[i]
            delta_k = delta_next[k]
            theta_i_k = np.deg2rad(Y_polar[i, k][1])
            Qi += np.abs(v_next[i]) * np.abs(v_next[k]) * Y_polar[i, k][0] * np.sin(theta_i_k + delta_k - delta_i)
        q_current_calculated[i] = -Qi
        Qi = 0.0
    return q_current_calculated
    
def inject_power_to_slack(p_current_calculated, q_current_calculated, Y_polar, v_current, delta_current):
    """
    Injects the calculated power into the slack bus.
    """
    p_current_calculated[0] = 0.0
    q_current_calculated[0] = 0.0
    
    for k in range(NB):
        delta_k = delta_current[k]
        theta_i_k = np.deg2rad(Y_polar[0, k][1])
        p_current_calculated[0]  += np.abs(v_current[0]) * np.abs(v_current[k]) * Y_polar[0, k][0] * np.cos(theta_i_k + delta_k)
        q_current_calculated[0]  -= np.abs(v_current[0]) * np.abs(v_current[k]) * Y_polar[0, k][0] * np.sin(theta_i_k + delta_k)
    
def log_iteration(log_df, iteration, p_calc, q_calc, v, delta, error):
    """
    Log the results of each iteration into a pandas DataFrame.
    """
    log_df.loc[iteration] = {
        'Iteration': iteration,
        'Max|P|': np.max(np.abs(p_calc)),
        'Max|Q|': np.max(np.abs(q_calc)),
        'Max|V|': np.max(np.abs(v)),
        'Max|Delta|': np.max(np.abs(delta)),
        'Error': error
    }
    return log_df

def display_final_results(v_next, delta_next, p_current_calculated, q_current_calculated, log_df):
    """
    Display the final results after convergence: voltages, angles, power, and iteration summary.
    """
    print("\nFinal Bus Voltages (magnitude and angle in degrees):")
    for i, v in enumerate(v_next):
        mag = np.abs(v)
        ang = np.rad2deg(delta_next[i])  # Use delta_next directly
        print(f"Bus {i + 1}: |V| = {mag:.4f}, angle = {ang:.2f}°")

    print("\nFinal Bus Power Calculations:")
    for i, (p, q) in enumerate(zip(p_current_calculated, q_current_calculated)):
        print(f"Bus {i+1}: P = {p:.4f}, Q = {q:.4f}")

    print("\nSummary of Iterations:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(log_df.to_string(index=False))


# --- Main execution starts here ---

A, Y = get_admitance("new_book_6_9.txt")
B = np.imag(Y)
G = np.real(Y)
Y_polar = admitance_rect_to_polar(Y)
# print_admitance_matrix_polar(Y_polar)
data = read_power_system_file("problem_6_9.txt")
NB = data["NB"]
NQ = data["NQ"]
NV = data["NV"]
node_ids = data["node_ids"]

p_demand = data["demand_real_bus_power_PQ_PV"]
q_demand = data["demand_reactive_bus_power_PQ"]
p_generated = data["p_generated"]
q_generated = data["q_generated"]

# Initialize voltages and angles
v_current = np.array([1.04+1j*0.0, 1.0+1j*0.0, 1.04+1j*0.0], dtype=complex)
v_next = v_current.copy()
delta_current = np.array([0.00]*len(v_current), dtype=float)
delta_next = delta_current.copy()
Pi = 0.0
Qi = 0.0

p_current_calculated = np.zeros(NB)
q_current_calculated = np.zeros(NB)

p_specified = np.array(p_generated.copy()) - np.array(p_demand.copy())
q_specified = np.array(q_generated.copy()) - np.array(q_demand.copy())

pq_indices = np.where(np.isin(np.array(data["types"]), ['PQ', 'PV']))[0].ravel()
pv_indices = np.where(np.array(data["types"]) == 'PV')[0].ravel()
qp_indices = np.where(np.isin(np.array(data["types"]), ['PQ']))[0].ravel()

r = 0
error = 1.0

log_df = pd.DataFrame(columns=['Iteration', 'Max|P|', 'Max|Q|', 'Max|V|', 'Max|Delta|', 'Error'])
while (r < data["R"] and error > data["tol"]):
    log_df = log_iteration(log_df, r, p_current_calculated, q_current_calculated, v_next, delta_next, error)
    print(
        f"Iteration {r}: Error={error:.6f}, Max|V|={np.max(np.abs(v_next)):.4f}, Max|Delta|={np.max(np.abs(delta_next)):.4f}")
    for i in pq_indices:
        for k in range(NB):
            delta_i = delta_current[i]
            delta_k = delta_current[k]
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
    all_deltas = merge_lists_as_column(delta_p_current[1:],delta_q_current[1:2])
    H = calculate_jacobian(v_current, delta_current, Y_polar, pq_indices, qp_indices)

    residuals = np.linalg.inv(H)@all_deltas

    current_deltas = np.hstack((np.array(delta_current[1:]), np.abs(v_current[qp_indices]))).reshape(-1,1)
    next_deltas = current_deltas + residuals

    delta_next[pq_indices] = next_deltas[:len(pq_indices), 0]
    v_next[qp_indices] = next_deltas[len(pq_indices):, 0]

    # Update Q at PV buses
    calculate_Q_at_PV(q_current_calculated, pv_indices, delta_next, v_next ,Y_polar)

    error = np.max(np.abs(delta_next - delta_current)) + np.max(np.abs(v_next - v_current))
    v_current = v_next.copy()
    delta_current = delta_next.copy()
    r = r+1

print("Done with iterations")
inject_power_to_slack(p_current_calculated, q_current_calculated, Y_polar, v_current, delta_current)

display_final_results(v_next, delta_next, p_current_calculated, q_current_calculated, log_df)
