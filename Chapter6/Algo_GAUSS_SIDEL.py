import cmath

import numpy as np

from YBUS_singlular_transformation_skip_comments import get_admitance
from utils import insert_slack_power


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

    # V and delta for slack bus
    V_slack, delta_slack = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Real bus power for PQ and PV buses
    Pid_Qid = [float(x) for x in lines[idx].split(',')]
    idx += 1

    # Reactive bus power for PQ buses
    Q_PQ = [float(x) for x in lines[idx].split(',')]
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
        'real_bus_power_PQ_PV': Pid_Qid,
        'reactive_bus_power_PQ': Q_PQ,
        'v_magnitudes_PV': voltage_limits_PV,
        'v_magnitude_max_limits_PQ': max_voltage_buses,
        'v_magnitude_min_limits_PQ': min_voltage_buses,
        'types': types,
        'cost_function': cost_function,
        "deltaV_max": deltaV_max
    }

def get_initial_voltage_vector_rectangular(data):
    V_initial_rec = []
    for _ in range(data["NB"]):
        V_initial_rec.append(1.0 + 0.0j)
    V_initial_rec[0] = data['v_slack'] * cmath.exp(1j * data['angle_slack'])
    return np.array(V_initial_rec)

def get_initial_voltage_vector_polar(data):
    V_initial_polar = []
    for _ in range(data["NB"]):
        V_initial_polar.append((1.0, 0.0))  # (magnitude, angle)
    V_initial_polar[0] = (data['v_slack'], data['angle_slack'])
    return np.array(V_initial_polar)

import numpy as np

def populate_voltage_vector(data):
    """
    Populate the voltage vector based on bus types.

    Parameters:
        V_slack (complex): Slack bus voltage.
        NV (int): Number of PV buses.
        NB (int): Total number of buses.
        types (list): List of bus types ('SL', 'PQ', 'PV').

    Returns:
        numpy.ndarray: Voltage vector of size NB.

    Raises:
        ValueError: If the size of the resulting array is not NB.
    """
    v_slack = data["v_slack"]
    angle_slack = data["angle_slack"]
    types = data["types"]
    NV = data["NV"]
    NB = data["NB"]

    voltage_vector = np.ones(NB, dtype=complex)
    pq_count = 0
    pv_count = 0

    for i, bus_type in enumerate(types):
        if bus_type == 'SL':
            voltage_vector[i] = v_slack* np.exp(1j * angle_slack)
        elif bus_type == 'PQ':
            voltage_vector[i] = 1 + 0j
            pq_count += 1
        elif bus_type == 'PV':
            voltage_vector[i] = 1.04 + 0j
            pv_count += 1

    if pv_count != NV or len(voltage_vector) != NB:
        raise ValueError(f"Mismatch in bus counts: NV={NV}, PV={pv_count}, NB={NB}, Vector size={len(voltage_vector)}")

    return voltage_vector

def enforce_voltage_limits(v_next, data):
    """
    Enforce voltage limits on the V_next array.

    Parameters:
        V_next (numpy.ndarray): Array of complex voltages.
        max_V (numpy.ndarray): Array of maximum voltage magnitudes.
        min_V (numpy.ndarray): Array of minimum voltage magnitudes.

    Returns:
        numpy.ndarray: Adjusted V_next array.
    """

    max_v = data["v_magnitude_max_limits_PQ"]
    min_v = data["v_magnitude_min_limits_PQ"]
    for i in range(len(v_next)):
        magnitude = np.abs(v_next[i])
        if magnitude > max_v[i]:
            v_next[i] = max_v[i] * v_next[i] / magnitude  # Scale to max_V
        elif magnitude < min_v[i]:
            v_next[i] = min_v[i] * v_next[i] / magnitude  # Scale to min_V
    return v_next

def enforce_voltage_i_limits(v_next, data, i):
    """
    Enforce voltage limits on the V_next array.

    Parameters:
        V_next (numpy.ndarray): Array of complex voltages.
        max_V (numpy.ndarray): Array of maximum voltage magnitudes.
        min_V (numpy.ndarray): Array of minimum voltage magnitudes.

    Returns:
        numpy.ndarray: Adjusted V_next array.
    """
    max_v = data["v_magnitude_max_limits_PQ"]
    min_v = data["v_magnitude_min_limits_PQ"]
    magnitude = np.abs(v_next[i])
    if magnitude > max_v[i]:
        v_next[i] = max_v[i] * v_next[i] / magnitude  # Scale to max_V
    elif magnitude < min_v[i]:
        v_next[i] = min_v[i] * v_next[i] / magnitude  # Scale to min_V
    return v_next


data = read_power_system_file("problem_6_8.txt")
#for key, value in data.items():
#    print(f"{key}: {value}")

Y = get_admitance("new_book_6_8.txt")
#V_initial = get_initial_voltage_vector_rectangular(data)
V_initial = populate_voltage_vector(data)

A = np.zeros(data["NB"], dtype=complex)

for i in range(1,data["NB"]):
    A[i] = data["real_bus_power_PQ_PV"][i] - 1j * data["reactive_bus_power_PQ"][i]

B = np.zeros((data["NB"], data["NB"]), dtype=complex)
for i in range(1, data["NB"]):
    for k in range(data["NB"]):
        if i != k:
            B[i, k] = Y[i, k]


r = 0
v_current = V_initial
v_next = V_initial.copy()
deltaV_max = data["deltaV_max"]
pq_indices = [i for i, t in enumerate(data["types"]) if t == "PQ"]
pv_indices = [i for i, t in enumerate(data["types"]) if t == "PV"]

p_current = data["real_bus_power_PQ_PV"]
q_current = data["reactive_bus_power_PQ"]
delta_current = np.zeros(data["NB"], dtype=complex)

delta_next = delta_current.copy()
p_next = p_current.copy()
q_next = q_current.copy()

while r < data["R"]:
    for i in pq_indices:
        one = A[i] / np.conj(v_current[i])
        # Use np.where to select v_next for k < i, else v_current
        v_all = np.where(np.arange(data["NB"]) < i, v_next, v_current)
        two = np.dot(B[i, :], v_all)
        v_next[i] = (one - two) / Y[i, i]
        v_temp = v_current[i]
        v_current[i] = v_next[i]
        v_next = enforce_voltage_i_limits(v_next, data,i)
        v_diff = np.abs(v_next[i] - v_temp)
        if (v_diff <= deltaV_max):
            continue
        else:
            deltaV_max  = v_diff

    for i in pv_indices:
        one = 0.0
        two = 0.0
        three = 0.0
        for k in range(0, data["NB"]):
            if k < i:
                two +=  Y[i, k] * v_next[k]
                three -= B[i,k] * v_next[k]
            elif k >= i:
                two   += Y[i, k] * v_current[k]
                three -= B[i, k] * v_current[k]
            q_next[i] = -np.imag(two*np.conj(v_current[i]))
        A[i] = (data["real_bus_power_PQ_PV"][i] - 1j * q_next[i])
        one = A[i] / np.conj(v_current[i])
        delta_next[i] = np.angle((one + three) / Y[i, i])
        v_next[i] = np.abs(v_current[i]) * np.exp(1j * delta_next[i])
    if (v_diff < data['tol']):
        print("Convergence achieved at iteration", r + 1)
        break
    r += 1

if r == data["R"]:
    print("Maximum iterations reached without convergence.")
else:
    print("Convergence achieved at iteration", r + 1)
    print("Voltage vector after convergence:", np.round(v_next,4))


S = [p_next[i]+1j*q_next[i] for i in range(data["NB"])]
V = v_next.copy()
insert_slack_power(S, V, Y, data["types"])


Sik = np.zeros((data["NB"], data["NB"]), dtype=complex)
for i in range(data["NB"]):
    for k in range(data["NB"]):
        Sik[i,k] = v_next[i] * (np.conj(v_next[i])-np.conj(v_next[k])) * np.conj(Y[i, k])

Ski = np.zeros((data["NB"], data["NB"]), dtype=complex)
for k in range(data["NB"]):
    for i in range(data["NB"]):
        Ski[k,i] = v_next[k] * (np.conj(v_next[k])-np.conj(v_next[i])) * np.conj(Y[i, k])

print(np.round(Sik,4))
print(np.round(Ski,4))
print(S)


