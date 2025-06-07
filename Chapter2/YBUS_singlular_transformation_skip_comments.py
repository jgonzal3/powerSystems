import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from networkx.classes import edge_subgraph

def plot_graph(nodes, edges):
    """
    Plot a graph using networkx.

    Parameters:
        nodes (list): List of nodes.
        edges (list of tuples): List of edges as (from_node, to_node).
    """
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Graph Visualization")
    plt.show()


def parse_admitance_file_yield(file_path):
    """
    Parse the admitance file to extract the number of edges, number of nodes,
    node array, edge array, and admittance values.

    Parameters:
        file_path (str): Path to the file.

    Yields:
        tuple: Parsed data for each line (line_number, bus_code, shunt, series).
    """
    with open(file_path, 'r') as file:
        # Skip lines that are empty or start with '#'
        lines = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]
        idx = 0

        # Read the first line to get no_edges and no_nodes
        first_line = lines[idx]
        no_edges, no_nodes = map(int, first_line.split(','))
        idx += 1

        # Read the second line to get the node array
        second_line = lines[idx]
        node_array = list(map(int, second_line.split(',')))
        idx += 1

        # Read the next no_edges lines to get the edge array
        edge_array = []
        for _ in range(no_edges):
            edge_line = lines[idx]
            edge = tuple(map(int, edge_line.split(',')))
            edge_array.append(edge)
            idx += 1

        # Yield the basic graph structure
        additional_array = lines[idx]
        admitances_line = list(map(complex, additional_array.split(',')))
        idx += 1

        print(admitances_line)

        yield no_edges, no_nodes, node_array, edge_array, admitances_line

        # Yield each line's admittance data
        for _ in range(no_edges):
            line = lines[idx]
            line_number, bus_code, shunt, series = line.split(',')
            yield {
                'line_number': int(line_number),
                'bus_code': bus_code,
                'shunt': complex(shunt),
                'series': complex(series),
            }
            idx += 1


def parse_admitance_file_no_yield(file_path):
    """
    Parse the admitance file to extract the number of edges, number of nodes,
    node array, and edge array.

    Parameters:
        file_path (str): Path to the file.

    Returns:
        tuple: no_edges (int), no_nodes (int), node_array (list), edge_array (list of tuples)
    """
    with open(file_path, 'r') as file:
        # Skip lines that are empty or start with '#'
        lines = [line.strip() for line in file if line.strip() and not line.strip().startswith('#')]
        idx = 0

        # Read the first line to get no_edges and no_nodes
        first_line = lines[idx]
        no_edges, no_nodes = map(int, first_line.split(','))
        idx += 1

        # Read the second line to get the node array
        second_line = lines[idx]
        node_array = list(map(int, second_line.split(',')))
        idx += 1

        # Read the next no_edges lines to get the edge array
        edge_array = []
        for _ in range(no_edges):
            edge_line = lines[idx]
            edge = tuple(map(int, edge_line.split(',')))
            edge_array.append(edge)
            idx += 1

        lines_data = []
        for _ in range(no_edges):
            line = lines[idx]
            line_number, bus_code, shunt, series, shunt_matrix = line.split(',')
            line_data = {
                'line_number': int(line_number),
                'bus_code': bus_code,
                'shunt': complex(shunt),
                'series': complex(series),
                'shunt_admittance': complex(shunt_matrix)
            }
            lines_data.append(line_data)
            idx += 1

        Y = np.array([line['series'] for line in lines_data])

    return no_edges, no_nodes, node_array, edge_array, Y


def calculate_node_degrees(edges):
    """
    Calculate the degree of each node in the graph.

    Parameters:
        edges (list of tuples): List of edges as (from_node, to_node).

    Returns:
        dict: A dictionary with nodes as keys and their degrees as values.
    """
    from collections import defaultdict
    degree = defaultdict(int)
    for from_node, to_node in edges:
        degree[from_node] += 1
        degree[to_node] += 1
    return dict(degree)


def find_connected_components(nodes, edges):
    """
    Find connected components in the graph.

    Parameters:
        nodes (list): List of nodes.
        edges (list of tuples): List of edges as (from_node, to_node).

    Returns:
        list of sets: Each set contains nodes in a connected component.
    """
    import networkx as nx
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return [set(component) for component in nx.connected_components(G)]


def adjacency_matrix(nodes, edges):
    """
    Generate the adjacency matrix of the graph.

    Parameters:
        nodes (list): List of nodes.
        edges (list of tuples): List of edges as (from_node, to_node).

    Returns:
        numpy.ndarray: Adjacency matrix of the graph.
    """
    import numpy as np
    num_nodes = len(nodes)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for from_node, to_node in edges:
        adj_matrix[from_node - 1][to_node - 1] = 1
        adj_matrix[to_node - 1][from_node - 1] = 1
    return adj_matrix


def adjacency_to_singular(adj_matrix):
    """
    Convert an adjacency matrix to a singular transformation matrix.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        numpy.ndarray: Singular transformation matrix.
    """
    num_nodes = adj_matrix.shape[0]
    edges = []

    # Extract edges from the adjacency matrix
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid double counting
            if adj_matrix[i, j] != 0:
                edges.append((i + 1, j + 1))  # Convert to 1-based indexing

    # Create the incidence matrix
    num_edges = len(edges)
    incidence_matrix = np.zeros((num_nodes, num_edges), dtype=int)

    for edge_index, (from_node, to_node) in enumerate(edges):
        incidence_matrix[from_node - 1, edge_index] = 1  # Outgoing edge
        incidence_matrix[to_node - 1, edge_index] = -1  # Incoming edge

    # Singular transformation matrix is the transpose of the incidence matrix
    return np.transpose(incidence_matrix)


def singular_transformation_matrix(V, edges):
    """
    Generate a singular transformation matrix for a graph G = (V, e).

    Parameters:
        V (list): List of nodes (buses).
        edges (list of tuples): List of edges (connections) as (from_node, to_node).

    Returns:
        numpy.ndarray: Singular transformation matrix.
    """
    num_nodes = len(V)
    num_edges = len(edges)

    # Initialize the incidence matrix
    A = np.zeros((num_nodes, num_edges), dtype=int)
    # Populate the incidence matrix
    for edge_index, (from_node, to_node) in enumerate(edges):
        A[from_node - 1, edge_index] = 1  # Outgoing edge
        A[to_node - 1, edge_index] = -1  # Incoming edge

    # Singular transformation matrix is the incidence matrix
    return np.transpose(A)


# Example usage
def get_admitance(file_path):
    """
    Get the admittance matrix from the admitance file.

    Parameters:
        file_path (str): Path to the admitance file.

    Returns:
        numpy.ndarray: Admittance matrix.
    """
    data_generator = parse_admitance_file_yield(file_path)

    # Retrieve the first yielded value (graph structure)
    no_edges, no_nodes, nodes, edges, shunt_admitances = next(data_generator)
    line_charging_array, series_array = zip(*[(line['shunt'], line['series']) for line in data_generator])

    # Convert to numpy arrays
    # line_charging_array = np.transpose(np.array(line_charging_array))
    Y = np.array(series_array)
    print(Y)
    line_charging = np.diag(line_charging_array)
    shunt = np.diag(shunt_admitances)  # NBxNB matrix

    # Convert to numpy arrays

    #print(f"No. of edges: {no_edges}")
    #print(f"No. of nodes: {no_nodes}")
    #print(f"Node array: {nodes}")
    #print(f"Edge array: {edges}")

    # Because there are no shunt admittances, we can set the shunt matrix to zero
    # and the Y matrix to the series admittance matrix

    # The Y matrix is the series admittance matrix and becasue we are using a singular transformation.
    # we need to add the Y matrix to the diagonal of the Y_n matrix.

    # Additionaly, we can set all the admitances in the diagonal to zero for the
    # shunt elements and the series elements in the rest of the diagonal.

    Y_n = np.identity(no_nodes + no_edges, dtype=complex)
    A = singular_transformation_matrix(nodes, edges)
    A_n = np.vstack((shunt, A))

    # Create the shunt admittance matrix
    for i in range(no_nodes + no_edges):
        Y_n[i][i] = 0.0j if i < no_nodes else 1/Y[i - no_nodes]

    #shunt_admitances = np.array([0.06j,0.05j,0.04j,0.04j,0.03j,0.02j,0.05j])
    #if (len(edges) == len(shunt_admitances)):
    #    shunt_matrix = build_shunt_admittance_matrix(edges,shunt_admitances, no_nodes)
    #else:
    #    print(f"The number of edges {len(edges)} is not equal to the number of shunt admittances {len(shunt_admitances)}")
    #    raise (ValueError)


    admitance_matrix = np.transpose(A_n) @ Y_n @ A_n
    return (A, admitance_matrix+shunt)

def admitance_rect_to_polar(Y_rect, angle_in_degrees=True):
    dec_places =3
    NB = Y_rect.shape[0]
    Y_polar = np.empty((NB, NB), dtype=object)
    for i in range(NB):
        for j in range(NB):
            mag = float(np.round(np.abs(Y_rect[i, j]),dec_places))
            ang = float(np.round(np.angle(Y_rect[i, j], deg=angle_in_degrees),dec_places))
            Y_polar[i, j] = (mag, ang)
    return Y_polar

def build_shunt_admittance_matrix(edges, admittances, num_nodes):
    """
    Build the shunt admittance matrix.

    Parameters:
        edges (list of tuples): List of edges as (from_node, to_node).
        admittances (list of complex): List of admittance values for each edge.
        num_nodes (int): Number of nodes in the graph.

    Returns:
        numpy.ndarray: Shunt admittance matrix (nxn).

    Example:
        Z[0][0] += 0.11j  # shunt where line 1 is all the places:1-2 1-3
        Z[1][1] += 0.17j  # shunt where line 2 is all the places:1-2 2-3 2-4 2-5
        Z[2][2] += 0.11j  # shunt where line 3 is all the places:1-3 2-3 3-4
        Z[3][3] += 0.11j  # shunt where line 4 is all the places:2-4 3-4 4-5
        Z[4][4] += 0.08j  #

    """
    shunt_matrix = np.zeros((num_nodes, num_nodes), dtype=complex)

    for (from_node, to_node), admittance in zip(edges, admittances):
        shunt_matrix[from_node - 1, from_node - 1] += admittance
        shunt_matrix[to_node - 1, to_node - 1] += admittance

    return shunt_matrix

A, Y = get_admitance('new_book_6_9.txt')

print(admitance_rect_to_polar(Y))
