# This problem is from the book Power System Engineering, 3rd Edition
# by D.P. Kothari. The theory is based on Chapter 6. The data is from the same book according to example 6.4
import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

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
        # Read the first line to get no_edges and no_nodes
        first_line = file.readline().strip()
        no_edges, no_nodes = map(int, first_line.split(','))

        # Read the second line to get the node array
        second_line = file.readline().strip()
        node_array = list(map(int, second_line.split(',')))

        # Read the next no_edges lines to get the edge array
        edge_array = []
        for _ in range(no_edges):
            edge_line = file.readline().strip()
            edge = tuple(map(int, edge_line.split(',')))
            edge_array.append(edge)

        # Yield the basic graph structure
        yield no_edges, no_nodes, node_array, edge_array

        # Yield each line's admittance data
        for _ in range(no_edges):
            line = file.readline().strip()
            line_number, bus_code, shunt, series = line.split(',')
            yield {
                'line_number': int(line_number),
                'bus_code': bus_code,
                'shunt': complex(shunt),
                'series': complex(series)
            }

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
        # Read the first line to get no_edges and no_nodes
        first_line = file.readline().strip()
        no_edges, no_nodes = map(int, first_line.split(','))

        # Read the second line to get the node array
        second_line = file.readline().strip()
        node_array = list(map(int, second_line.split(',')))

        # Read the next no_edges lines to get the edge array
        edge_array = []
        for _ in range(no_edges):
            edge_line = file.readline().strip()
            edge = tuple(map(int, edge_line.split(',')))
            edge_array.append(edge)

        lines = []
        for _ in range(no_edges):
            line = file.readline().strip()
            line_number, bus_code, shunt, series = line.split(',')
            line_data = {
                'line_number': int(line_number),
                'bus_code': bus_code,
                'shunt': complex(shunt),
                'series': complex(series)
            }
            lines.append(line_data)

        Y = np.array([line['series'] for line in lines])

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

# Example usage
file_path = 'Admitance_3.txt'

data_generator = parse_admitance_file_yield(file_path)

# Retrieve the first yielded value (graph structure)
no_edges, no_nodes, nodes, edges = next(data_generator)

# Retrieve the admittance values (remaining yielded values)
Y = np.array([line['series'] for line in data_generator])

print(f"No. of edges: {no_edges}")
print(f"No. of nodes: {no_nodes}")
print(f"Node array: {nodes}")
print(f"Edge array: {edges}")
print(f"Admittance: {Y}")

# Because there are no shunt admittances, we can set the shunt matrix to zero
# and the Y matrix to the series admittance matrix
shunt = np.zeros((no_nodes, no_nodes), dtype=complex)  # NBxNB matrix

# The Y matrix is the series admittance matrix and becasue we are using a singular transformation.
# we need to add the Y matrix to the diagonal of the Y_n matrix.

# Additionaly, we can set all the admitances in the diagonal to zero for the
# shunt elements and the series elements in the rest of the diagonal.

Y_n = np.identity(no_nodes + no_edges, dtype=complex)
A = singular_transformation_matrix(nodes, edges)

A_n = np.vstack((shunt,A))

for i in range(no_nodes + no_nodes):
    Y_n[i][i] = 0.0 if i < no_nodes else Y_n[i][i] * Y[i - no_nodes]

print(np.transpose(A_n)@Y_n@A_n)

# It is possible to create a singular transformation matrix from the adjacency matrix
#adj_matrix = adjacency_matrix(nodes, edges)
#singular_matrix = adjacency_to_singular(adj_matrix)
#print("Singular Transformation Matrix:\n", singular_matrix)
