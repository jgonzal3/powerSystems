import numpy as np

def read_admitance_file(file_path):
    try:
        with open(file_path, 'r') as file:
            # Read the first line to get NL and NB
            first_line = file.readline().strip()
            NL, NB = map(int, first_line.split(','))

            # Read the remaining lines to get the line data
            lines = []
            for _ in range(NL):
                line = file.readline().strip()
                line_number, bus_code, shunt, series = line.split(',')
                line_data = {
                    'line_number': int(line_number),
                    'bus_code': bus_code,
                    'shunt': complex(shunt),
                    'series': complex(series)
                }
                lines.append(line_data)

            return NL, NB, lines

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        raise
    except ValueError as e:
        print(f"Error parsing file: {e}")
        raise

# Example usage
def get_admitance(file_path):
    NL, NB, lines = read_admitance_file(file_path)
    Y = np.zeros((NB, NB), dtype=complex)
    for i in lines:
        l, m = (int(V1) - 1 for V1 in i['bus_code'].split('-'))
        yi = 1/i['series']
        shunt = i['shunt']
        Y[l][l] = Y[l][l] + yi + shunt
        Y[m][m] = Y[m][m] + yi + shunt
        Y[l][m] = Y[l][m] - yi + shunt
        Y[m][l] = Y[m][l] - yi + shunt

    return (np.round(Y,4))

def get_admitance_from_impidence(file_path):
    NL, NB, lines = read_admitance_file(file_path)
    Y = np.zeros((NB, NB), dtype=complex)
    for i in lines:
        l, m = (int(V1) - 1 for V1 in i['bus_code'].split('-'))
        yi = i['series']
        shunt = i['shunt']
        Y[l][l] = Y[l][l] + yi + shunt
        Y[m][m] = Y[m][m] + yi + shunt
        Y[l][m] = Y[l][m] - yi + shunt
        Y[m][l] = Y[m][l] - yi + shunt

    return (np.round(Y,4))

y = get_admitance('Admitance_1.txt')
y = get_admitance_from_impidence('Admitance_2.txt')
