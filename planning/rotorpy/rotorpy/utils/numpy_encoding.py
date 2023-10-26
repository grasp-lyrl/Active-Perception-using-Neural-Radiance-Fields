import json
import numpy as np

def to_ndarray(obj, dtype=np.float64):
    """
    Greedily and recursively convert the given object to a dtype ndarray.
    """
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = to_ndarray(obj[k])
        return obj
    elif isinstance(obj, list):
        try:
            return np.array(obj, dtype=dtype)
        except:
            return [to_ndarray(o) for o in obj]
    else:
        return obj

class HelperNumpyJSONEncoder(json.JSONEncoder):
    """
    This encoder encodes Numpy arrays as lists.
    """
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

class NumpyJSONEncoder(json.JSONEncoder):
    """
    This encoder will print an entire collection onto a single line if it fits.
    Otherwise the individual elements are printed on separate lines. Numpy
    arrays are encoded as lists.

    This class is derived from contributions by Tim Ludwinski and Jannis
    Mainczyk to a stackoverflow discussion:
    https://stackoverflow.com/questions/16264515/json-dumps-custom-formatting
    """

    MAX_WIDTH = 80 # Maximum length of a single line list.
    MAX_ITEMS = 80 # Maximum number of items in a single line list.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indentation_level = 0

    def encode(self, o):
        # If o fits on a single line, do so.
        line = json.dumps(o, cls=HelperNumpyJSONEncoder)
        if len(line) <= self.MAX_WIDTH:
            return line
        # Otherwise, break o into pieces.
        else:
            # If a list, split each entry into a separate line.
            if isinstance(o, (list, tuple)):
                self.indentation_level += 1
                output = [self.indent_str + self.encode(el) for el in o]
                self.indentation_level -= 1
                return "[\n" + ",\n".join(output) + "\n" + self.indent_str + "]"
            # If a dict, each key/value pair into a separate line.
            if isinstance(o, dict):
                self.indentation_level += 1
                output = [self.indent_str + f"{json.dumps(k)}: {self.encode(v)}" for k, v in o.items()]
                self.indentation_level -= 1
                return "{\n" + ",\n".join(output) + "\n" + self.indent_str + "}"
            # Otherwise use default encoding.
            return json.dumps(o)

    @property
    def indent_str(self) -> str:
        if self.indent == None:
            indent = 0
        else:
            indent = self.indent
        return " " * self.indentation_level * indent

if __name__ == '__main__':
    import copy

    # Example data.
    data = {
        'bounds': {'extents': [0, 5.0, 0, 2.0, 0, 13.0]},
        'blocks': [
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0], 'color': [1, 0, 0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0]},
            {'extents': [2, 3, 0.0, 2, 0.0, 10.0]}],
        'start': np.array([0, 0, 1]),
        'goal': np.array([4, 0, 2]),
        'resolution': np.array([0.25, 0.25, 0.5]),
        'margin': 0.1,
        'expected_path_length': 20.52
    }
    data['more'] = copy.deepcopy(data)

    # Print JSON string to terminal.
    print(json.dumps(data, cls=NumpyJSONEncoder, indent=4))

    # Using 'dump' not yet supported.
    with open('example.json', 'w') as file:
        file.write(json.dumps(data, cls=NumpyJSONEncoder, indent=4))

    with open('example.json') as file:
        data_out = json.load(file)
    data_out = to_ndarray(data_out)

    print(data_out)
