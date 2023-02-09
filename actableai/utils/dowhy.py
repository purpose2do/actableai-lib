from io import StringIO
from networkx import relabel_nodes
from networkx.drawing.nx_pydot import write_dot


def causal_model_to_dot(causal_model):
    graph = causal_model._graph._graph

    node_map = {}

    for node in graph.nodes:
        if ':' in node:
            node_map[node] = f'"{node}"'

    new_graph = relabel_nodes(graph, node_map)

    buffer = StringIO()
    write_dot(new_graph, buffer)

    return buffer.getvalue()
