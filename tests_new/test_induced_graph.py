import networkx as nx
import pytest

from src.encoding.decoder import is_induced_subgraph_by_features


def make_g(nodes, edges, attrs, node_key="feat"):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for n, v in attrs.items():
        G.nodes[n][node_key] = v
    G.add_edges_from(edges)
    return G

@pytest.mark.parametrize(
    "g1,g2,expected",
    [
        # #1: triangle (c,c,o) is present identically in g2
        (
                make_g([1,2,3], [(1,2),(2,3),(1,3)], {1:"c",2:"c",3:"o"}),
                make_g([1,2,3,9], [(1,2),(2,3),(1,3),(9,1)], {1:"c",2:"c",3:"o",9:"c"}),
                True,
        ),
        # #2: node IDs differ; features (c,c,o) still match a triangle in g2
        (
                make_g([2,3,4], [(2,3),(3,4),(2,4)], {2:"c",3:"c",4:"o"}),
                make_g([1,2,3,9], [(1,2),(2,3),(1,3),(9,1)], {1:"c",2:"c",3:"o",9:"c"}),
                True,
        ),
        # #3: feature multiset (c,n,o) doesn't exist in g2 (only c,c,o)
        (
                make_g([2,3,4], [(2,3),(3,4),(2,4)], {2:"c",3:"n",4:"o"}),
                make_g([1,2,3,9], [(1,2),(2,3),(1,3),(9,1)], {1:"c",2:"c",3:"o",9:"c"}),
                False,
        ),
    ],
)
def test_examples(g1, g2, expected):
    assert is_induced_subgraph_by_features(g1, g2, node_keys=["feat"]) is expected
