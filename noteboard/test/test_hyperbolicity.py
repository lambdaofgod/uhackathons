import pytest
import networkx as nx
from noteboard.hyperbolicity import CCLHyperbolicity, NaiveHyperbolicity


@pytest.fixture
def example_graphs():
    return {
        "square grid": nx.grid_2d_graph(10, 10),
        "rectangular grid": nx.grid_2d_graph(2, 10),
        "circulant graph": nx.circulant_graph(16, [2]),
        "petersen graph": nx.petersen_graph(),
        "watts-strogatz graph": nx.watts_strogatz_graph(20, k=4, p=0.1, seed=0),
        "karate club graph": nx.karate_club_graph(),
        "les miserables graph": nx.les_miserables_graph(),
        "florentine families graph": nx.florentine_families_graph()
    }


def test_hyperbolicity_implementations_equal(example_graphs):
    """
    naive hyperbolicity implements straightforward algorithm
    that serves as a ground truth
    """
    debug = False
    for (name, graph) in example_graphs.items():
        graph_dists = nx.floyd_warshall_numpy(graph)
        delta_ccl = CCLHyperbolicity.get_hyperbolicity(
            graph_dists, debug=debug)
        delta_naive = NaiveHyperbolicity.get_hyperbolicity(
            graph_dists, debug=debug)
        assert delta_ccl == delta_naive
