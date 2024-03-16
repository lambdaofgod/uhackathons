import numba
import networkx as nx
import numpy as np
import abc
from noteboard import hyperbolicity_impl

__all__ = ["HyperbolicityAlgorithm", "CCLHyperbolicity", "NaiveHyperbolicity"]


class HyperbolicityAlgorithm(abc.ABC):

    @abc.abstractmethod
    def get_hyperbolicity(cls, graph_dists: np.ndarray, debug=False):
        pass


class CCLHyperbolicity(HyperbolicityAlgorithm):

    @classmethod
    def _get_sorted_graph_dists_with_indices(cls, graph_dists):
        graph_dists_flat = graph_dists.reshape(-1)
        graph_dists_flat_indices = (-graph_dists_flat).argsort()
        # graph_dists_flat = graph_dists_flat[graph_dists_flat > 0]
        return graph_dists_flat_indices, graph_dists_flat[graph_dists_flat_indices]

    @classmethod
    def get_hyperbolicity(cls, graph_dists, debug=False):
        sorted_indices, sorted_dists = cls._get_sorted_graph_dists_with_indices(
            graph_dists)
        return hyperbolicity_impl.get_delta_hyperbolicity_ccl(graph_dists, sorted_indices, sorted_dists, debug=debug)


class NaiveHyperbolicity(HyperbolicityAlgorithm):

    @classmethod
    def get_hyperbolicity(cls, graph_dists, debug=False):
        return hyperbolicity_impl.get_delta_hyperbolicity_naive(graph_dists, debug=debug)
