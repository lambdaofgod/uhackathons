"""Utilities for converting between Weaviate and pytrec_eval formats."""

import pytrec_eval
from typing import Dict, Any, List, Optional
import weaviate


def weaviate_results_to_pytrec_dict(weaviate_results, is_hybrid: bool = False) -> Dict[str, float]:
    """
    Converts Weaviate search results to the format expected by pytrec_eval.run.

    Args:
        weaviate_results: The raw results from a Weaviate query (Python dictionary).
        is_hybrid: A boolean indicating if the search was hybrid (affects score handling).

    Returns:
        A dictionary suitable for use with pytrec_eval, where keys are document IDs
        and values are relevance scores.
    """
    run_dict = {}

    if not weaviate_results:
        return run_dict

    for result in weaviate_results:
        doc_id = result.properties.get('i', result.id)
        if is_hybrid:
            relevance_score = result.metadata.score
        else:
            relevance_score = 1 - result.metadata.distance  # Invert distance for similarity

        run_dict[doc_id] = relevance_score

    return run_dict


def perform_search_and_evaluate(
    collection,
    query: str,
    query_id: str,
    qrel_dict: Dict[str, Dict[str, int]],
    properties: List[str],
    query_type: str = "bm25",
    alpha: float = 0.5
) -> Dict[str, Dict[str, float]]:
    """
    Performs a search using specified parameters, converts results to pytrec_eval
    format, and adds them to the run dictionary.

    Args:
        collection: Weaviate collection object.
        query: The search query.
        query_id: ID of the query.
        qrel_dict: Dictionary containing the ground truth relevance judgments.
        properties: Properties to retrieve in search results.
        query_type: Type of search query. Default is "bm25".
        alpha: Parameter used in hybrid search. Default is 0.5.

    Returns:
        A dictionary representing the run for this query, suitable for pytrec_eval.
    """
    run_dict = {}
    if query_type == "bm25":
        results = collection.query.bm25(
            query=query,
            return_properties=properties,
            limit=10
        ).with_additional(["distance"]).objects

    elif query_type == "hybrid":
        results = collection.query.hybrid(
            query=query,
            alpha=alpha,
            return_properties=properties,
            limit=10,
            fusion_type="relativeScore"
        ).with_additional(["score"]).objects

    else:
        raise ValueError(f"Invalid query type: {query_type}")

    run_dict[query_id] = weaviate_results_to_pytrec_dict(
        results,
        is_hybrid=(query_type == "hybrid")
    )
    return run_dict


def evaluate_search_results(
    run_dict: Dict[str, Dict[str, float]],
    qrel_dict: Dict[str, Dict[str, int]],
    metrics: Optional[set] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates search results using pytrec_eval.

    Args:
        run_dict: Dictionary containing search results in pytrec_eval format.
        qrel_dict: Dictionary containing relevance judgments.
        metrics: Set of metric names to compute. Defaults to {"map", "ndcg"}.

    Returns:
        Dictionary containing evaluation results per query and metric.
    """
    if metrics is None:
        metrics = {"map", "ndcg"}

    evaluator = pytrec_eval.RelevanceEvaluator(qrel_dict, metrics)
    return evaluator.evaluate(run_dict)
