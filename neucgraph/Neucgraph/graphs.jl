using Distributions
using Manopt
using Manifolds
using LightGraphs
using ManifoldDiff

## MODULE

import Neucgraph.OrgGraph
import Neucgraph.NamedGraphs

org_links = OrgGraph.load_links("../data/org_roam_records_2023_10_16.json")

edges = [(link.source_name, link.destination_name) for link in org_links]

named_graph = NamedGraphs.from_edges(edges)

positive_indices = NamedGraphs.adjlist(named_graph)
negative_indices = NamedGraphs.neg_adjlist(named_graph)


function sample_neighbor_indices(neighbors, vertices, n)
    anchors = Int64[]
    neighbor_indices = Int64[]
    for i in vertices
        v_neighbors = sample(neighbors[i], n)
        v_anchors = fill(i, length(v_neighbors))
        append!(anchors, v_anchors)
        append!(neighbor_indices, v_neighbors)
    end
    (anchors, neighbor_indices)
end

## Manifold and initial embeddings
M = Hyperbolic(2)
X = rand(M, named_graph.v_size)

(positive_anchor_indices, sampled_positive_indices) = sample_neighbor_indices(positive_indices, [1, 2], 2)
(negative_anchor_indices, sampled_negative_indices) = sample_neighbor_indices(negative_indices, [1, 2], 2)

X_anchor = X[positive_anchor_indices]
X_positive = X[sampled_positive_indices]

function get_manifold_distance(M, X)
    n = trunc(Int, size(X)[1] / 2)
    X_anchor = X[1:n]
    X_positive = X[n+1:end]
    sum(distance.(Ref(M), X_anchor, X_positive))
end

## Vectorizing distance computation
X_mat = hcat(X...)'

X_sample_mat = X_mat[vcat(positive_anchor_indices, sampled_positive_indices),:]
X_sample = X[vcat(positive_anchor_indices, sampled_positive_indices)]

distance(Ref(M), X_sample_mat[1:4, :], X_sample_mat[5:end,:])

ManifoldDiff.gradient(M, get_manifold_distance, X_sample)

function sample_positive_negative_indices(anchor_indices)

end


p0 = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
size(p0)
