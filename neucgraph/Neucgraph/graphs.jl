using Distributions
using Manopt
using Manifolds
using LightGraphs

## MODULE

using ManifoldDiff
using Neucgraph.OrgGraph
using Neucgraph.NamedGraphs
using FiniteDifferences, ManifoldDiff


org_links = OrgGraph.load_links("../data/org_roam_records_2023_10_16.json")

edges = [(link.source_name, link.destination_name) for link in org_links]

named_graph = NamedGraphs.from_edges(edges)

all_positive_indices = NamedGraphs.adjlist(named_graph)
all_negative_indices = NamedGraphs.neg_adjlist(named_graph)


struct NeighborSampler
    all_positive_indices::Vector{Vector{Int64}}
    all_negative_indices::Vector{Vector{Int64}}

    function NeighborSampler(
        graph::NamedGraph
    )
        all_positive_indices = NamedGraphs.adjlist(named_graph)
        all_negative_indices = NamedGraphs.neg_adjlist(named_graph)
        new(all_positive_indices, all_negative_indices)
    end
end


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

sampler = NeighborSampler(named_graph)

left, right = sample_batch(sampler, [1, 2, 3, 4])

M = Hyperbolic(2)
X = rand(M, named_graph.v_size)

X_anchors = X[positive_anchors]
X_positive = X[positive_indices]

function neighbor_distance(M, X, positive_anchors, positive_indices)
    d = distance.(Ref(M), X[positive_anchors], X[positive_indices])
    sum(d)
end


neighbor_distance(M, X, positive_anchors, positive_indices)


function negative_distance()


f(X) = distance(M, X, X)
grad_f(M, X) = Manifolds.gradient(M, f, X)

r_backend = ManifoldDiff.TangentDiffBackend(
    ManifoldDiff.FiniteDifferencesBackend()
)

function g(X, Y)
    distance(M, X, Y)
end

ManifoldDiff.gradient(M, g, (X[1], X[2]), r_backend)
