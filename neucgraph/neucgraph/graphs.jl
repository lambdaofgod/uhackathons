using Distributions
using Manopt
using Manifolds
using LightGraphs

g = SimpleGraph(4)
add_edge!(g, 1, 2)
add_edge!(g, 1, 3)
add_edge!(g, 2, 4)
add_edge!(g, 3, 4)
add_edge!(g, 2, 3)

neighbors.(Ref(g), [1,2])


indices = 1:size(graph_dists)[1] |> collect
neighbor_indices = [
    [i for (i, d) in enumerate(ds) if d == 1]
    for ds in graph_dists
]
all_negative_indices = [
    [i for (i, d) in enumerate(ds) if d > 1]
    for ds in graph_dists
]
positive_indices = neighbor_indices[1]
negative_indices = sample(all_negative_indices[1], 2, replace=false)

X = rand(M, 4)
M = Hyperbolic(2)


function sample_positive_negative_indices(anchor_indices)

end
