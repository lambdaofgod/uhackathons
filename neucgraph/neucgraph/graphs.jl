using Distributions
using Manopt
using Manifolds
using LightGraphs


module GraphUtils

using Distributions
using Manopt
using Manifolds
using LightGraphs

struct NodeInfo
    name::String
end

struct NamedGraph
    graph::SimpleGraph
    names::Dict{String,Int64}
end


function make_named_graph(edges)
    vertices = [v for e in edges for v in collect(e)] |> Set |> collect
    vertices_dict = Dict([v => i for (i, v) in enumerate(vertices)])
    g = SimpleGraph(vertices |> length)
    for (v, w) in edges
        add_edge!(g, vertices_dict[v], vertices_dict[w])
    end
    NamedGraph(g, vertices_dict)
end


function neighbors(ng::NamedGraph, vertex_name::String)
    LightGraphs.neighbors(ng.graph, ng.names[vertex_name])
end

end

edges = [("a", "b"), ("c", "d"), ("b", "c")]

named_graph = GraphUtils.make_named_graph(edges)

GraphUtils.neighbors.(Ref(named_graph), ["a", "b"])


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
