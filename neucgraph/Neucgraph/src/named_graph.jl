module NamedGraphs

using Distributions
using Manopt
using Manifolds
using LightGraphs

struct NodeInfo
    name::String
end

struct NamedGraph
    graph::SimpleGraph
    names_to_indices::Dict{String,Int64}
    indices_to_names::Vector{String}
    v_size::Int64
    e_size::Int64

    function NamedGraph(;
        graph,
        names_to_indices
    )
        indices_to_names = sort(names_to_indices; byvalue=true) |> keys |> collect
        v_size = length(names_to_indices)
        e_size = graph.ne
        new(graph, names_to_indices, indices_to_names, v_size, e_size)
    end

end

function from_edges(edges)
    vertices = [v for e in edges for v in collect(e)] |> Set |> collect
    vertices_dict = Dict([v => i for (i, v) in enumerate(vertices)])
    g = SimpleGraph(vertices |> length)
    for (v, w) in edges
        add_edge!(g, vertices_dict[v], vertices_dict[w])
    end
    NamedGraph(graph=g, names_to_indices=vertices_dict)
end


function neighbors(ng::NamedGraph, vertex_name::String)
    nbs = LightGraphs.neighbors(ng.graph, ng.names_to_indices[vertex_name])
    ng.indices_to_names[nbs]
end

function adjlist(ng::NamedGraph)
    ng.graph.fadjlist
end

function neg_adjlist(ng::NamedGraph)
    vs = 1:ng.v_size |> collect
    [
        setdiff(vs, vcat([v_neighbors, i]...))
        for (i, v_neighbors) in enumerate(ng.graph.fadjlist)
    ]
end

end
