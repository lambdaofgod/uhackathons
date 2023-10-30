using Distributions
using Manopt
using Manifolds
using LightGraphs

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

sample_neighbor_indices(positive_indices, [1, 2], 2)

X = rand(M, 4)
M = Hyperbolic(2)


function sample_positive_negative_indices(anchor_indices)

end
