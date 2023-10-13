### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ 5a980518-667b-11ee-1701-8b9b21975813
begin
	using Pkg
	Pkg.activate("../neucgraph/")
	using DataFrames
	using LinearAlgebra
	using LinearAlgebra
	using PyPlot
	using MultivariateStats
	using InMemoryDatasets
	using LightGraphs # graph library
	using SimpleWeightedGraphs
	using StatisticalGraphics
	ENV["LD_LIBRARY_PATH"] = ""
end

# ╔═╡ 461df8e8-cb6d-4927-8f31-48b591fbc194
begin
	using JSON
	path = "../data/org_roam_records_2023_10_08.json"
end

# ╔═╡ 4115223d-d5ca-43df-98f8-a30aac00ed94
using ManifoldLearning

# ╔═╡ 3cb49c9e-793a-4bc7-a6f6-6fddc72f0ae4
begin
	using Manifolds
	using Manopt
	using Random
	using Plots
	Random.seed!(42)
end

# ╔═╡ 3a7f0c50-a31e-41d0-a687-72957dafeec6


# ╔═╡ b1c737a4-ba9c-4335-b2c3-73d72a79e79e
records = JSON.parse(read(path, String));

# ╔═╡ c24341de-1bbc-4d2b-bae8-c19ebbc891af
@kwdef struct OrgLink
    source_name :: String
    destination_name :: String
    source_date :: String
    destination_date :: String
end

# ╔═╡ 0fb53a68-4a57-4556-bc68-6980f57fbcd9
function get_input_dict(struct_type, dict)
    fields = fieldnames(struct_type)
    Dict(
        [Symbol(k) => v for (k, v) in dict if Symbol(k) in fields]
    )
end

# ╔═╡ aad1a5eb-191a-470f-bcda-298c46a27c71
function dict_to_struct(struct_type, dict)
    mapped_dict = get_input_dict(struct_type, dict)
    struct_type(;mapped_dict...)
end

# ╔═╡ 9f4bd24c-0c0d-4837-ac49-76253f9b7741
record = records[1];

# ╔═╡ 300ecd9f-37f9-484c-b41f-8a140d6d87d0
begin
	fields = fieldnames(OrgLink)
	mapped_record = Dict(
	    [Symbol(k) => v for (k, v) in record if Symbol(k) in fields]
	)
end

# ╔═╡ 0bcadf0d-3e57-436a-840d-3526f7d82fe9
links = [dict_to_struct(OrgLink, record) for record in records];

# ╔═╡ 1a02921e-94b0-4dd7-8828-806910392505
begin
	
	struct LabeledGraph
	    graph
	    node_to_index :: Dict{String, Int64}    
	end
	
	function get_sorted_vertices(labeled_graph)
	    [name for (name, _) in sort(collect(labeled_graph.node_to_index), by=x -> x[2])]
	end
	    
	function make_graph(edges)
	    # Get unique nodes to determine # of vertices
	    nodes = unique(v for e in edges for v in e)
	
	    # Create a mapping from node names to indices
	    node_to_index = Dict(node => index for (index, node) in enumerate(nodes))
	    
	    num_vertices = length(nodes)
	
	    # Create an unweighted simple graph
	    G = SimpleGraph(num_vertices)
	
	    # Add edges to the graph
	    for edge in edges
	        add_edge!(G, node_to_index[edge[1]], node_to_index[edge[2]])
	    end
	
	    # Compute pairwise shortest path lengths
	    #dist_matrix = floyd_warshall_shortest_paths(G).dists
	    LabeledGraph(G, node_to_index)
	end
	
	function get_distance_matrix(labeled_graph :: LabeledGraph) 
	    floyd_warshall_shortest_paths(labeled_graph.graph).dists
	end
	
	
	function filter_small_connected_components(labeled_graph, min_component_size)
	    valid_vertices :: Vector{Int64} = []
	    for cc in weakly_connected_components(labeled_graph.graph) 
	        if length(cc) > min_component_size
	            valid_vertices = [valid_vertices; cc]
	        end
	    end
	    filtered_graph, _ = induced_subgraph(labeled_graph.graph, valid_vertices)
	    node_to_index = Dict(node => index for (node, index) in labeled_graph.node_to_index if index in valid_vertices)
	    LabeledGraph(filtered_graph, node_to_index)
	end
	
	link_edges = [(link.source_name, link.destination_name) for link in links]
	
	raw_labeled_graph = make_graph(link_edges)
end

# ╔═╡ 2f3a5bee-b904-468c-a4b5-0fcb83ad1a02
labeled_graph = filter_small_connected_components(raw_labeled_graph, 10)

# ╔═╡ 9f716477-f681-42d8-adc1-050e19f6da4d
md"""
## Now the hyperbolic stuff

https://browse.arxiv.org/pdf/1804.03329.pdf
"""

# ╔═╡ 39b08699-372b-47cd-9011-d2e52bc8f7b8
dists = (labeled_graph |> get_distance_matrix);

# ╔═╡ fc4d4a30-7616-4a8f-9c47-e90a17315e26
@kwdef struct HMDS
	hyperbolic_dists :: Matrix{Real}
	pca
end

# ╔═╡ 3f7642b4-e438-4e2b-80e8-44f7305a3a20
function fit_hmds(dists, dim=3)
	D = dists ./ maximum(dists)
	hyperbolic_dists = cosh.(D)
	hpca = MultivariateStats.fit(PCA, -hyperbolic_dists, maxoutdim=dim)
	HMDS(hyperbolic_dists=hyperbolic_dists, pca=hpca)
end

# ╔═╡ b3f10d7e-988e-4e84-9c4e-60f2665f8834
function get_embeddings(hmds :: HMDS, space::Symbol)
	X = hmds.pca.proj
	if space == :lorentz
		return X
	elseif space == :poincare
		return X[:, 2:3] ./ (1 .+ X[:,1])
	end
end

# ╔═╡ 30281eb5-222b-4c0a-8986-17407d9d6f7c
hmds = fit_hmds(dists);

# ╔═╡ 2b55695f-d53c-4b78-8093-6c8fa0610073
X = get_embeddings(hmds, :poincare);

# ╔═╡ c68d85dd-4961-4cc9-8912-821eb1ac8b35
ds_poincare = Dataset(x=X[:,1], y=X[:,2], name=get_sorted_vertices(labeled_graph));

# ╔═╡ 61391d79-07f9-489d-918b-65551e8fc9f8
sgplot(ds_poincare, StatisticalGraphics.Scatter(x=:x,y=:y,labelresponse=:name))

# ╔═╡ fa277357-69b9-4abe-a6fe-1346d7034a54
md"""
## Regular MDS
"""

# ╔═╡ 351af441-c417-43bb-974b-ba7a296c2b6a
begin
	mds = fit(MDS, dists * 1.0; distances=true, maxoutdim=2);
	Y_mds = predict(mds);
	ds_mds = Dataset(x=Y_mds[1,:], y=-Y_mds[2,:], name=get_sorted_vertices(labeled_graph));
	sgplot(ds_mds, StatisticalGraphics.Scatter(x=:x,y=:y,labelresponse=:name))
end

# ╔═╡ 83626734-c2b2-478b-aabd-0ef3604786b6
md"""
## Isomap
"""

# ╔═╡ ee66593f-93ab-4f92-9316-2e9361d9fa85
begin
	isomap = fit(Isomap, dists * 1.0);
	Y_isomap = predict(mds);
	ds_isomap = Dataset(x=Y_isomap[1,:], y=-Y_isomap[2,:], name=get_sorted_vertices(labeled_graph));
	sgplot(ds_isomap, StatisticalGraphics.Scatter(x=:x,y=:y,labelresponse=:name))
end

# ╔═╡ 9781e526-71b1-4e4d-a0e7-fe127104fbd9


# ╔═╡ 5baa86c4-1972-4779-9f54-8f0cc63b892d


# ╔═╡ 4527c79a-4657-4331-a9a0-3068021aff15
sgplot(ds_mds, StatisticalGraphics.Scatter(x=:x,y=:y,labelresponse=:name))

# ╔═╡ 0bd00c5a-00df-45f4-a2ad-5710fd85b40f


# ╔═╡ d2a79196-c747-438d-b6bf-047074dd99e9


# ╔═╡ 44d958ea-b0af-448a-8eef-c926ec2862a3


# ╔═╡ c31c7488-a8e0-4dbb-a43f-167645b44c8d


# ╔═╡ 7f082478-f761-4e1e-ab6c-66f0d643b760


# ╔═╡ ea18299b-e2ff-47ab-a4e4-6881a05e6be4


# ╔═╡ 5875994b-39da-4861-9e2c-5aaf12033a21
md"""
## Manifolds

2d Lorentz hyperbolic model is embedded in 3d
"""

# ╔═╡ bea8e470-8266-42c5-b399-38735fbff300
lorenz_model = Hyperbolic(2)

# ╔═╡ 30631dbc-c39f-4620-babe-24cdad1bebbe
begin
	pts =  [ [0.85*cos(φ), 0.85*sin(φ), sqrt(0.85^2+1)] for φ ∈ range(0,2π,length=11) ]
	scene = Plots.plot(lorenz_model, pts; geodesic_interpolation=100)
end

# ╔═╡ ef77dc32-1718-419e-8be1-362b1cd55e1b


# ╔═╡ 73d77fe2-d921-46c4-b760-60960b77f4d9


# ╔═╡ 3210cccf-4295-4382-b9c6-95e6d984c1dd
p=[0,0,1.0]

# ╔═╡ 3182dd25-262e-45f2-93c6-a201bd857833
# ╠═╡ disabled = true
#=╠═╡
data = [exp(lorentz_manifold, p,  0.01 * rand(lorentz_manifold)) for i in 1:length(X)];
  ╠═╡ =#

# ╔═╡ af50c3e3-a598-4241-805f-7f2a47e9ec38


# ╔═╡ f27ffd49-8f1b-4993-81c3-717913e762db


# ╔═╡ afa6047e-1049-4aa6-8332-af642da7f400


# ╔═╡ bd65f2c5-e0d2-4c18-8226-25a3a06ae099


# ╔═╡ 0e89d3c5-463c-4329-b11c-8586cc834fd5
data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];

# ╔═╡ 5078689e-81fc-4710-be33-9f17af00dd48
data

# ╔═╡ 616f27ef-f0f7-4628-bca7-eeb087707044


# ╔═╡ Cell order:
# ╠═5a980518-667b-11ee-1701-8b9b21975813
# ╠═3a7f0c50-a31e-41d0-a687-72957dafeec6
# ╠═461df8e8-cb6d-4927-8f31-48b591fbc194
# ╠═b1c737a4-ba9c-4335-b2c3-73d72a79e79e
# ╠═c24341de-1bbc-4d2b-bae8-c19ebbc891af
# ╠═0fb53a68-4a57-4556-bc68-6980f57fbcd9
# ╠═aad1a5eb-191a-470f-bcda-298c46a27c71
# ╠═9f4bd24c-0c0d-4837-ac49-76253f9b7741
# ╠═300ecd9f-37f9-484c-b41f-8a140d6d87d0
# ╠═0bcadf0d-3e57-436a-840d-3526f7d82fe9
# ╠═1a02921e-94b0-4dd7-8828-806910392505
# ╠═2f3a5bee-b904-468c-a4b5-0fcb83ad1a02
# ╟─9f716477-f681-42d8-adc1-050e19f6da4d
# ╠═39b08699-372b-47cd-9011-d2e52bc8f7b8
# ╠═fc4d4a30-7616-4a8f-9c47-e90a17315e26
# ╠═3f7642b4-e438-4e2b-80e8-44f7305a3a20
# ╠═b3f10d7e-988e-4e84-9c4e-60f2665f8834
# ╠═30281eb5-222b-4c0a-8986-17407d9d6f7c
# ╠═2b55695f-d53c-4b78-8093-6c8fa0610073
# ╠═c68d85dd-4961-4cc9-8912-821eb1ac8b35
# ╠═61391d79-07f9-489d-918b-65551e8fc9f8
# ╠═fa277357-69b9-4abe-a6fe-1346d7034a54
# ╠═351af441-c417-43bb-974b-ba7a296c2b6a
# ╠═83626734-c2b2-478b-aabd-0ef3604786b6
# ╠═4115223d-d5ca-43df-98f8-a30aac00ed94
# ╠═ee66593f-93ab-4f92-9316-2e9361d9fa85
# ╠═9781e526-71b1-4e4d-a0e7-fe127104fbd9
# ╠═5baa86c4-1972-4779-9f54-8f0cc63b892d
# ╠═4527c79a-4657-4331-a9a0-3068021aff15
# ╠═0bd00c5a-00df-45f4-a2ad-5710fd85b40f
# ╠═d2a79196-c747-438d-b6bf-047074dd99e9
# ╠═44d958ea-b0af-448a-8eef-c926ec2862a3
# ╠═c31c7488-a8e0-4dbb-a43f-167645b44c8d
# ╠═7f082478-f761-4e1e-ab6c-66f0d643b760
# ╠═ea18299b-e2ff-47ab-a4e4-6881a05e6be4
# ╠═5875994b-39da-4861-9e2c-5aaf12033a21
# ╠═3cb49c9e-793a-4bc7-a6f6-6fddc72f0ae4
# ╠═bea8e470-8266-42c5-b399-38735fbff300
# ╠═30631dbc-c39f-4620-babe-24cdad1bebbe
# ╠═ef77dc32-1718-419e-8be1-362b1cd55e1b
# ╠═73d77fe2-d921-46c4-b760-60960b77f4d9
# ╠═3210cccf-4295-4382-b9c6-95e6d984c1dd
# ╠═3182dd25-262e-45f2-93c6-a201bd857833
# ╠═af50c3e3-a598-4241-805f-7f2a47e9ec38
# ╠═f27ffd49-8f1b-4993-81c3-717913e762db
# ╠═afa6047e-1049-4aa6-8332-af642da7f400
# ╠═5078689e-81fc-4710-be33-9f17af00dd48
# ╠═bd65f2c5-e0d2-4c18-8226-25a3a06ae099
# ╠═0e89d3c5-463c-4329-b11c-8586cc834fd5
# ╠═616f27ef-f0f7-4628-bca7-eeb087707044
