### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 5a980518-667b-11ee-1701-8b9b21975813
begin
	using Pkg
	Pkg.activate("../neucgraph/")
	using DataFrames
	using Distributions
	using LinearAlgebra
	using LogExpFunctions
	using LinearAlgebra
	using StatsBase
	using MultivariateStats
	using InMemoryDatasets
	using LightGraphs # graph library
	using SimpleWeightedGraphs
	
	using PyPlot
	using StatisticalGraphics
	using Plotly
	using PlutoPlotly
	
	using ManifoldLearning
	ENV["LD_LIBRARY_PATH"] = ""
end

# ╔═╡ 461df8e8-cb6d-4927-8f31-48b591fbc194
begin
	using JSON
	path = "../data/org_roam_records_2023_10_16.json";
	records = JSON.parse(read(path, String));
end

# ╔═╡ 3cb49c9e-793a-4bc7-a6f6-6fddc72f0ae4
begin
	using Manifolds
	using Manopt
	using Random
	using Plots
	Random.seed!(42)
end

# ╔═╡ d2b9ea9c-66e4-41cd-a961-216320c18aec
Manopt

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
labeled_graph = filter_small_connected_components(raw_labeled_graph, 25)

# ╔═╡ 79c84dfb-464a-49ab-a2a5-51972f94aebe
md"""
## Now the hyperbolic stuff
"""

# ╔═╡ 00d0c1db-2212-41ee-87c3-cca1b59229ce


# ╔═╡ 9f716477-f681-42d8-adc1-050e19f6da4d
md"""
## WARNING this works assuming dists come from Poincare model

Will have to either
- embed the points in Poincare space
- embed into Lorentz using method from [Learning Continuous Hierarchies in the Lorentz Model of Hyperbolic Geometry](https://arxiv.org/pdf/1806.03417.pdf)

[Representation Tradeoffs for Hyperbolic Embeddings](https://browse.arxiv.org/pdf/1804.03329.pdf)
"""

# ╔═╡ 6ab4f097-7ad8-45de-a136-12e7ad68b500


# ╔═╡ 6b181a52-ac04-458a-bbe7-cac7c903c275
function optim_f()
	n = 100
	σ = π / 8
	M = Sphere(2)
	p = 1 / sqrt(2) * [1.0, 0.0, 1.0]
	println(typeof(Ref(p)))
	data = [exp(M, p,  σ * rand(M; vector_at=p)) for i in 1:n];
	println(size(distance.(Ref(M), Ref(p), data)));
	f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), data) .^ 2)
	println(typeof(p))
	println(typeof(data))
	data
end

# ╔═╡ ed20f785-8f08-4c20-8dab-b1b1abc7ecc0
optim_f()

# ╔═╡ 4bf03d54-7e31-40d8-acbd-e895f5ee332e

function vectorized_manifold_dist(M, pv, data)
	for pi in eachindex(eachrow(pv))
		#println(typeof(pv[pi]))
		break
	end

	sum([distance.(Ref(M), Ref(pv[i,:]), data) for i in eachindex(eachrow(pv))])
end

# ╔═╡ 110de150-6244-4c16-b539-32a541d68a84
function grad_vectorized_manifold_dist(M, pv, data)
	n = size(data)[1]
	println(size(grad_distance.(Ref(M), data, Ref(pv[1,:]))))
	grads = [
		(1 / n .* grad_distance.(Ref(M), data, Ref(pv[i, :])))
		for i in eachindex(eachrow(pv))
	];
	println(typeof(grads))
	println(size(grads))
	sum(grads)
end

# ╔═╡ 7b9988dd-ecb6-46be-b60f-7f8ea77e8301
rng = MersenneTwister(0)

# ╔═╡ 4dfdbbcc-ee5c-4b04-af64-6477ccca72b8
begin	
	n = 100
	σ = π / 8
	M = Hyperbolic(2)
	#n_params = size(p_unnorm)[1];
	# println(typeof(p_unnorm))
	pv = hcat([rand(rng, M) for _ in 1:2]...)
	M_data = hcat([exp(M, pv[(i % 2) + 1,:],  σ * rand(M; vector_at=pv[(i % 2) + 1,:])) for i in 1:n]);
	#ds =  vectorized_manifold_dist(M, pv, M_data);
	#println(typeof(ds))
	f(M, p) = sum(1 / (2 * n) * distance.(Ref(M), Ref(p), M_data) .^ 2)
	grad_f(M, p) = sum(1 / n * grad_distance.(Ref(M), M_data, Ref(p)));
	
	F(M, pv) = sum(f.(Ref(M), pv))
	grad_F(M, pv) = sum(grad_f.(Ref(M), pv))
	#m1 = gradient_descent(M, f, grad_f, M_data)
end

# ╔═╡ 7fd32c56-bcbf-4cc4-a36e-756a25689ee9
(typeof(M_data), size(M_data))

# ╔═╡ 87dfa350-b8dd-43aa-87a5-b40bc35b0bd3
graph_dists = [
	[0, 2, 1, 2],
	[2, 0, 2, 1],
	[1, 2, 0, 2],
	[2, 1, 2, 0]
]

# ╔═╡ b16fa860-9add-4e25-b244-415e3570a13d
begin
	indices = 1:size(graph_dists)[1] |> collect
	neighbor_indices = [
		[i for (i, d) in enumerate(ds) if d == 1]
		for ds in graph_dists
	]
	all_negative_indices = [
		[i for (i, d) in enumerate(ds) if d > 1]
		for ds in graph_dists
	]
end

# ╔═╡ abff6230-0feb-4898-88a2-0f06e96ccae7
begin
	positive_indices = neighbor_indices[1]
	negative_indices = sample(all_negative_indices[1], 2, replace=false)
end

# ╔═╡ 0b928607-df5c-44c5-9b8a-69c06d726de7
X = rand(M, 4)

# ╔═╡ f18d4056-de9a-4187-bff3-ed02df186f72
function embedding_dist(M, point, other)
	distance.(Ref(M), Ref(point), other)
end

# ╔═╡ 46a9e180-832a-424c-a5c9-324e48e13f9b
anchor_indices = [1]

# ╔═╡ 87a63fc4-2aa5-468c-b6af-8523264afa4c


# ╔═╡ 70a7e25f-f8ac-4a0b-bc57-7efa15f92a16
function embedding_loss(M, point, X, positive_indices, negative_indices)
	negative_loss = logsumexp(embedding_dist(M, point, X[negative_indices]))
	positive_loss = sum(embedding_dist(M, point, X[positive_indices]))
	positive_loss + negative_loss
end

# ╔═╡ 4157355d-eabe-4386-9e29-7e79b15960b8
begin
	embedding_loss(M, X[1], X, positive_indices, negative_indices)
end

# ╔═╡ 8b845a7c-28f5-453b-8857-7ab3d5d197ce
embedding_loss.(Ref(M), X, Ref(X), neighbor_indices, all_negative_indices)

# ╔═╡ 014f62ab-ecf8-43fc-8b17-fb1099a3cded


# ╔═╡ 472d15f5-f4e2-454a-8dd9-aff21221d0c9
function embedding_loss_grad(M, point, X, positive_indices, negative_indices)
	Manopt.get_gradients(Ref(M), params -> embedding_loss(M, point, params, positive_indices, negative_indices), X)
end

# ╔═╡ dfd85b48-57f4-4852-8173-ac2f3fac42cf
begin
	#p0 =  [[1.0,0.0], [0.0,1.0], [0.0,0.0]] #[1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
	#p_start = p0 .- 1.0
	p_target = [1.0 0.0; 0.0 1.0; 0.0 0.0; 0.0 0.0; 0.0 0.0]
	p0 = p_target .- 1.0
end

# ╔═╡ fc6550b0-bb75-4fcd-a015-f67e71e93dfa
p_target

# ╔═╡ 4c2908da-e779-45a6-9863-e588ab4e06b0
p0

# ╔═╡ 41683763-7d32-4213-857b-45d0734f3cd9


# ╔═╡ b45cf641-a333-4e75-8e67-1ce58b36e26d
begin
	MEuc = Euclidean(2)

	foo(M, p) = distance(MEuc, p, p_target)
	foo_vect(M, p) = sum(distance.(Ref(MEuc), eachrow(p), eachrow(p_target)))
	grad_foo(M, p) = grad_distance(M, p, p_target)
	grad_foo_vect(M, p) = grad_distance.(Ref(M), eachrow(p), eachrow(p_target))

	grad_foo(MEuc, p0)
	#get_gradient(MEuc, foo, p)
	#(foo(MEuc, p0 .+ 1), sum(foo_vect(MEuc, p0 .+ 1)))
end

# ╔═╡ dc8d6982-f29e-43f0-b839-b9f803733339


# ╔═╡ bb08bb32-e0ec-40fa-8fe4-b399286383a1


# ╔═╡ 3615ec49-3777-4856-af60-c54b6306e2f6
gradient_descent(
	MEuc,
	foo_vect,
	grad_foo_vect,
	p_start
)

# ╔═╡ c435e202-b37c-455c-bf50-03fa6db6a4b7
Manopt.get_gradient(MEuc, foo, p0)

# ╔═╡ ace7b3ea-cd31-44b4-aaed-3408df240b78
embedding_loss_grad(M, X[1], X, neighbor_indices[1], all_negative_indices[1])

# ╔═╡ 63b120ae-a6f1-427b-9c0d-64549dbd3bb5


# ╔═╡ a0bd31b1-a841-486e-8bed-774cd9224a50
Manopt.gradient_descent(M, embedding_loss_grad., )

# ╔═╡ c0b0d723-276f-4c35-bfaf-8800be082843
sum(-embedding_dist(M, X[1], X[positive_indices]))

# ╔═╡ af9c21e7-44b0-46a7-bf12-9f40d7fc3da0


# ╔═╡ 5e49b952-7c61-4986-abfb-89adb17d5dc1
begin
	u = hcat(rand(M,1)...)
	V = rand(M, 3)
	embedding_dist(M, u, V)
end

# ╔═╡ c569a49f-faf1-4959-9afb-5fcc9d8591a6
check_gradient(M, F, grad_F; plot=true)

# ╔═╡ 2502b6fd-7a1f-4226-8449-c1bbf043ce8f
typeof(hcat(pv...))

# ╔═╡ d21c6354-13cf-4ca5-9f14-398dc1ce8970
size(hcat(pv))

# ╔═╡ 2f709bd8-f07d-4a3a-86f9-4318ff9c6a7c
grad_F(M, pv)

# ╔═╡ 7e4af6ef-91da-4a53-89b2-254c469841c0


# ╔═╡ fc941ec5-8aec-47e6-85d2-1c1fe4475fd2
gradient_descent(M, F, grad_F, M_data)

# ╔═╡ 557e020b-3937-4413-b7a7-8a150555142b


# ╔═╡ f4ec53f8-6d72-4050-8d16-893ca1263750


# ╔═╡ c03c65e3-8801-43d7-982f-3edef8affc43
gradient_descent(M, F, grad_F, M_data)

# ╔═╡ 8d4c8020-1ec0-4609-a4cd-c088c97fb06e


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
		return X[:, 2:3] ./ sqrt.(1 .+ X[:,1] .^ 2)
	end
end

# ╔═╡ 810ced18-9e65-4008-8665-9949465982c6
dists

# ╔═╡ 80d56ccc-c7cb-4322-916c-638348bc2254


# ╔═╡ 30281eb5-222b-4c0a-8986-17407d9d6f7c
hmds = fit_hmds(dists);

# ╔═╡ 2b55695f-d53c-4b78-8093-6c8fa0610073
X = get_embeddings(hmds, :poincare);
X

# ╔═╡ c68d85dd-4961-4cc9-8912-821eb1ac8b35
ds_poincare = DataFrame(x=X[:,1], y=X[:,2], name=get_sorted_vertices(labeled_graph));

# ╔═╡ ad996687-790d-46ae-9bb4-74117f34bab7
function plot_embeddings_ds(plotter :: Type{PlutoPlot}, ds, title)
	text_annotations = [((i - 1) % 20 == 0) ? ds[!, :name][i] : "" for i in 1:length(ds[!, :name])] 
	
	trace = Plotly.scatter(
	    x=ds[!, :x], 
	    y=ds[!, :y], 
	    mode="markers+text",  
	    #text=text_annotations,
	    textposition="bottom center",
	    hovertext=ds[!, :name],
	    hoverinfo="text"
	)
	
	layout = Layout(
	    title=title
	)
	
	plotter(Plot([trace], layout))
end

# ╔═╡ 4095f1d7-0c44-4104-b45b-c6f228e76c53
plot_embeddings_ds(PlutoPlot, ds_poincare, "HMDS projected to Poincare disk")

# ╔═╡ fa277357-69b9-4abe-a6fe-1346d7034a54
md"""
## Regular MDS
"""

# ╔═╡ 351af441-c417-43bb-974b-ba7a296c2b6a
begin
	mds = fit(MDS, dists * 1.0; distances=true, maxoutdim=2);
	Y_mds = predict(mds)';
	ds_mds = DataFrame(x=-Y_mds[:,1], y=Y_mds[:,2], name=get_sorted_vertices(labeled_graph));

	plot_embeddings_ds(PlutoPlot, ds_mds, "Euclidean MDS")
end

# ╔═╡ 75623451-8691-431f-97f6-f49225ecdd4b
ds_mds[!,:name] |> Set |> length

# ╔═╡ 3ae98661-18b7-481a-911d-14a41a12287e


# ╔═╡ 83626734-c2b2-478b-aabd-0ef3604786b6
md"""
## Isomap
"""

# ╔═╡ ee66593f-93ab-4f92-9316-2e9361d9fa85
begin
	isomap = fit(Isomap, dists * 1.0);
	Y_isomap = predict(mds);
	ds_isomap = DataFrame(x=Y_isomap[1,:], y=-Y_isomap[2,:], name=get_sorted_vertices(labeled_graph));

	plot_embeddings_ds(PlutoPlot, ds_isomap, "Isomap")
end

# ╔═╡ 9781e526-71b1-4e4d-a0e7-fe127104fbd9
plot_embeddings_ds(PlutoPlot, ds_isomap, "Isomap")

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

# ╔═╡ 66af16ce-cca2-438a-ae62-99fa960bd26b
Ref(p)

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
# ╠═d2b9ea9c-66e4-41cd-a961-216320c18aec
# ╠═461df8e8-cb6d-4927-8f31-48b591fbc194
# ╠═c24341de-1bbc-4d2b-bae8-c19ebbc891af
# ╠═0fb53a68-4a57-4556-bc68-6980f57fbcd9
# ╠═aad1a5eb-191a-470f-bcda-298c46a27c71
# ╠═9f4bd24c-0c0d-4837-ac49-76253f9b7741
# ╠═300ecd9f-37f9-484c-b41f-8a140d6d87d0
# ╠═0bcadf0d-3e57-436a-840d-3526f7d82fe9
# ╠═1a02921e-94b0-4dd7-8828-806910392505
# ╠═2f3a5bee-b904-468c-a4b5-0fcb83ad1a02
# ╠═79c84dfb-464a-49ab-a2a5-51972f94aebe
# ╠═00d0c1db-2212-41ee-87c3-cca1b59229ce
# ╠═9f716477-f681-42d8-adc1-050e19f6da4d
# ╠═6ab4f097-7ad8-45de-a136-12e7ad68b500
# ╠═6b181a52-ac04-458a-bbe7-cac7c903c275
# ╠═ed20f785-8f08-4c20-8dab-b1b1abc7ecc0
# ╠═66af16ce-cca2-438a-ae62-99fa960bd26b
# ╠═4bf03d54-7e31-40d8-acbd-e895f5ee332e
# ╠═110de150-6244-4c16-b539-32a541d68a84
# ╠═7b9988dd-ecb6-46be-b60f-7f8ea77e8301
# ╠═4dfdbbcc-ee5c-4b04-af64-6477ccca72b8
# ╠═7fd32c56-bcbf-4cc4-a36e-756a25689ee9
# ╠═87dfa350-b8dd-43aa-87a5-b40bc35b0bd3
# ╠═b16fa860-9add-4e25-b244-415e3570a13d
# ╠═abff6230-0feb-4898-88a2-0f06e96ccae7
# ╠═0b928607-df5c-44c5-9b8a-69c06d726de7
# ╠═f18d4056-de9a-4187-bff3-ed02df186f72
# ╠═46a9e180-832a-424c-a5c9-324e48e13f9b
# ╠═87a63fc4-2aa5-468c-b6af-8523264afa4c
# ╠═70a7e25f-f8ac-4a0b-bc57-7efa15f92a16
# ╠═4157355d-eabe-4386-9e29-7e79b15960b8
# ╠═8b845a7c-28f5-453b-8857-7ab3d5d197ce
# ╠═014f62ab-ecf8-43fc-8b17-fb1099a3cded
# ╠═472d15f5-f4e2-454a-8dd9-aff21221d0c9
# ╠═dfd85b48-57f4-4852-8173-ac2f3fac42cf
# ╠═fc6550b0-bb75-4fcd-a015-f67e71e93dfa
# ╠═4c2908da-e779-45a6-9863-e588ab4e06b0
# ╠═41683763-7d32-4213-857b-45d0734f3cd9
# ╠═b45cf641-a333-4e75-8e67-1ce58b36e26d
# ╠═dc8d6982-f29e-43f0-b839-b9f803733339
# ╠═bb08bb32-e0ec-40fa-8fe4-b399286383a1
# ╠═3615ec49-3777-4856-af60-c54b6306e2f6
# ╠═c435e202-b37c-455c-bf50-03fa6db6a4b7
# ╠═ace7b3ea-cd31-44b4-aaed-3408df240b78
# ╠═63b120ae-a6f1-427b-9c0d-64549dbd3bb5
# ╠═a0bd31b1-a841-486e-8bed-774cd9224a50
# ╠═c0b0d723-276f-4c35-bfaf-8800be082843
# ╠═af9c21e7-44b0-46a7-bf12-9f40d7fc3da0
# ╠═5e49b952-7c61-4986-abfb-89adb17d5dc1
# ╠═c569a49f-faf1-4959-9afb-5fcc9d8591a6
# ╠═2502b6fd-7a1f-4226-8449-c1bbf043ce8f
# ╠═d21c6354-13cf-4ca5-9f14-398dc1ce8970
# ╠═2f709bd8-f07d-4a3a-86f9-4318ff9c6a7c
# ╠═7e4af6ef-91da-4a53-89b2-254c469841c0
# ╠═fc941ec5-8aec-47e6-85d2-1c1fe4475fd2
# ╠═557e020b-3937-4413-b7a7-8a150555142b
# ╠═f4ec53f8-6d72-4050-8d16-893ca1263750
# ╠═c03c65e3-8801-43d7-982f-3edef8affc43
# ╠═8d4c8020-1ec0-4609-a4cd-c088c97fb06e
# ╠═39b08699-372b-47cd-9011-d2e52bc8f7b8
# ╠═fc4d4a30-7616-4a8f-9c47-e90a17315e26
# ╠═3f7642b4-e438-4e2b-80e8-44f7305a3a20
# ╠═b3f10d7e-988e-4e84-9c4e-60f2665f8834
# ╠═810ced18-9e65-4008-8665-9949465982c6
# ╠═80d56ccc-c7cb-4322-916c-638348bc2254
# ╠═30281eb5-222b-4c0a-8986-17407d9d6f7c
# ╠═2b55695f-d53c-4b78-8093-6c8fa0610073
# ╠═c68d85dd-4961-4cc9-8912-821eb1ac8b35
# ╠═ad996687-790d-46ae-9bb4-74117f34bab7
# ╠═4095f1d7-0c44-4104-b45b-c6f228e76c53
# ╠═fa277357-69b9-4abe-a6fe-1346d7034a54
# ╠═351af441-c417-43bb-974b-ba7a296c2b6a
# ╠═75623451-8691-431f-97f6-f49225ecdd4b
# ╠═3ae98661-18b7-481a-911d-14a41a12287e
# ╠═83626734-c2b2-478b-aabd-0ef3604786b6
# ╠═ee66593f-93ab-4f92-9316-2e9361d9fa85
# ╠═9781e526-71b1-4e4d-a0e7-fe127104fbd9
# ╠═5875994b-39da-4861-9e2c-5aaf12033a21
# ╠═3cb49c9e-793a-4bc7-a6f6-6fddc72f0ae4
# ╠═bea8e470-8266-42c5-b399-38735fbff300
# ╠═30631dbc-c39f-4620-babe-24cdad1bebbe
# ╠═ef77dc32-1718-419e-8be1-362b1cd55e1b
# ╠═73d77fe2-d921-46c4-b760-60960b77f4d9
# ╠═3210cccf-4295-4382-b9c6-95e6d984c1dd
# ╠═af50c3e3-a598-4241-805f-7f2a47e9ec38
# ╠═f27ffd49-8f1b-4993-81c3-717913e762db
# ╠═afa6047e-1049-4aa6-8332-af642da7f400
# ╠═5078689e-81fc-4710-be33-9f17af00dd48
# ╠═bd65f2c5-e0d2-4c18-8226-25a3a06ae099
# ╠═0e89d3c5-463c-4329-b11c-8586cc834fd5
# ╠═616f27ef-f0f7-4628-bca7-eeb087707044
