### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ 4dc336fd-7676-4f1c-9772-a74cacea1571
begin
	using Pkg
	Pkg.activate("../Neucgraph")
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
	using Statistics

	using MarketTechnicals
	using Printf
	using StatisticalGraphics
	using Plotly
	using PlutoPlotly
	using ManifoldLearning

	using Manifolds
	using Manopt
	using Random
	using Plots
	using Zygote
	ENV["LD_LIBRARY_PATH"] = ""
end

# ╔═╡ 461df8e8-cb6d-4927-8f31-48b591fbc194
begin
	using JSON
	path = "../data/org_roam_records_2023_10_16.json";
	records = JSON.parse(read(path, String));
end

# ╔═╡ deafca23-adc3-4e1a-84d6-7cc96e25f838
using ForwardDiff

# ╔═╡ 4e33f833-4457-4ab5-b4d8-f4dd3902d740
md"""
Load graph records in JSON
"""

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

# ╔═╡ 78e48617-71a5-4a8f-a381-7c0ce68e2752


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
# ╠═╡ disabled = true
#=╠═╡
graph_dists = [
	[0, 2, 1, 2, 2],
	[2, 0, 2, 1, 2],
	[1, 2, 0, 2, 2],
	[2, 1, 2, 0, 1],
	[2, 2, 2, 1, 0]
]
  ╠═╡ =#

# ╔═╡ b16fa860-9add-4e25-b244-415e3570a13d
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ abff6230-0feb-4898-88a2-0f06e96ccae7
#=╠═╡
begin
	positive_indices = neighbor_indices[1]
	negative_indices = sample(all_negative_indices[1], 2, replace=false)
end
  ╠═╡ =#

# ╔═╡ 0b928607-df5c-44c5-9b8a-69c06d726de7
X = rand(M, 5) * 0.1

# ╔═╡ 46a9e180-832a-424c-a5c9-324e48e13f9b
anchor_indices = [1]

# ╔═╡ 87a63fc4-2aa5-468c-b6af-8523264afa4c
# ╠═╡ disabled = true
#=╠═╡
x :: Integer = 1
  ╠═╡ =#

# ╔═╡ 322680da-dcbc-41a8-83d8-f9fe50978072
struct GraphNeighborhoodIndices
	neighbor_indices:: Vector{Vector{Int64}}
	nonneighbor_indices:: Vector{Vector{Int64}} 
end

# ╔═╡ e112ce7a-de69-410d-aacc-40e5941770a8
begin
	struct NodeBatch
		point_indices:: Vector{Int64}
		positive_indices:: Vector{Vector{Int64}}
		negative_indices:: Vector{Vector{Int64}}
	end
	
		
	function sample_node_batch(graph_indices:: GraphNeighborhoodIndices, batch_size=2)
		n = size(graph_indices.neighbor_indices)[1]
		point_indices = sample(1:n, batch_size)
		NodeBatch(
			point_indices,
			graph_indices.neighbor_indices[point_indices],
			graph_indices.nonneighbor_indices[point_indices],
		)
	end
end

# ╔═╡ 69761f12-8442-4be3-adc0-5a891ef7ded7
function sample_node_batch(labeled_graph:: LabeledGraph, batch_size=2, n_negative=10)
	n_nodes = length(labeled_graph.node_to_index)
	point_indices = sample(1:n_nodes, batch_size)
	positive_indices = [
		labeled_graph.graph.fadjlist[pi]
		for pi in point_indices
	]
	negative_indices = [
		[
			neg_index
			for neg_index in sample(1:n_nodes, n_negative)
			if !(neg_index in positive_indices[i]) && neg_index != pi
		]
		for (i, pi) in enumerate(point_indices)
	]
	NodeBatch(
		point_indices,
		positive_indices,
		negative_indices
	)
end

# ╔═╡ 3a49dcd9-201a-4f79-a0f3-0f0d410da7e0
function embedding_dist(M, point, other)
	distance.(Ref(M), Ref(point), other)
end

# ╔═╡ 83dfc090-17d6-4848-bad1-44c967f2b0cc
md"""
## Set up the graph
"""

# ╔═╡ 0636f25d-9b70-4e6a-a5c6-f24b5877b696
#=╠═╡
begin
	example_graph = GraphNeighborhoodIndices(neighbor_indices, all_negative_indices)
	star_graph_indices = GraphNeighborhoodIndices(
		# 1s node is connected to everything
		[
			[2,3,4],
			[1],
			[1],
			[1],
			[4]
		],
		[
			[],
			[3,4],
			[2,4],
			[2,3,5]
		]
	)
	node_batch_sampled = sample_node_batch(example_graph)
	node_batch = NodeBatch([1,2], [neighbor_indices[i] for i in [1,2]], [all_negative_indices[i] for i in [1,2]])

	star_graph_node_batch = NodeBatch([1,2,3,4, 5], star_graph_indices.neighbor_indices, star_graph_indices.nonneighbor_indices)
end
  ╠═╡ =#

# ╔═╡ d938520d-d832-48cb-bf67-766f25610dcd


# ╔═╡ 7866d9a6-8432-44f9-a6cd-6c73a435c678


# ╔═╡ 240542ec-bcc2-420a-87f2-82c164c55924
md"""

### Loss function

Loss per neighborhood of node $i$: pos are positive indices, $\mathcal{N}$ is the whole sampled neighborhood (positive + negative)

$P(X, i, pos, \mathcal{N}) = \frac{
	\sum_{p \in pos}e^{-d(X_i, X_{p})}
	}{
	\sum_{n \in \mathcal{N}}e^{-d(X_i, X_{n})}
	}$

$$L(X, i, pos, \mathcal{N}) = -log\ P(X, i, pos, \mathcal{N}) = $$

$$= - log(\sum_{p \in pos}e^{-d(X_i, X_{p})}) + log(\sum_{n \in \mathcal{N}}e^{-d(X_i, X_{n})})$$
"""

# ╔═╡ 70a7e25f-f8ac-4a0b-bc57-7efa15f92a16
function embedding_loss(M, X, point_idx:: Integer, point_positive_indices, point_negative_indices)
	all_indices = [point_positive_indices; point_negative_indices]
	negative_loss = logsumexp(-embedding_dist(M, X[point_idx], X[all_indices]) .+ 1e-6)
	positive_loss = -logsumexp(-embedding_dist(M, X[point_idx], X[point_positive_indices]) .+ 1e-6)
	positive_loss + negative_loss
end

# ╔═╡ eca17896-76ea-4017-b8a3-bfe1847b7047
begin
	function embedding_loss(M, X, node_batch:: NodeBatch)
		sum(embedding_losses(M, X, node_batch))
	end
	
	function embedding_losses(M, X, node_batch:: NodeBatch)
		[
			embedding_loss(M, X, p_idx, pos_idxs, neg_idxs)
			for (p_idx, pos_idxs, neg_idxs) in zip(node_batch.point_indices, node_batch.positive_indices, node_batch.negative_indices)
		]
	end
end

# ╔═╡ f2a914e2-226a-453a-a409-f19b84e38758
#=╠═╡
begin
	println("single loss: ", round(embedding_loss(M, X, 1, neighbor_indices[1], all_negative_indices[1]), digits=2))

	println("batch loss for sample batch: ", round(embedding_loss(M, X, node_batch), digits=2))

	println("batch loss for star graph batch: ", round(embedding_loss(M, X, star_graph_node_batch), digits=2))
	
end
  ╠═╡ =#

# ╔═╡ 984dfc8c-62d3-436f-a501-ab927cf7cec8
function fill_nothings(X, l)
	if X === nothing
		zeros(l)
	else
		replace(X, NaN=>0)
	end
end

# ╔═╡ 43fce4a1-c817-4d7b-a08c-eee3f22e1f3b
function embedding_loss_grad_euc(M, X, node_batch:: NodeBatch)
	loss(x) = embedding_loss(M, x, node_batch)
	grad_result = Zygote.gradient(loss, X)
	grad_result[1]
end

# ╔═╡ 89a7f2dd-3ed0-40cd-8d4a-609b707ce348
#=╠═╡
embedding_loss_grad_euc(M, X, node_batch)
  ╠═╡ =#

# ╔═╡ 472d15f5-f4e2-454a-8dd9-aff21221d0c9
function embedding_loss_grad(M, X, node_batch:: NodeBatch)
	l = size(X[1][1])
	egrad = fill_nothings.(embedding_loss_grad_euc(M, X, node_batch), Ref(l))
	project.(Ref(M), X, egrad)
end


# ╔═╡ 5977755b-9c29-404d-8723-1c6f1b634068
#=╠═╡
embedding_loss_grad(M, X, node_batch)
  ╠═╡ =#

# ╔═╡ f4d362b9-9552-44ec-b95b-6097e929681d
#=╠═╡
begin
	local alpha = 1e-3
	sample_riemmanian_grad = embedding_loss_grad_euc(M, X, node_batch)
	retract.(Ref(M), X, - alpha * sample_riemmanian_grad)
end
  ╠═╡ =#

# ╔═╡ 56133133-ae05-494a-8394-58250bfb92c3
md"""

# Riemannian gradient descent

"""

# ╔═╡ 64998d98-4f51-4bbd-9e12-4a5e99e677ea
sample(1:5,2)

# ╔═╡ 01030291-8ffa-48da-a066-2abd243eadbc
function embedding_update_step(M, X, node_batch:: NodeBatch, alpha=1e-3)
	# get riemannian gradient
	rgrad = embedding_loss_grad(M, X, node_batch)
	#=
	update step: X' = retract_X(- alpha grad f(X))
	corresponds to Euclidean X' = X - alpha grad f(X) 
	=#
	
	retract.(Ref(M), X, - alpha * rgrad)
end

# ╔═╡ 5c1e9aad-db93-4d9e-9e1e-382b0d5058d4
begin
	struct OptimizationStepState
		inputs
		egrad
		egrad_filled
		rgrad
		X_next
		losses
	end
	
	function embedding_update_step(M, X, node_batch, alpha, verbose=true)

		losses = embedding_losses(M, X, node_batch)
		egrad = embedding_loss_grad_euc(M, X, node_batch)
		l = size(X[1])[1]
		egrad_filled = fill_nothings.(egrad, Ref(l))
		rgrad = project.(Ref(M), X, egrad_filled)
		X_next = retract.(Ref(M), X, - alpha * rgrad)
		OptimizationStepState(
			node_batch,
			egrad,
			egrad_filled, 
			rgrad,
			X_next,
			losses
		)
	end
end

# ╔═╡ 1f9a422e-026d-4686-b862-7e28a423b141
#=╠═╡
begin
	α = 1e-3
	iters = 1000
	global X_updated = X
	for _ in 1:iters
		global X_updated = embedding_update_step(M, X_updated, star_graph_node_batch, α)
	end

	["original loss"=> embedding_loss(M, X, star_graph_node_batch), "loss after step" => embedding_loss(M, X_updated, star_graph_node_batch)]
end
  ╠═╡ =#

# ╔═╡ ed0c20ed-d136-4e84-b734-9236cdd82254
function hyperboloid_to_poincare(X)
	l = size(X)[1]
	X[1:(l-1)] / (1 + X[l])
end

# ╔═╡ 70946a03-7a1d-46b0-83f9-ac37653a088f
#=╠═╡
begin
	X_poincare = reduce(hcat, hyperboloid_to_poincare.(X))'
	X_updated_poincare = reduce(hcat, hyperboloid_to_poincare.(X_updated))'
end
  ╠═╡ =#

# ╔═╡ 6a5f2210-db86-4d39-bfa7-642c8465de0b
md"""

The displayed embeddings after (red) and before (blue) optimization show that embeddings start to look more like they come from a star graph
"""

# ╔═╡ 4a1ec5f0-85b0-4e96-ab38-068f21e19df8
#=╠═╡
begin
	Plots.scatter(X_poincare[:,1], X_poincare[:,2], markercolor=:blue)
	Plots.scatter!(X_updated_poincare[:,1], X_updated_poincare[:,2], markercolor=:red)
end
  ╠═╡ =#

# ╔═╡ 1332265a-23b2-496b-bf99-d04b9c57186b
function pairwise_manifold_dists(M, X)
	n = size(X)[1]
	dists = [
		[
			distance(M, X[i], X[j])
			for i in 1:n
		]
		for j in 1:n
	]
	reduce(hcat, dists)'
end

# ╔═╡ a3d52860-1528-469d-8427-6d1e4bac0777
#=╠═╡
begin
	manifold_dists_before_optimization = pairwise_manifold_dists(M, X)
	manifold_dists_after_optimization = pairwise_manifold_dists(M, X_updated)
	star_graph_dists = [
		[0, 1, 1, 1, 2],
		[1, 0, 2, 2, 2],
		[1, 2, 0, 2, 2],
		[1, 2, 2, 0, 1],
		[2, 2, 2, 1, 0]
	]
	graph_dists_mat = reduce(hcat, star_graph_dists)'
end
  ╠═╡ =#

# ╔═╡ e90e444c-edb6-4866-8d60-f99b58a81443
function matrix_kendall(X, Y)
	n = size(X)[1]
	corr = 0
	for i in 1:n
		corr += StatsBase.corkendall(sortperm(X[1,:]), sortperm(Y[1,:]))
	end
	corr / n
end

# ╔═╡ 06036fe1-aad3-4611-8597-63f9ed983336
md"""
### Comparing the distances

We also see that the Kendall's tau rank correlation between distances is better for embeddings after optimization

"""

# ╔═╡ e191afca-a696-422b-a2c2-3fbc2b93a46c
#=╠═╡
matrix_kendall(graph_dists_mat, manifold_dists_after_optimization)
  ╠═╡ =#

# ╔═╡ 3847fcf9-3a09-4b07-a06a-20f550b85013
#=╠═╡
matrix_kendall(graph_dists_mat, manifold_dists_before_optimization)
  ╠═╡ =#

# ╔═╡ bd9a5cfc-0662-453e-aa95-5008384d4a78
md"""

## Running on LabeledGraph

"""

# ╔═╡ dc8d6982-f29e-43f0-b839-b9f803733339
begin
	mutable struct GraphEmbeddingTrainer
		samplable_graph # something that can be used to sample NodeBatch
		embeddings
		M
		batch_size
		n_negative
		alpha
		log
	end
	
	function init_trainer(labeled_graph:: LabeledGraph; M=Hyperbolic(2), batch_size=5, n_negative=20, alpha=1e-4, init_embeddings_scale=0.1)
		embeddings = rand(M, length(labeled_graph.node_to_index)) * init_embeddings_scale
	
		GraphEmbeddingTrainer(
			labeled_graph,
			embeddings,
			M,
			batch_size,
			n_negative,
			alpha,
			[]
		)
	end

	function train(trainer:: GraphEmbeddingTrainer, n_steps; verbose=false)
		step_state = Nothing
		for _ in 1:n_steps
			step_state = train_step(trainer)
			if verbose
				debug_print_step_state(step_state)
			end
			if !is_step_valid(step_state)
				println("state invalid")
				return step_state
			end
			trainer.embeddings = step_state.X_next
			append!(trainer.log, Statistics.mean(step_state.losses))
		end

		step_state
	end

	function is_valid_number(value)
		isa(value, Number) && !isnan(value)
	end
	
	function is_valid_numeric_vector(row)
		if !isa(row, Vector)
				return false
		end
		for value in row
			if !is_valid_number(value)
				return false
			end
		end
		return true
	end
	
	function is_valid_numeric(a)
		for row in a
			if !is_valid_numeric_vector(row)
				return false
			end
		end
		return true
	end

	function mean_nonzero_norm(vs)
		norms = norm.(vs)
		nonzero_norms = norms .> 1e-6
		Statistics.mean(norms[nonzero_norms])
	end

	function debug_print_step_state(step_state)
		println("embeddings valid: ", is_valid_numeric(step_state.X_next))
		println("egrad valid: ", is_valid_numeric(step_state.egrad_filled))
		println("rgrad valid: ", is_valid_numeric(step_state.rgrad))
		println("embeddings mean norm: ", mean_nonzero_norm(step_state.X_next))
		println("nonzero egrad mean norm: ", mean_nonzero_norm(step_state.egrad_filled))
		println("nonzero rgrad mean norm: ", mean_nonzero_norm(step_state.rgrad))
		println("losses: ", step_state.losses)
		println("loss: ", Statistics.mean(step_state.losses))
	end

	function is_step_valid(step_state)
		grads_valid = all([
			is_valid_numeric(field)
			for field in [step_state.egrad_filled, step_state.rgrad]
		])
		is_valid_numeric_vector(step_state.losses) && grads_valid && is_valid_numeric(step_state.X_next)
	end
	
	function train_verbose(trainer:: GraphEmbeddingTrainer, n_steps)
		step_state = Nothing
		println("TRAINER embeddings valid: ", is_valid_numeric(trainer.embeddings))
		for _ in 1:n_steps
			step_state = train_step(trainer)
			debug_print_step_state(step_state)
			if !is_step_valid(step_state)
				println("state invalid")
				return step_state
			end
			println("TRAINER embeddings valid: ", is_valid_numeric(trainer.embeddings))
			trainer.embeddings = step_state.X_next
			append!(trainer.log, Statistics.mean(step_state.losses))
		end

		step_state
	end

	function train_step(trainer)
		node_batch = sample_node_batch(trainer.samplable_graph, trainer.batch_size, trainer.n_negative)
		opt_state = embedding_update_step(trainer.M, trainer.embeddings, node_batch, trainer.alpha, true)
		opt_state
	end
end

# ╔═╡ 79cef6a7-bc77-4c36-a2cf-dfc09ee89123
begin
	d = Dict([])
	
	d["a"] = 5
end

# ╔═╡ d989054b-ced6-44c5-a7f4-2a1b1c613f5b
begin
	vs = [
	
		[1, 2],
		[0, 0]
	]
	mean_nonzero_norm(vs)
	true && false
end

# ╔═╡ bb08bb32-e0ec-40fa-8fe4-b399286383a1
trainer = init_trainer(labeled_graph, batch_size=5, alpha=1e-4, init_embeddings_scale=0.5)

# ╔═╡ dac6400e-e9b5-437a-8ffa-8568f935cc30
labeled_graph_dists = floyd_warshall_shortest_paths(labeled_graph.graph).dists;

# ╔═╡ a6efecc3-ca20-4039-8df0-0c41aa9be5ab
matrix_kendall(pairwise_manifold_dists(trainer.M, trainer.embeddings), labeled_graph_dists)

# ╔═╡ ace438dd-9cc7-4042-8f13-928fe1516343
opt_step_state = train(trainer, 100000)

# ╔═╡ d32b70e6-a61c-41e5-a368-973f6b291002
Plots.plot(MarketTechnicals.sma(trainer.log, 250))

# ╔═╡ 87e0a87a-b1a2-4dc4-b974-6d24ef2a5912
length(trainer.log)

# ╔═╡ db0da446-b913-41a8-b797-0d05e3c29e33
begin
	embeddings_dists = pairwise_manifold_dists(trainer.M, trainer.embeddings)
	matrix_kendall(embeddings_dists, labeled_graph_dists)
end

# ╔═╡ 72c39cc7-7ebc-427b-a37d-e09f8fda641b
begin
	embeddings_poincare = hyperboloid_to_poincare.(trainer.embeddings)
	embeddings_poincare_matrix = reduce(hcat, embeddings_poincare)'
	NaN
end

# ╔═╡ 3faaf7a1-8a11-4817-96b7-b423ab80e969
Plots.scatter(embeddings_poincare_matrix[:,1], embeddings_poincare_matrix[:,2])

# ╔═╡ 09de7780-41d6-400f-bcdf-e0c7329f9bd9
i = 2

# ╔═╡ ed62adac-8099-46f8-9ca4-2872e8632028
findall(x -> x < quantile(embeddings_dists[i,:], 0.05), embeddings_dists[i,:])

# ╔═╡ 985f0999-77dd-4ebf-afd7-3d008b72044e
findall(x -> x < 2, labeled_graph_dists[i,:])

# ╔═╡ ba75aaa4-cecf-4ccc-a9d2-f460bb2ab809
# ╠═╡ disabled = true
#=╠═╡
labeled_graph_dists = floyd_warshall_shortest_paths(labeled_graph.graph).dists;
  ╠═╡ =#

# ╔═╡ 2cbf0057-4064-4b21-9bc6-a72ae73b4381
matrix_kendall(embedding_dists, labeled_graph_dists)

# ╔═╡ 192e2d46-df17-4100-b7dd-510f1343da55
# ╠═╡ disabled = true
#=╠═╡

  ╠═╡ =#

# ╔═╡ 3f9131a4-204e-419c-a91a-b7b12dcb0805
#=╠═╡
begin
	invalid_row_idx= [
		ri
		for (ri, row) in enumerate(opt_step_state.egrad_filled)
		if any(isnan.(row))
	][1]
	
	invalid_row_batch_idx = [i for (i, ri) in enumerate(opt_step_state.inputs.point_indices) if ri == invalid_row_idx][1]

	invalid_positive_indices = opt_step_state.inputs.positive_indices[invalid_row_batch_idx]
	invalid_negative_indices = opt_step_state.inputs.negative_indices[invalid_row_batch_idx]
	invalid_batch = NodeBatch([invalid_row_idx], [invalid_positive_indices], [invalid_negative_indices])
end
  ╠═╡ =#

# ╔═╡ 1fa62b8f-ae94-4239-9418-3ba7b0b397be
#=╠═╡
invalid_batch_grads = embedding_loss_grad_euc(trainer.M, trainer.embeddings, invalid_batch)
  ╠═╡ =#

# ╔═╡ 6996ebbd-cf78-4ca2-a7b5-8d57d57effd6


# ╔═╡ 52bdd3c3-eb3b-4d3d-b62e-841325fd7d74
#=╠═╡
invalid_batch_grads[invalid_row_idx]
  ╠═╡ =#

# ╔═╡ c9d21b80-386d-4353-9c5d-d6b2c6ee80dc
#=╠═╡
invalid_batch_all_indices = [[invalid_row_idx]; invalid_positive_indices; invalid_negative_indices ]
  ╠═╡ =#

# ╔═╡ 09c437ac-8692-4ba4-94b8-7b3b80282186
#=╠═╡
pairwise_manifold_dists(trainer.M, trainer.embeddings[invalid_batch_all_indices])
  ╠═╡ =#

# ╔═╡ acdabe68-e436-46cd-94e7-c2b7745dfcad
#=╠═╡
f_pos(embs) = logsumexp(-embedding_dist(trainer.M, embs[invalid_row_idx], embs[invalid_positive_indices]))
  ╠═╡ =#

# ╔═╡ 9aeaae7d-ba26-454d-aa52-5bb5c496e1f4
#=╠═╡
f_neg(embs) = logsumexp(-embedding_dist(trainer.M, embs[invalid_row_idx], embs[invalid_negative_indices]))
  ╠═╡ =#

# ╔═╡ 3bc638ce-d30d-48bb-84c3-8d0416afe22c
#=╠═╡
is_valid_numeric(fill_nothings.(ForwardDiff.gradient(f_neg, trainer.embeddings)[1], 3))
  ╠═╡ =#

# ╔═╡ c9ae6ed2-22c0-4903-8438-9e1b5bd110d2
#=╠═╡
is_valid_numeric(fill_nothings.(Zygote.gradient(f_pos, trainer.embeddings)[1], 3))
  ╠═╡ =#

# ╔═╡ 0d9b90e0-027a-4b00-9806-0f7545c78bc5
#=╠═╡
begin
	invalid_idx = 482
	invalid_pos_indices = opt_step_state.inputs.positive_indices[5]
	invalid_neg_indices = opt_step_state.inputs.negative_indices[5]
end
  ╠═╡ =#

# ╔═╡ d59a4d24-e7fe-41a4-b5d4-d7025c39a1ed
#=╠═╡
for (i, ps, ns) in zip(batch.point_indices, batch.positive_indices, batch.negative_indices)
	println(i)
	nb = NodeBatch([i], [ps], [ns])
	#embedding_loss_grad(trainer.M, trainer.embeddings, )
	global egrad = embedding_loss_grad_euc(trainer.M, trainer.embeddings, node_batch)
	project.(Ref(trainer.M), trainer.embeddings, egrad)
	egrad
end
  ╠═╡ =#

# ╔═╡ bdba719c-e7bc-4bd1-9346-fe9a5a25ba9a
begin
	g(x, i) = x[i]^2
	
	Zygote.gradient(x -> g(x,1), [1, 0])
end

# ╔═╡ 0fe61a07-1d29-4dd5-afe1-e83d891cd813
[length(ps) for ps in batch.positive_indices]

# ╔═╡ abce3855-246b-49e5-9268-3c6bd03dd9d8
[length(ns) for ns in batch.negative_indices]

# ╔═╡ 7c413a2b-d030-4afa-9f47-baf6baf4a563
# ╠═╡ disabled = true
#=╠═╡
begin

	alpha_trainer = 1e-3
	trainer_iters = 1000
	original_embs = copy(trainer.embeddings)
	embs = trainer.embeddings
	
	println("single loss: ", round(embedding_loss(trainer.M, embs, batch.point_indices[1], batch.positive_indices[1], batch.negative_indices[1]), digits=2))
	
	for _ in 1:iters
		embs = embedding_update_step(trainer.M, embs, batch, alpha_trainer)
	end

	["original loss"=> embedding_loss(trainer.M, original_embs, batch), "loss after step" => embedding_loss(trainer.M, trainer.embeddings, batch)]
	
end
  ╠═╡ =#

# ╔═╡ 486cf55b-1ed9-4029-88ec-d96c1db645e5


# ╔═╡ 573c63a9-9fc0-4272-8db1-2df82b451bb2
batch

# ╔═╡ 96acfdfe-68bd-412e-b386-8dab1b31a4a1


# ╔═╡ f32a220c-dd16-4e49-8670-369d93a1d758
#=╠═╡
train(trainer)
  ╠═╡ =#

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
#=╠═╡
embedding_loss_grad(M, X[1], X, neighbor_indices[1], all_negative_indices[1])
  ╠═╡ =#

# ╔═╡ 63b120ae-a6f1-427b-9c0d-64549dbd3bb5


# ╔═╡ a0bd31b1-a841-486e-8bed-774cd9224a50
Manopt.gradient_descent(M, embedding_loss_grad., )

# ╔═╡ c0b0d723-276f-4c35-bfaf-8800be082843
#=╠═╡
sum(-embedding_dist(M, X[1], X[positive_indices]))
  ╠═╡ =#

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

# ╔═╡ ad4533a9-4682-4387-8875-6de3bfd10bc0


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
# ╠═4dc336fd-7676-4f1c-9772-a74cacea1571
# ╠═4e33f833-4457-4ab5-b4d8-f4dd3902d740
# ╠═461df8e8-cb6d-4927-8f31-48b591fbc194
# ╠═c24341de-1bbc-4d2b-bae8-c19ebbc891af
# ╠═0fb53a68-4a57-4556-bc68-6980f57fbcd9
# ╠═aad1a5eb-191a-470f-bcda-298c46a27c71
# ╠═9f4bd24c-0c0d-4837-ac49-76253f9b7741
# ╠═300ecd9f-37f9-484c-b41f-8a140d6d87d0
# ╠═0bcadf0d-3e57-436a-840d-3526f7d82fe9
# ╠═1a02921e-94b0-4dd7-8828-806910392505
# ╠═2f3a5bee-b904-468c-a4b5-0fcb83ad1a02
# ╠═78e48617-71a5-4a8f-a381-7c0ce68e2752
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
# ╠═46a9e180-832a-424c-a5c9-324e48e13f9b
# ╠═87a63fc4-2aa5-468c-b6af-8523264afa4c
# ╠═322680da-dcbc-41a8-83d8-f9fe50978072
# ╠═e112ce7a-de69-410d-aacc-40e5941770a8
# ╠═69761f12-8442-4be3-adc0-5a891ef7ded7
# ╠═3a49dcd9-201a-4f79-a0f3-0f0d410da7e0
# ╠═83dfc090-17d6-4848-bad1-44c967f2b0cc
# ╠═0636f25d-9b70-4e6a-a5c6-f24b5877b696
# ╠═d938520d-d832-48cb-bf67-766f25610dcd
# ╠═7866d9a6-8432-44f9-a6cd-6c73a435c678
# ╠═240542ec-bcc2-420a-87f2-82c164c55924
# ╠═70a7e25f-f8ac-4a0b-bc57-7efa15f92a16
# ╠═eca17896-76ea-4017-b8a3-bfe1847b7047
# ╠═f2a914e2-226a-453a-a409-f19b84e38758
# ╠═984dfc8c-62d3-436f-a501-ab927cf7cec8
# ╠═43fce4a1-c817-4d7b-a08c-eee3f22e1f3b
# ╠═89a7f2dd-3ed0-40cd-8d4a-609b707ce348
# ╠═472d15f5-f4e2-454a-8dd9-aff21221d0c9
# ╠═5977755b-9c29-404d-8723-1c6f1b634068
# ╠═f4d362b9-9552-44ec-b95b-6097e929681d
# ╠═56133133-ae05-494a-8394-58250bfb92c3
# ╠═64998d98-4f51-4bbd-9e12-4a5e99e677ea
# ╠═01030291-8ffa-48da-a066-2abd243eadbc
# ╠═5c1e9aad-db93-4d9e-9e1e-382b0d5058d4
# ╠═1f9a422e-026d-4686-b862-7e28a423b141
# ╠═ed0c20ed-d136-4e84-b734-9236cdd82254
# ╠═70946a03-7a1d-46b0-83f9-ac37653a088f
# ╠═6a5f2210-db86-4d39-bfa7-642c8465de0b
# ╠═4a1ec5f0-85b0-4e96-ab38-068f21e19df8
# ╠═1332265a-23b2-496b-bf99-d04b9c57186b
# ╠═a3d52860-1528-469d-8427-6d1e4bac0777
# ╠═e90e444c-edb6-4866-8d60-f99b58a81443
# ╠═06036fe1-aad3-4611-8597-63f9ed983336
# ╠═e191afca-a696-422b-a2c2-3fbc2b93a46c
# ╠═3847fcf9-3a09-4b07-a06a-20f550b85013
# ╠═bd9a5cfc-0662-453e-aa95-5008384d4a78
# ╠═dc8d6982-f29e-43f0-b839-b9f803733339
# ╠═79cef6a7-bc77-4c36-a2cf-dfc09ee89123
# ╠═d989054b-ced6-44c5-a7f4-2a1b1c613f5b
# ╠═bb08bb32-e0ec-40fa-8fe4-b399286383a1
# ╠═dac6400e-e9b5-437a-8ffa-8568f935cc30
# ╠═a6efecc3-ca20-4039-8df0-0c41aa9be5ab
# ╠═ace438dd-9cc7-4042-8f13-928fe1516343
# ╠═d32b70e6-a61c-41e5-a368-973f6b291002
# ╠═87e0a87a-b1a2-4dc4-b974-6d24ef2a5912
# ╠═db0da446-b913-41a8-b797-0d05e3c29e33
# ╠═72c39cc7-7ebc-427b-a37d-e09f8fda641b
# ╠═3faaf7a1-8a11-4817-96b7-b423ab80e969
# ╠═09de7780-41d6-400f-bcdf-e0c7329f9bd9
# ╠═ed62adac-8099-46f8-9ca4-2872e8632028
# ╠═985f0999-77dd-4ebf-afd7-3d008b72044e
# ╠═ba75aaa4-cecf-4ccc-a9d2-f460bb2ab809
# ╠═2cbf0057-4064-4b21-9bc6-a72ae73b4381
# ╠═192e2d46-df17-4100-b7dd-510f1343da55
# ╠═3f9131a4-204e-419c-a91a-b7b12dcb0805
# ╠═1fa62b8f-ae94-4239-9418-3ba7b0b397be
# ╠═6996ebbd-cf78-4ca2-a7b5-8d57d57effd6
# ╠═52bdd3c3-eb3b-4d3d-b62e-841325fd7d74
# ╠═c9d21b80-386d-4353-9c5d-d6b2c6ee80dc
# ╠═09c437ac-8692-4ba4-94b8-7b3b80282186
# ╠═acdabe68-e436-46cd-94e7-c2b7745dfcad
# ╠═9aeaae7d-ba26-454d-aa52-5bb5c496e1f4
# ╠═deafca23-adc3-4e1a-84d6-7cc96e25f838
# ╠═3bc638ce-d30d-48bb-84c3-8d0416afe22c
# ╠═c9ae6ed2-22c0-4903-8438-9e1b5bd110d2
# ╠═0d9b90e0-027a-4b00-9806-0f7545c78bc5
# ╠═d59a4d24-e7fe-41a4-b5d4-d7025c39a1ed
# ╠═bdba719c-e7bc-4bd1-9346-fe9a5a25ba9a
# ╠═0fe61a07-1d29-4dd5-afe1-e83d891cd813
# ╠═abce3855-246b-49e5-9268-3c6bd03dd9d8
# ╠═7c413a2b-d030-4afa-9f47-baf6baf4a563
# ╠═486cf55b-1ed9-4029-88ec-d96c1db645e5
# ╠═573c63a9-9fc0-4272-8db1-2df82b451bb2
# ╠═96acfdfe-68bd-412e-b386-8dab1b31a4a1
# ╠═f32a220c-dd16-4e49-8670-369d93a1d758
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
# ╠═ad4533a9-4682-4387-8875-6de3bfd10bc0
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
