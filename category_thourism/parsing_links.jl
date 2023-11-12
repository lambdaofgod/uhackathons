using Glob
using Distributed
using Graphs, MetaGraphs
using ProgressBars
using DataFrames
using StatsBase
using CSV
using GraphIO.GraphML


example_nlab_contents = read("example_nlab_page.md", String)


function extract_nlab_links(page_contents)
    matches = eachmatch(r"\[\[(.*?)\]\]", page_contents)
    links = [m[1] for m in matches if !startswith(m[1], "!")]
    links |> Set |> collect
end


links = extract_nlab_links(example_nlab_contents)


struct NLabContent
    title::String
    contents::String
end


function load_content(path)
    contents = read(path * "/content.md", String)
    name = read(path * "/name", String)
    NLabContent(name, contents)
end


function load_contents(glob_dir="nlab-content/pages/")
    nlab_files_tree = walkdir(glob_dir) |> collect
    [
        load_content(path)
        for (path, _, files) in nlab_files_tree
        if ("content.md" in files) & ("name" in files)
    ]
end

function get_valid_links(article_title, links, titles)
    [
        l
        for l in links
        if (lowercase(l) in titles) & (l != article_title)
    ]
end

struct NLabArticle
    title::Symbol
    title_name::String
    contents::String
    links::Vector{String}
end


function load_article(content::NLabContent, titles)
    links = get_valid_links(content.title, extract_nlab_links(content.contents), titles)
    NLabArticle(Symbol(content.title), content.title, content.contents, links)
end

function load_articles(contents::Vector{NLabContent})
    titles = [lowercase(a.title) for a in contents]
    [load_article(c, titles) for c in tqdm(contents)]
end


nlab_articles = load_articles(load_contents());

# Graph with neighbor names

function make_nlab_metagraph(nlab_articles)

    raw_graph = MetaGraph(length(nlab_articles))
    for (i, article) in tqdm(enumerate(nlab_articles))
        set_prop!(raw_graph, i, :title, Symbol(article.title))
    end
    set_indexing_prop!(raw_graph, :title)


    for article in tqdm(nlab_articles)
        for raw_link in article.links
            link = raw_link |> lowercase |> Symbol
            if haskey(raw_graph.metaindex[:title], link)
                in_idx = raw_graph.metaindex[:title][article.title]
                out_idx = raw_graph.metaindex[:title][link]
                add_edge!(raw_graph, in_idx, out_idx)
                set_prop!(raw_graph, Graphs.Edge(in_idx, out_idx), :type, "link")
            end
        end
    end

    (graph, _) = induced_subgraph(raw_graph, findall(>(0), degree(raw_graph)))
    set_indexing_prop!(graph, :title)
    graph
end

graph = make_nlab_metagraph(nlab_articles);


function get_neighbor_names(graph, vertex_name, propname=:title)
    neighbor_idxs = neighbors(graph, graph.metaindex[propname][vertex_name])
    [graph.vprops[n][propname] for n in neighbor_idxs]
end


"""
dataframe with node centrality metrics
"""
function make_graph_info_df(graph)
    v_size = graph.vprops |> length
    titles = get_prop.(Ref(graph), 1:v_size, Ref(:title))
    df = DataFrame(
        title=titles,
        degree=degree(graph),
        degree_centrality=degree_centrality(graph),
        pagerank=pagerank(graph)
    )
    df[!, :agg_centrality] = harmmean.(zip(df.degree_centrality, df.pagerank))
    df
end


graph_info_df = make_graph_info_df(graph)

graph_info_df = sort(graph_info_df, :agg_centrality, rev=true)

sort(graph_info_df, :pagerank, rev=true) |> (df -> first(df, 20))

sort(graph_info_df, :degree_centrality, rev=true) |> (df -> first(df, 20))
graph_info_top_df = graph_info_df |> (df -> first(df, 20))

CSV.write("outputs/nlab_graph_info.csv", graph_info_df, delim="\t")
CSV.write("outputs/nlab_graph_info_top.csv", graph_info_top_df, delim="\t")

@time degree_centrality(graph);


@time pgrk = pagerank(graph);

graph_info = make_graph_info_df(graph)

# Exporting

savegraph("outputs/nlab_graph.dot", graph, MetaGraphs.DOTFormat())

using SGtSNEpi

y = sgtsnepi(graph)

using CairoMakie, Colors, LinearAlgebra

show_embedding(y, res=(2000,2000))

using GraphDataFrameBridge


struct GraphDFs
    node_df::DataFrame
    edge_df::DataFrame
end


function get_edge_df(graph)
    extract_src_dst(edge) = (edge.src, edge.dst)
    raw_df = GraphDataFrameBridge.DataFrame(graph, type=:edge)
    raw_df[:, :edge_tuple] = extract_src_dst.(raw_df[:, :edge])
    DataFrame(src=first.(raw_df[:, :edge_tuple]), dst=last.(raw_df[:, :edge_tuple]))
end

function get_graph_dfs(graph)
    node_df = GraphDataFrameBridge.DataFrame(graph)
    edge_df = get_edge_df(graph) 
    GraphDFs(node_df, edge_df)
end


graph_dfs = get_graph_dfs(graph)


CSV.write("outputs/nlab_graph_nodes.csv", graph_dfs.node_df, delim="\t")
CSV.write("outputs/nlab_graph_edges.csv", graph_dfs.edge_df, delim=",")
