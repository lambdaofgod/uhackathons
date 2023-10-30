module OrgGraph
using JSON

export OrgLink, load_links

@kwdef struct OrgLink
    source_name::String
    destination_name::String
    source_date::String
    destination_date::String
end


function load_links(path)
    records = JSON.parse(read(path, String))
    links = [dict_to_struct(OrgLink, record) for record in records]
end

function get_input_dict(struct_type, dict)
    fields = fieldnames(struct_type)
    Dict(
        [Symbol(k) => v for (k, v) in dict if Symbol(k) in fields]
    )
end

function dict_to_struct(struct_type, dict)
    mapped_dict = get_input_dict(struct_type, dict)
    struct_type(; mapped_dict...)
end

end
