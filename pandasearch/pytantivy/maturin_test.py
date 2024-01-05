import pytantivy

# pytantivy.print_dict({"foo": "1"})

pytantivy.initialize_index("foo", "title", ["body"])

query = "title"
pytantivy.index_document("foo", "a title", "a text")


results = pytantivy.search("foo", query)
print("results for query: ", query)
print(results)
