use pyo3::prelude::*;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy};

pub struct IndexWrapper {
    index: Index,
    schema: Schema,
}

fn from_tantivy_result<T>(tantivy_res: tantivy::Result<T>) -> Result<T, PyErr> {
    match tantivy_res {
        Ok(res) => Ok(res),
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            err.to_string(),
        )),
    }
}

impl IndexWrapper {
    pub fn new(index_name: String, name_field: String, fields: Vec<String>) -> Self {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field(&name_field.to_string(), TEXT | STORED);
        for field in fields {
            schema_builder.add_text_field(&field.to_string(), TEXT);
        }
        let schema = schema_builder.build();
        let index = Index::create_in_ram(schema.clone());
        IndexWrapper { index, schema }
    }

    // TODO: accept a hashmap
    fn try_add_document(&self, title: &str, text: &str) -> tantivy::Result<()> {
        let mut index_writer = self.index.writer(50_000_000)?;
        let title_field = self.schema.get_field("title").unwrap();
        let body_field = self.schema.get_field("body").unwrap();
        index_writer.add_document(doc!(
            title_field => title,
            body_field => text,
        ));
        index_writer.commit()?;
        Ok(())
    }

    pub fn add_document(&self, title: &str, text: &str) -> Result<(), PyErr> {
        from_tantivy_result(self.try_add_document(title, text))
    }

    fn try_search(&self, query: &str) -> tantivy::Result<Vec<String>> {
        let reader = self.index.reader()?;
        let searcher = reader.searcher();
        let title_field = self.schema.get_field("title").unwrap();
        let body_field = self.schema.get_field("body").unwrap();
        let query_parser = QueryParser::for_index(&self.index, vec![title_field, body_field]);
        let query = query_parser.parse_query(query)?;
        let top_docs = searcher.search(&query, &TopDocs::with_limit(10))?;
        let mut results = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc = searcher.doc(doc_address)?;
            let title = retrieved_doc
                .get_first(title_field)
                .unwrap()
                .as_text()
                .unwrap();
            results.push(title.to_string());
        }
        Ok(results)
    }

    pub fn search(&self, query: &str) -> Result<Vec<String>, PyErr> {
        from_tantivy_result(self.try_search(query))
    }
}
