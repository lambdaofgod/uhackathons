use crate::polars_documents::IndexableCollection;
use crate::wrappers::TantivyIndexWrapper;
use polars::prelude::DataFrame;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct IndexRegistry {
    indices: RwLock<HashMap<String, Box<TantivyIndexWrapper>>>,
}

fn from_tantivy_result<T>(tantivy_res: tantivy::Result<T>) -> Result<T, PyErr> {
    match tantivy_res {
        Ok(res) => Ok(res),
        Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            err.to_string(),
        )),
    }
}

impl IndexRegistry {
    pub fn new() -> Self {
        IndexRegistry {
            indices: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_index_names(&self) -> Result<Vec<String>, PyErr> {
        let binding = self.indices.read();
        match binding {
            Ok(binding) => Ok(binding.keys().cloned().collect()),
            Err(err) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                err.to_string(),
            )),
        }
    }

    pub fn initialize_index(
        &self,
        name: String,
        name_field: String,
        fields: Vec<String>,
    ) -> Result<(), PyErr> {
        let index_wrapper = TantivyIndexWrapper::new(name.clone(), name_field, fields);
        self.indices
            .write()
            .unwrap()
            .insert(name, Box::new(index_wrapper));
        Ok(())
    }

    pub fn index_document(
        &self,
        name: String,
        document_map: HashMap<String, String>,
    ) -> Result<(), PyErr> {
        let binding = self.indices.read().unwrap();
        let index_wrapper = binding.get(&name).unwrap();

        from_tantivy_result(index_wrapper.add_document(document_map))
    }

    pub fn search(&self, name: String, query: String) -> Result<Vec<String>, PyErr> {
        let binding = self.indices.read().unwrap();
        let index_wrapper = binding.get(&name.to_string()).unwrap();

        from_tantivy_result(index_wrapper.search(&query.to_string()))
    }

    pub fn index_df(&self, name: String, df: &DataFrame) -> Result<(), PyErr> {
        let binding = self.indices.read().unwrap();
        let index_wrapper = binding.get(&name.to_string()).unwrap();

        from_tantivy_result(df.index_collection(index_wrapper))
    }
}
