use crate::wrappers::TantivityIndexWrapper;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::RwLock;

pub struct IndexRegistry {
    indices: RwLock<HashMap<String, Box<TantivityIndexWrapper>>>,
}

impl IndexRegistry {
    pub fn new() -> Self {
        IndexRegistry {
            indices: RwLock::new(HashMap::new()),
        }
    }

    pub fn initialize_index(
        &self,
        name: String,
        name_field: String,
        fields: Vec<String>,
    ) -> Result<(), PyErr> {
        let index_wrapper = TantivityIndexWrapper::new(name.clone(), name_field, fields);
        self.indices
            .write()
            .unwrap()
            .insert(name, Box::new(index_wrapper));
        Ok(())
    }

    pub fn index_document(&self, name: String, title: String, text: String) -> Result<(), PyErr> {
        let binding = self.indices.read().unwrap();
        let index_wrapper = binding.get(&name.to_string()).unwrap();

        index_wrapper.add_document(&title.to_string(), &text.to_string())
    }

    pub fn search(&self, name: String, query: String) -> Result<Vec<String>, PyErr> {
        let binding = self.indices.read().unwrap();
        let index_wrapper = binding.get(&name.to_string()).unwrap();

        index_wrapper.search(&query.to_string())
    }
}
