mod index_registry;
pub mod polars_documents;
pub mod wrappers;

use index_registry::IndexRegistry;
use lazy_static::lazy_static;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use std::collections::HashMap;
use tantivy::Result as TantivyResult;
use wrappers::TantivyIndexWrapper;

lazy_static! {
    static ref INDEX_REGISTRY: IndexRegistry = IndexRegistry::new();
}

// TODO: make a pyclass so that the functions do not have to pass index name
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn initialize_index(name: &PyString, name_field: &PyString, fields: &PyList) -> PyResult<()> {
    let fields_list = fields.extract::<Vec<String>>()?;
    INDEX_REGISTRY.initialize_index(name.to_string(), name_field.to_string(), fields_list)?;
    Ok(())
}

#[pyfunction]
fn index_document<'a>(name: &'a PyString, document_dict: &'a PyDict) -> PyResult<()> {
    let document_map = document_dict.extract::<HashMap<String, String>>()?;

    INDEX_REGISTRY.index_document(name.to_string(), document_map)
}

#[pyfunction]
fn search<'a>(py: Python<'a>, name: &'a PyString, query: &'a PyString) -> PyResult<&'a PyList> {
    let vec_results = INDEX_REGISTRY.search(name.to_string(), query.to_string())?;
    Ok(PyList::new(py, vec_results))
}

#[pyfunction]
fn get_index_names<'a>(py: Python<'a>) -> PyResult<&'a PyList> {
    let index_names = INDEX_REGISTRY.get_index_names()?;
    Ok(PyList::new(py, index_names))
}

#[pymodule]
#[pyo3(name = "pytantivy")]
fn pytantivy(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_index, m)?)?;
    m.add_function(wrap_pyfunction!(index_document, m)?)?;
    m.add_function(wrap_pyfunction!(search, m)?)?;

    Ok(())
}
