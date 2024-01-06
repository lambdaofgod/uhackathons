use lazy_static::lazy_static;
use pyo3::prelude::*;
mod index_registry;
mod wrappers;
use index_registry::IndexRegistry;
use pyo3::types::{PyDict, PyList, PyString};
use tantivy::Result as TantivyResult;
use wrappers::TantivityIndexWrapper;

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
fn index_document(name: &PyString, title: &PyString, text: &PyString) -> PyResult<()> {
    INDEX_REGISTRY.index_document(name.to_string(), title.to_string(), text.to_string())
}

#[pyfunction]
fn search<'a>(py: Python<'a>, name: &'a PyString, query: &'a PyString) -> PyResult<&'a PyList> {
    let vec_results = INDEX_REGISTRY.search(name.to_string(), query.to_string())?;
    Ok(PyList::new(py, vec_results))
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
