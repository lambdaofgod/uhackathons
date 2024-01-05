use lazy_static::lazy_static;
use pyo3::prelude::*;
mod wrappers;
use pyo3::types::{PyDict, PyList, PyString};
use std::collections::HashMap;
use std::sync::Mutex;
use tantivy::Result as TantivyResult;
use wrappers::IndexWrapper;

lazy_static! {
    static ref INDEXWRAPPERS: Mutex<HashMap<String, Box<IndexWrapper>>> =
        Mutex::new(HashMap::new());
}

// TODO: make a pyclass so that the functions do not have to pass index name

#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn initialize_index(name: &PyString, name_field: &PyString, fields: &PyList) -> PyResult<()> {
    let fields_list = fields.extract::<Vec<String>>()?;
    let index_wrapper = IndexWrapper::new(
        name.to_string(),
        name_field.to_string(),
        fields.iter().map(|x| x.to_string()).collect(),
    );
    INDEXWRAPPERS
        .lock()
        .unwrap()
        .insert(name.to_string(), Box::new(index_wrapper));
    Ok(())
}

#[pyfunction]
fn index_document(name: &PyString, title: &PyString, text: &PyString) -> PyResult<()> {
    let mut binding = INDEXWRAPPERS.lock().unwrap();
    let index_wrapper = binding.get_mut(&name.to_string()).unwrap();

    index_wrapper.add_document(&title.to_string(), &text.to_string())
}

#[pyfunction]
fn search<'a>(py: Python<'a>, name: &'a PyString, query: &'a PyString) -> PyResult<&'a PyList> {
    let mut binding = INDEXWRAPPERS.lock().unwrap();
    let index_wrapper = binding.get_mut(&name.to_string()).unwrap();

    let vec_results = index_wrapper.search(&query.to_string())?;
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
