use crate::wrappers::TantivyIndexWrapper;
use polars::prelude::*;

use polars::datatypes::DataType;
use std::collections::HashMap;
trait IndexableCollection {
    fn index_collection(&self, index: &TantivyIndexWrapper) -> tantivy::Result<()>;
}

impl IndexableCollection for DataFrame {
    fn index_collection(&self, tantivy_index: &TantivyIndexWrapper) -> tantivy::Result<()> {
        df_rows_foreach(&self, &|row_hashmap| {
            tantivy_index.add_document(row_hashmap)
        })?;
        Ok(())
    }
}

pub fn df_rows_foreach<E>(
    df: &DataFrame,
    function: &dyn Fn(HashMap<String, String>) -> Result<(), E>,
) -> Result<(), E> {
    // Initialize an empty vector of hashmaps
    // Get the number of rows
    let num_rows = df.height();

    // Iterate over each row
    for row_index in 0..num_rows {
        // Create a new hashmap for each row
        let mut row_hashmap: HashMap<String, String> = HashMap::new();

        // Iterate over each of the columns, add name-value entries to the hashmap
        df.get_columns().iter().for_each(|s| {
            let name = s.name().to_string();
            match s.dtype() {
                DataType::String => {
                    let val = s.str().unwrap().get(row_index);
                    let str = val.unwrap_or("None").clone();
                    row_hashmap.insert(name, str.to_string());
                }
                _ => (),
            }
        });

        // Push the hashmap into the vector
        function(row_hashmap)?;
    }
    Ok(())
}

#[test]
fn test_indexing_df() {
    let index = TantivyIndexWrapper::new(
        "test_index".to_string(),
        "title".to_string(),
        vec!["body".to_string()],
    );

    let df = load_test_df();
    df.index_collection(&index).unwrap();

    let num_docs = index.num_docs().unwrap();
    assert_eq!(num_docs, 2);
}

fn load_test_df() -> DataFrame {
    let df = DataFrame::new(vec![
        Series::new("title", &["test title", "test title 2"]),
        Series::new("body", &["test body", "test body 2"]),
    ])
    .unwrap();
    df
}
