use anyhow::{Error as E, Result};

pub fn anyhow_to_std_result<T>(res: Result<T>) -> std::result::Result<T, String> {
    match res {
        Ok(v) => Ok(v),
        Err(e) => Err(e.to_string()),
    }
}
