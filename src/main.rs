extern crate pyo3;

use pyo3::prelude::*;

fn main() -> PyResult<()> {
    let code = include_str!("hello.py");
    Python::with_gil(|py| -> PyResult<()> {
        PyModule::from_code(py, code, "hello", "hello")?;

        Ok(())
    })?;

    Ok(())
}