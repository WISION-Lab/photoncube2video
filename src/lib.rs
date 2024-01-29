pub mod cli;
pub mod cube;
pub mod ffmpeg;
pub mod signals;
pub mod transforms;
pub mod utils;

use pyo3::prelude::*;

use crate::{cli::__pyo3_get_function_cli_entrypoint, cube::PhotonCube};

#[pymodule]
fn photoncube2video(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PhotonCube>()?;
    m.add_wrapped(wrap_pyfunction!(cli_entrypoint))?;

    Ok(())
}
