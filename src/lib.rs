pub mod cli;
pub mod cube;
pub mod ffmpeg;
pub mod signals;
pub mod transforms;
pub mod utils;

use pyo3::prelude::*;
use transforms::Transform;

use crate::{cli::cli_entrypoint, cube::PhotonCube};

#[pymodule]
fn photoncube(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(cli_entrypoint, m)?)?;
    m.add_class::<PhotonCube>()?;
    m.add_class::<Transform>()?;

    Ok(())
}
