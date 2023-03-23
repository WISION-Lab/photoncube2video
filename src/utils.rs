use std::path::{Path, PathBuf};

pub use anyhow::Result;
use glob::glob;
use natord::compare;

pub fn sorted_glob(path: &Path, pattern: &str) -> Result<Vec<String>> {
    let paths: Vec<PathBuf> =
        glob(path.join(pattern).to_str().unwrap())?.collect::<Result<Vec<PathBuf>, _>>()?;
    let paths: Vec<&str> = paths
        .iter()
        .map(|v| v.to_str())
        .collect::<Option<Vec<&str>>>()
        .unwrap();
    let mut paths: Vec<String> = paths.iter().map(|p| p.to_string()).collect();
    paths.sort_by(|a, b| compare(a, b));

    Ok(paths)
}
