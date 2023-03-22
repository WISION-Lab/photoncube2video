pub use anyhow::{anyhow, Result};

pub use ndarray::prelude::*;
pub use std::path::Path;

use image::io::Reader as ImageReader;
use ndarray_npy::read_npy;
use nshare::ToNdarray3;

use ffmpeg_sidecar::command::{ffmpeg_is_installed, FfmpegCommand};
use ffmpeg_sidecar::paths::sidecar_dir;
use indicatif::{ProgressBar, ProgressStyle};

pub fn ensure_ffmpeg(verbose: bool) {
    if !ffmpeg_is_installed() {
        if verbose {
            println!(
                "No ffmpeg installation found, downloading one to {}...",
                &sidecar_dir().unwrap().display()
            );
        }
        ffmpeg_sidecar::download::auto_download().unwrap();
    }
}

pub fn make_video(
    pattern: &str,
    outfile: &str,
    fps: u64,
    num_frames: u64,
    pbar_style: Option<ProgressStyle>,
) {
    let pbar = if let Some(style) = pbar_style {
        ProgressBar::new(num_frames).with_style(style)
    } else {
        ProgressBar::hidden()
    };

    let cmd = format!(
        "-framerate {fps} -f image2 -i {pattern} -y -vcodec libx264 -crf 22 -pix_fmt yuv420p {outfile}"
    );

    let mut ffmpeg_runner = FfmpegCommand::new().args(cmd.split(' ')).spawn().unwrap();
    ffmpeg_runner
        .iter()
        .unwrap()
        .filter_progress()
        .for_each(|progress| pbar.set_position(progress.frame as u64));
    pbar.finish_and_clear();
}

/// Given an Option of a path, try to load the image or npy file at that path
/// and return it as a Result of an Option of array. Bubble up any io errors.  
/// Currently only supports RGB8-type images.
pub fn try_load(path: Option<String>) -> Result<Option<Array3<u8>>> {
    if path.is_none() {
        return Ok(None);
    }

    let path_str = path.unwrap();
    let path = Path::new(&path_str);
    let ext = path.extension().unwrap().to_ascii_lowercase();

    if !path.exists() {
        // This should probably be a specific IO error?
        Err(anyhow!("File not found at {}!", path_str))
    } else if ext == "npy" || ext == "npz" {
        let arr = read_npy::<std::string::String, Array3<u8>>(path_str)?.into_dimensionality()?;
        Ok(Some(arr))
    } else {
        let arr = ImageReader::open(path_str)?
            .decode()?
            .into_rgb8()
            .into_ndarray3();
        Ok(Some(arr))
    }
}

// let cfa_mask: Option<Array2<bool>> = match args.cfa_path {
//     Some(cfa_path) => {
//         let cfa_mask: Array2<bool> = ImageReader::open(cfa_path)?
//             .decode()?
//             .into_rgb8()
//             .into_ndarray3()
//             .mapv(|v| v == 0)
//             .map_axis(Axis(0), |p| {
//                 p.to_vec().into_iter().reduce(|a, b| a | b).unwrap()
//             });
//         Some(cfa_mask)
//     }
//     None => None,
// };
// let inpaint_mask: Option<Array2<bool>> = args.inpaint_path.map(|p| read_npy(p).unwrap());
