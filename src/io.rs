pub use anyhow::{anyhow, Result};

pub use ndarray::prelude::*;
pub use std::path::Path;

use image::io::Reader as ImageReader;
use ndarray_npy::read_npy;
use nshare::ToNdarray2;
use std::fs::File;
use std::io::prelude::*;

use ffmpeg_sidecar::command::{ffmpeg_is_installed, FfmpegCommand};
use ffmpeg_sidecar::paths::sidecar_dir;
use indicatif::{ProgressBar, ProgressStyle};

use crate::utils::sorted_glob;

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
pub fn try_load_cube(path_str: String, shape: Option<(usize, usize)>) -> Result<Array3<u8>> {
    let path = Path::new(&path_str);

    if !path.exists() {
        // This should probably be a specific IO error?
        Err(anyhow!("File not found at {}!", path_str))
    } else if path.is_dir() {
        let (h, w) = shape.expect("Must specify shape if reading from .bin files");
        let paths = sorted_glob(path, "**/*.bin")?;

        if paths.is_empty() {
            return Err(anyhow!("No .bin files found in {}!", path_str));
        }

        let mut buffer = Vec::new();

        for p in paths {
            let mut f = File::open(p)?;
            f.read_to_end(&mut buffer)?;
        }

        let t = buffer.len() / (h * w / 8);
        let arr = Array::from_vec(buffer).into_shape((t, h, w / 8))?;
        Ok(arr.mapv(|v| v.reverse_bits()))
    } else {
        let ext = path.extension().unwrap().to_ascii_lowercase();

        if ext != "npy" && ext != "npz" {
            // This should probably be a specific IO error?
            Err(anyhow!(
                "Expexted numpy array with extension `npy` or `npz`, got {:?}.",
                ext
            ))
        } else {
            let arr: Array3<u8> = read_npy(path_str)?;
            Ok(arr)
        }
    }
}

/// Load either a 2D NPY file or an intensity-only image file as an array of booleans.
/// Note: For the image, any pure white pixels are false, all others are true.
///       This is contrary to what you might expect but enables us to load in the 
///       colorSPAD's cfa array and have a mask representing the colored pixels.
pub fn try_load_mask(path: Option<String>) -> Result<Option<Array2<bool>>> {
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
        let arr: Array2<bool> = read_npy(path_str)?;
        Ok(Some(arr))
    } else {
        let arr = ImageReader::open(path_str)?
            .decode()?
            .into_luma8()
            .into_ndarray2()
            .mapv(|v| v != 0);
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
