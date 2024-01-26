#![allow(dead_code)] // Todo: Remove
                     // #![allow(unused_imports)]

use std::collections::HashMap;
use std::convert::From;
use std::fs::{create_dir_all, File};
use std::io::{BufReader, Read};
use std::ops::BitOr;
use std::path::Path;

use anyhow::{anyhow, Result};
use ffmpeg::{ensure_ffmpeg, make_video};
use flate2::read::GzDecoder;
use image::Rgb;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use memmap2::Mmap;
use rayon::prelude::*;
use tempfile::tempdir;

use ndarray::{Array, Array2, Array3, ArrayView3, Axis, Slice};
use ndarray_npy::{ReadNpyExt, ViewNpyExt};

mod cli;
mod ffmpeg;
mod io;
mod transforms;
mod utils;

use crate::cli::*;
use crate::io::*;
use crate::transforms::*;
use crate::utils::sorted_glob;

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() -> Result<()> {
    // Parse arguments defined in struct
    let args = Args::parse();

    // Ensure the user set at least one output
    if args.output.is_none() && args.img_dir.is_none() {
        return Err(anyhow!(
            "At least one output needs to be specified. Please either set --output or --img-dir (or both)."
        ));
    }

    // Load all the neccesary files
    let path = Path::new(&args.input);
    let _arr: Array3<u8>; // These are needed to keep the underlying object's data in scope
    let _mmap: Mmap; // otherwise we get a use-after-free error.

    let mut cube: ArrayView3<u8> = if !path.exists() {
        // This should probably be a specific IO error?
        return Err(anyhow!("File not found at {}!", args.input));
    } else if path.is_dir() {
        let (h, w) = (256, 512);
        let paths = sorted_glob(path, "**/*.bin")?;

        if paths.is_empty() {
            return Err(anyhow!("No .bin files found in {}!", args.input));
        }

        let mut buffer = Vec::new();

        for p in paths {
            let mut f = File::open(p)?;
            f.read_to_end(&mut buffer)?;
        }

        let t = buffer.len() / (h * w / 8);
        _arr = Array::from_vec(buffer)
            .into_shape((t, h, w / 8))?
            .mapv(|v| v.reverse_bits());
        _arr.view()
    } else {
        let ext = path.extension().unwrap().to_ascii_lowercase();

        if ext != "npy" && ext != "gz" {
            // This should probably be a specific IO error?
            return Err(anyhow!(
                "Expexted numpy array with extension `npy`, or `npy.gz`, got {:?}.",
                ext
            ));
        }

        if ext == "gz" {
            let file = File::open(args.input).unwrap();
            let file = BufReader::new(file);
            let mut file = GzDecoder::new(file);
            let mut bytes = Vec::new();
            file.read_to_end(&mut bytes).unwrap();

            _arr = Array3::<u8>::read_npy(&bytes[..])?;
            _arr.view()
        } else {
            let file = File::open(args.input)?;
            _mmap = unsafe { Mmap::map(&file)? };

            ArrayView3::<u8>::view_npy(&_mmap)?
        }
    };

    let inpaint_mask: Option<Array2<bool>> = if !args.inpaint_path.is_empty() {
        // Vec<Result<Option<Array2<bool>>>> -> Result<Vec<Option<Array2<bool>>>> -> Vec<Option<Array2<bool>>>
        let mask_vec = args
            .inpaint_path
            .iter()
            .map(|path| try_load_mask(Some(path.to_string())))
            .collect::<Result<Vec<_>>>()?;
        let mask_vec: Option<Vec<_>> = mask_vec.into_iter().collect();
        mask_vec.unwrap().into_iter().reduce(|acc, e| acc.bitor(e))
    } else {
        None
    };

    // let inpaint_mask: Option<Array2<bool>> = try_load_mask(args.inpaint_path)?;
    let cfa_mask: Option<Array2<bool>> = try_load_mask(args.cfa_path)?;
    ensure_ffmpeg(true);

    // Get img path or tempdir, ensure it exists.
    let tmp_dir = tempdir()?;
    let img_dir = args
        .img_dir
        .unwrap_or(tmp_dir.path().to_str().unwrap().to_owned());
    create_dir_all(&img_dir).ok();

    // // Create chunked iterator over all data
    let start_offset = args.start.unwrap_or(0);
    cube.slice_axis_inplace(Axis(0), Slice::new(start_offset, args.end, 1));
    let groups = cube.axis_chunks_iter(Axis(0), args.burst_size);
    let pbar_style = ProgressStyle::with_template(
        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
    )
    .unwrap();

    // Create parallel iterator over all chunks of frames and process them
    let num_frames = groups
        // Make it parallel
        .into_par_iter()
        // Add progress bar
        .progress_count(cube.len_of(Axis(0)) as u64 / args.burst_size as u64)
        .with_style(pbar_style.clone())
        .enumerate()
        // Process data, save and count frames, this consumes/runs the iterator to completion
        .try_fold(
            || 0,
            |count, (i, group)| {
                let frame = group
                    // Unpack every frame in group
                    .axis_iter(Axis(0))
                    .map(|bitplane| unpack_single::<f32>(&bitplane, 1).unwrap())
                    // Sum frames together, use reduce not `.sum` as it's not
                    // implemented for this type, maybe use `accumulate_axis_inplace`
                    .reduce(|acc, e| acc + e)
                    .unwrap();

                // Normalize by burst_size, giving us an f32 image in [0, 1]
                let mut frame = frame / (args.burst_size as f32);

                // Invert SPAD response
                if args.invert_response {
                    frame = binary_avg_to_rgb(frame, 1.0, None);
                }

                // Apply sRGB tonemapping
                if args.tonemap2srgb {
                    frame = linearrgb_to_srgb(frame);
                }

                // Convert to uint8 in [0, 255] range
                let mut frame = frame.mapv(|x| (x * 255.0) as u8);

                // Apply any frame-level fixes (only for ColorSPAD at the moment)
                if args.colorspad_fix {
                    frame = process_colorspad(frame);
                }

                // Demosaic frame by interpolating white pixels
                if let Some(mask) = &cfa_mask {
                    frame = interpolate_where_mask(frame, mask, false)?;
                }

                // Inpaint any hot/dead pixels
                if let Some(mask) = &inpaint_mask {
                    frame = interpolate_where_mask(frame, mask, false)?;
                }

                // Convert to image and rotate/flip as needed
                let frame = array2_to_grayimage(frame.to_owned());
                let frame = apply_transform(frame, &args.transform);

                if args.annotate {
                    let text = format!(
                        "{:06}:{:06}",
                        i * args.burst_size + (start_offset as usize),
                        (i + 1) * args.burst_size + (start_offset as usize)
                    );
                    let mut frame = gray_to_rgbimage(&frame);
                    annotate(&mut frame, &text, Rgb([252, 186, 3]));

                    // Throw error if we cannot save, and stop all processing.
                    let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
                    frame.save(&path)?;
                } else {
                    // Throw error if we cannot save, and stop all processing.
                    let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
                    frame.save(&path)?;
                }

                Ok::<u32, anyhow::Error>(count + 1)
            },
        )
        .try_reduce(|| 0, |acc, x| Ok(acc + x))?;

    // Finally, make a call to ffmpeg to assemble to video
    let cmd = format!(
        "Created using: '{}'",
        std::env::args().collect::<Vec<_>>().join(" ")
    );

    if let Some(output) = args.output {
        make_video(
            Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
            &output,
            args.fps,
            num_frames as u64,
            Some(pbar_style),
            Some(HashMap::from([("comment", cmd.as_str())])),
        );
    }
    tmp_dir.close()?;
    Ok(())
}
