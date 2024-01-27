#![allow(dead_code)] // Todo: Remove
                     // #![allow(unused_imports)]

use std::{collections::HashMap, convert::From, fs::create_dir_all, path::Path};

use anyhow::{anyhow, Result};
use clap::Parser;
use tempfile::tempdir;

use crate::{
    cli::Cli,
    cube::PhotonCube,
    ffmpeg::{ensure_ffmpeg, make_video},
};

mod cli;
mod cube;
mod ffmpeg;
mod transforms;
mod utils;

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn main() -> Result<()> {
    // Parse arguments defined in struct
    let args = Cli::parse();

    // if let Some(dst) = args.output {
    //     PhotonCube::convert_to_npy(&args.input, &dst, Some("Converting..."))?;
    // }
    // return Ok(());

    // Ensure the user set at least one output
    if args.output.is_none() && args.img_dir.is_none() {
        return Err(anyhow!(
            "At least one output needs to be specified. Please either set --output or --img-dir (or both)."
        ));
    }

    // Load all the neccesary files
    let mut cube = PhotonCube::open(&args.input)?;
    if let Some(cfa_path) = args.cfa_path {
        cube.load_cfa(&cfa_path)?;
    }
    for inpaint_path in args.inpaint_path.iter() {
        cube.load_mask(inpaint_path)?;
    }

    // Get img path or tempdir, ensure it exists.
    let tmp_dir = tempdir()?;
    let img_dir = args
        .img_dir
        .unwrap_or(tmp_dir.path().to_str().unwrap().to_owned());
    create_dir_all(&img_dir).ok();

    // Generate preview frames
    cube.set_range(args.start.unwrap_or(0), args.end, args.burst_size);
    let process = cube.process_single(args.invert_response, args.tonemap2srgb, args.colorspad_fix);
    let num_frames = cube.save_images(
        &img_dir,
        Some(process),
        args.annotate_frames,
        &args.transform[..],
        Some("Processing Frames..."),
    )?;

    // Finally, make a call to ffmpeg to assemble to video
    let cmd = format!(
        "Created using: '{}'",
        std::env::args().collect::<Vec<_>>().join(" ")
    );

    if let Some(output) = args.output {
        ensure_ffmpeg(true);
        make_video(
            Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
            &output,
            args.fps,
            num_frames as u64,
            Some("Making video..."),
            Some(HashMap::from([("comment", cmd.as_str())])),
        );
    }
    tmp_dir.close()?;
    Ok(())
}
