use std::{collections::HashMap, convert::From, env};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use pyo3::prelude::*;

use crate::{cube::PhotonCube, signals::DeferedSignal, transforms::Transform};

/// Convert a photon cube (npy file/directory of bin files) between formats or to
/// a video preview (mp4) by naively averaging frames.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    #[clap(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    /// Convert photoncube from a collection of .bin files to a .npy file.
    Convert(ConvertArgs),

    /// Extract and preview virtual exposures from a photoncube.
    Preview(PreviewArgs),

    /// Apply corrections directly to every bitplane and save results as new .npy file.
    Process(ProcessArgs),
}

#[derive(Debug, Args)]
pub struct ConvertArgs {
    /// Path of input photon cube (directory of .bin files)
    #[arg(short, long)]
    pub input: String,

    /// Path of output photon cube (must be .npy)
    #[arg(short, long)]
    pub output: String,

    /// If enabled, assume data is 512x512
    #[arg(long, action)]
    pub full_array: bool,
}

// Ensures that at least one of these two is set.
#[derive(Debug, Args)]
#[group(required = true, multiple = true)]
pub struct OutputGroup {
    /// Path of output video
    #[arg(short, long)]
    pub output: Option<String>,

    /// Output directory to save PNGs in
    #[arg(short = 'd', long)]
    pub img_dir: Option<String>,
}

// Ensures that at most one of these two is set.
#[derive(Debug, Args, Clone, Copy)]
#[group(required = false, multiple = false)]
pub struct FixGroup {
    /// If enabled, swap columns that are out of order and crop to 254x496
    #[arg(long, action)]
    pub colorspad_fix: bool,

    /// If enabled, insert missing row between array halves and crop to 509x496
    #[arg(long, action)]
    pub grayspad_fix: bool,
}

#[derive(Debug, Args, Clone)]
pub struct PreviewProcessCommonArgs {
    /// Path to photon cube (.npy file expected)
    #[arg(short, long)]
    pub input: String,

    /// Path of color filter array to use for demosaicing
    #[arg(long, default_value = None)]
    pub cfa_path: Option<String>,

    /// Path of inpainting mask to use for filtering out dead/hot pixels
    #[arg(long, num_args(0..))]
    pub inpaint_path: Vec<String>,

    /// Apply transformations to each frame (these can be composed)
    #[arg(short, long, value_enum, num_args(0..))]
    pub transform: Vec<Transform>,

    #[clap(flatten)]
    pub fix: FixGroup,

    /// Index of binary frame at which to start the preview from (inclusive)
    #[arg(short, long, default_value = None)]
    pub start: Option<isize>,

    /// Index of binary frame at which to end the preview at (exclusive)
    #[arg(short, long, default_value = None)]
    pub end: Option<isize>,
}

#[derive(Debug, Args)]
pub struct PreviewArgs {
    #[clap(flatten)]
    pub common: PreviewProcessCommonArgs,

    #[clap(flatten)]
    pub outputs: OutputGroup,

    /// Number of frames to average together
    #[arg(short, long, default_value_t = 256)]
    pub burst_size: usize,

    /// Frame rate of resulting video
    #[arg(long, default_value_t = 25)]
    pub fps: u64,

    /// If enabled, add bitplane indices to images
    #[arg(short, long, action)]
    pub annotate_frames: bool,

    /// If enabled, invert the SPAD's response (Bernoulli process)
    #[arg(long, action)]
    pub invert_response: bool,

    /// When inverting the SPAD's response, use this quantile to normalize data
    #[arg(long, default_value = None)]
    pub quantile: Option<f32>,

    /// If enabled, apply sRGB tonemapping to output
    #[arg(long, action)]
    pub tonemap2srgb: bool,
}

#[derive(Debug, Args)]
pub struct ProcessArgs {
    #[clap(flatten)]
    pub common: PreviewProcessCommonArgs,

    /// Path of output .npy file
    #[arg(short, long)]
    pub output: String,
}

fn load_cube(args: &PreviewProcessCommonArgs, step: Option<usize>, quantile: Option<f32>) -> Result<PhotonCube> {
    // Load all the neccesary files
    let mut cube = PhotonCube::open(&args.input)?;
    if let Some(cfa_path) = &args.cfa_path {
        cube.load_cfa(&cfa_path)?;
    }
    for inpaint_path in args.inpaint_path.iter() {
        cube.load_mask(inpaint_path)?;
    }
    cube.set_range(args.start.unwrap_or(0), args.end, step);
    cube.set_transforms(args.transform.clone());
    cube.set_quantile(quantile.clone());
    Ok(cube)
}

fn preview(args: PreviewArgs) -> Result<()> {
    // Load all the neccesary files
    let cube = load_cube(&args.common, Some(args.burst_size), args.quantile)?;
    let process = cube.process_single(
        args.invert_response,
        args.tonemap2srgb,
        args.common.fix.colorspad_fix,
        args.common.fix.grayspad_fix,
    )?;

    // Generate preview
    if let Some(output) = args.outputs.output {
        let cmd = format!(
            "Created using: '{}'",
            env::args().skip(1).collect::<Vec<_>>().join(" ")
        );

        cube.save_video(
            output.as_str(),
            args.fps,
            args.outputs.img_dir.as_deref(),
            Some(process),
            args.annotate_frames,
            Some("Making video..."),
            Some(HashMap::from([("comment", cmd.as_str())])),
        )?;
    } else {
        cube.save_images(
            args.outputs.img_dir.unwrap().as_str(),
            Some(process),
            args.annotate_frames,
            Some("Processing Frames..."),
        )?;
    };

    Ok(())
}

fn process(args: ProcessArgs) -> Result<()> {
    // Load all the neccesary files
    let cube = load_cube(&args.common, None, None)?;
    let shape = cube.process_cube(
        args.output.as_str(),
        args.common.fix.colorspad_fix,
        args.common.fix.grayspad_fix,
        Some("Processing Cube..."),
    )?;
    println!("Saved cube of shape {shape:?} to {}", args.output.as_str());
    Ok(())
}

#[pyfunction]
pub fn cli_entrypoint(py: Python) -> Result<()> {
    // Start by telling python to not intercept CTRL+C signal,
    // Otherwise we won't get it here and will not be interruptable.
    // See: https://github.com/PyO3/pyo3/pull/3560
    let _defer = DeferedSignal::new(py, "SIGINT")?;

    // Parse arguments defined in struct
    // Since we're actually calling this via python, the first argument
    // is going to be the path to the python interpreter, so we skip it.
    // See: https://www.maturin.rs/bindings#both-binary-and-library
    let args = Cli::parse_from(env::args_os().skip(1));

    match args.command {
        Commands::Convert(args) => Ok(PhotonCube::convert_to_npy(
            &args.input,
            &args.output,
            args.full_array,
            Some("Converting..."),
        )?),
        Commands::Preview(args) => preview(args),
        Commands::Process(args) => process(args),
    }
}
