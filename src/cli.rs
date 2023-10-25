pub use clap::{Parser, ValueEnum};

/// Convert a photon cube (npy file/directory of bin files) to a video preview (mp4) by naively averaging frames.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to photon cube (npy file or directory of .bin files)
    #[arg(short, long)]
    pub input: String,

    /// Path of output video
    #[arg(short, long, default_value = "out.mp4")]
    pub output: String,

    /// Output directory to save PNGs in
    #[arg(long)]
    pub img_dir: Option<String>,

    /// Path of color filter array to use for demosaicing
    #[arg(long, default_value = None)]
    pub cfa_path: Option<String>,

    /// Path of inpainting mask to use for filtering out dead/hot pixels
    #[arg(long, num_args(0..))]
    pub inpaint_path: Vec<String>,

    /// Number of frames to average together
    #[arg(short, long, default_value_t = 256)]
    pub burst_size: usize,

    /// Frame rate of resulting video
    #[arg(long, default_value_t = 25)]
    pub fps: u64,

    /// Apply transformations to each frame (these can be composed)
    #[arg(short, long, value_enum, num_args(0..))]
    pub transform: Vec<Transform>,

    /// If enabled, add bitplane indices to images
    #[arg(short, long, action)]
    pub annotate: bool,

    /// If enabled, swap columns that are out of order and crop to 254x496
    #[arg(long, action)]
    pub colorspad_fix: bool,

    /// Index of binary frame at which to start the preview from (inclusive)
    #[arg(short, long, default_value = None)]
    pub start: Option<isize>,

    /// Index of binary frame at which to end the preview at (exclusive)
    #[arg(short, long, default_value = None)]
    pub end: Option<isize>,

    /// If enabled, invert the SPAD's response (bernouilli process)
    #[arg(long, action)]
    pub invert_response: bool,

    /// If enabled, apply sRGB tonemapping to output
    #[arg(long, action)]
    pub tonemap2srgb: bool,
}

#[derive(ValueEnum, Clone, Copy, Debug)]
pub enum Transform {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipUD,
    FlipLR,
}
