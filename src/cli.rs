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
    #[arg(long, default_value = None)]
    pub inpaint_path: Option<String>,

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

    /// If enabled, swap columns and crop to 254x496
    #[arg(long, action)]
    pub colorspad_fix: bool,
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
