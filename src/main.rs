use rand::Rng;
use rusttype::{Font, Scale};
use std::convert::{From, Into};
use std::fs::create_dir_all;
use std::path::Path;
use std::process;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use image::io::Reader as ImageReader;
use image::{imageops, GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::map::map_pixels;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use tempfile::tempdir;

use ffmpeg_sidecar::command::{ffmpeg_is_installed, FfmpegCommand};
use ffmpeg_sidecar::paths::sidecar_dir;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_npy::read_npy;
use nshare::ToNdarray3;
use num_traits::AsPrimitive;

/// Convert a photon cube (npy file) to a video preview (mp4) by naively averaging frames.
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to photon cube (npy file)
    #[arg(short, long)]
    input: String,

    /// Path of output video
    #[arg(short, long, default_value = "out.mp4")]
    output: String,

    /// Output directory to save PNGs in
    #[arg(long)]
    img_dir: Option<String>,

    /// Path of color filter array to use for demosaicing
    #[arg(long, default_value = None)]
    cfa_path: Option<String>,

    /// Path of inpainting mask to use for filtering out dead/hot pixels
    #[arg(long, default_value = None)]
    inpaint_path: Option<String>,

    /// Number of frames to average together
    #[arg(short, long, default_value_t = 256)]
    burst_size: usize,

    /// Frame rate of resulting video
    #[arg(long, default_value_t = 25)]
    fps: usize,

    /// Apply transformations to each frame
    #[arg(short, long, value_enum, num_args(0..))]
    transform: Vec<Transform>,

    /// If enabled, add bitplane indices to images
    #[arg(short, long, action)]
    annotate: bool,

    /// If enabled, swap columns and crop to 254x496
    #[arg(long, action)]
    colorspad_fix: bool,
}

// #[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
#[derive(ValueEnum, Clone, Copy, Debug)]
enum Transform {
    Identity,
    Rot90,
    Rot180,
    Rot270,
    FlipUD,
    FlipLR,
}

#[allow(dead_code)]
fn print_type_of<T>(_: &T) {
    println!("{}", std::any::type_name::<T>())
}

fn unpack_single(bitplane: &ArrayView2<'_, u8>, axis: usize) -> Array2<u8> {
    // This pattern is cluncky, is there an easier one that's as readable as unpacking?
    let [h_orig, w_orig, ..] = bitplane.shape() else {process::exit(exitcode::DATAERR)};
    let h = if axis == 0 { h_orig * 8 } else { *h_orig };
    let w = if axis == 1 { w_orig * 8 } else { *w_orig };

    // Allocate full sized frame
    let mut unpacked_bitplane = Array2::<u8>::zeros((h, w));

    // Iterate through slices with stride 8 of the full array and fill it up
    // Note: We reverse the shift to account for LSB/MSB
    for shift in 0..8 {
        let ishift = 7 - shift;
        let mut slice =
            unpacked_bitplane.slice_axis_mut(Axis(axis), Slice::from(shift..).step_by(8));
        slice.assign(&((bitplane & (1 << ishift)) >> ishift));
    }

    unpacked_bitplane
}

fn array2grayimage(frame: Array2<u8>) -> GrayImage {
    GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec(),
    )
    .unwrap()
}

fn array2rgbimage(frame: Array2<u8>) -> RgbImage {
    // Create grayscale image
    let img = array2grayimage(frame);

    // Convert it to rgb by duplicating each pixel
    map_pixels(&img, |_x, _y, p| Rgb([p[0], p[0], p[0]]))
}

fn annotate(frame: &mut RgbImage, text: &str) {
    let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();
    let scale = Scale { x: 20.0, y: 20.0 };

    draw_text_mut(frame, Rgb([252, 186, 3]), 5, 5, scale, &font, text);
    text_size(scale, &font, text);
}

fn apply_transform(frame: RgbImage, transform: &[Transform]) -> RgbImage {
    // Not if we don't shadow `frame` as a mut, we cannot override it in the loop
    let mut frame = frame;

    for t in transform.iter() {
        frame = match t {
            Transform::Identity => continue,
            Transform::Rot90 => imageops::rotate90(&frame),
            Transform::Rot180 => imageops::rotate180(&frame),
            Transform::Rot270 => imageops::rotate270(&frame),
            Transform::FlipUD => imageops::flip_vertical(&frame),
            Transform::FlipLR => imageops::flip_horizontal(&frame),
        };
    }
    frame
}

fn process_colorspad(mut frame: Array2<u8>) -> Array2<u8> {
    // Crop dead regions around edges
    let mut crop = frame.slice_mut(s![2.., ..496]);

    // Swap rows (Can we do this inplace?)
    let (mut slice_a, mut slice_b) = crop.multi_slice_mut((s![.., 252..256], s![.., 260..264]));
    let tmp_slice = slice_a.to_owned();
    slice_a.assign(&slice_b);
    slice_b.assign(&tmp_slice);

    // This clones the array, can we avoid this somehow??
    crop.to_owned()
}

// Note: The use of generics here is heavy handed, we only really want this function
//       to work with T=u8 or maybe T=f32/i32. Is there a better way? I.e generic over primitives?
fn interpolate_where_mask<T>(frame: &Array2<T>, mask: &Array2<bool>, dither: bool) -> Array2<T>
where
    T: Into<f32> + Copy + 'static,
    f32: AsPrimitive<T>,
    bool: AsPrimitive<T>,
{
    let mut rng = rand::thread_rng();
    let (h, w) = frame.dim();

    Array2::from_shape_fn((h, w), |(i, j)| {
        if mask[(i, j)] {
            let mut counter = 0.0;
            let mut value = 0.0;

            for ki in [(i as isize) - 1, (i as isize) + 1] {
                if (ki >= 0) && (ki < h as isize) {
                    counter += 1.0;
                    value += T::into(frame[(ki as usize, j)]);
                }
            }
            for kj in [(j as isize) - 1, (j as isize) + 1] {
                if (kj >= 0) && (kj < w as isize) {
                    counter += 1.0;
                    value += T::into(frame[(i, kj as usize)]);
                }
            }

            if dither {
                (rng.gen_range(0.0..1.0) < (value / counter)).as_()
            } else {
                ((value / counter).round()).as_()
            }
        } else {
            frame[(i, j)]
        }
    })
}

fn main() -> Result<()> {
    if !ffmpeg_is_installed() {
        println!(
            "No ffmpeg installation found, downloading one to {}...",
            &sidecar_dir().unwrap().display()
        );
        ffmpeg_sidecar::download::auto_download().unwrap();
    }

    // Parse arguments defined in struct
    let args = Args::parse();

    // Error out if the numpy file does not exist
    if !Path::new(&args.input).exists() {
        eprintln!("Photon cube data not found at {}!", args.input);
        process::exit(exitcode::OSFILE);
    }

    // Get img path or tempdir, ensure it exists.
    let img_dir = args
        .img_dir
        .unwrap_or(tempdir()?.path().to_str().unwrap().to_owned());
    create_dir_all(&img_dir).ok();

    // Read in the npy file, we expext it to have ndim==3, and of type u8, error if it is not.
    let cube: Array3<u8> = read_npy(args.input).unwrap();
    let t = cube.len_of(Axis(0)) as u64;

    // It would be much nice to be able to use Option.map here but then
    // we cannot use the `?` operator as it would be inside a closure
    // which does not return a Result type...
    let cfa_mask: Option<Array2<bool>> = match args.cfa_path {
        Some(cfa_path) => {
            let cfa_mask: Array2<bool> = ImageReader::open(cfa_path)?
                .decode()?
                .into_rgb8()
                .into_ndarray3()
                .mapv(|v| v == 0)
                .map_axis(Axis(0), |p| {
                    p.to_vec().into_iter().reduce(|a, b| a | b).unwrap()
                });
            Some(cfa_mask)
        }
        None => None,
    };
    let inpaint_mask: Option<Array2<bool>> = args.inpaint_path.map(|p| read_npy(p).unwrap());

    // // Create chunked iterator over all data
    let groups = cube.axis_chunks_iter(Axis(0), args.burst_size);
    let pbar_style = ProgressStyle::with_template(
        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
    )
    .unwrap();

    // Create parallel iterator over all chunks of frames and process them
    let frames = groups
        // Make it parallel
        .into_par_iter()
        // Add progress bar
        .progress_count(t / args.burst_size as u64)
        .with_style(pbar_style.clone())
        .enumerate()
        .map(|(i, group)| {
            let frame = group
                // Unpack every frame in group
                .axis_iter(Axis(0))
                .map(|bitplane| unpack_single(&bitplane, 1))
                // Convert all frames to i32 to avoid overflows when summing
                .map(|bitplane| bitplane.mapv(|x| x as i32))
                // Sum frames together, use reduce not `.sum` as it's not
                // implemented for this type.
                .reduce(|acc, e| acc + e)
                .unwrap();

            // Convert to float and normalize by burst_size, then convert to img
            let frame = frame.mapv(|x| x as f32) / (args.burst_size as f32) * 255.0;
            let mut frame = frame.mapv(|x| x as u8);

            // Apply any frame-level fixes (only for ColorSPAD at the moment)
            if args.colorspad_fix {
                frame = process_colorspad(frame);
            }

            // Demosaic frame by interpolating white pixels
            let frame = if let Some(mask) = &cfa_mask {
                interpolate_where_mask(&frame, mask, false)
            } else {
                frame
            };

            // Inpaint any hot/dead pixels
            let frame = if let Some(mask) = &inpaint_mask {
                interpolate_where_mask(&frame, mask, false)
            } else {
                frame
            };

            // Convert to image and rotate/flip as needed
            let frame = array2rgbimage(frame);
            let mut frame = apply_transform(frame, &args.transform);

            if args.annotate {
                let text = format!(
                    "{:06}:{:06}",
                    i * args.burst_size,
                    (i + 1) * args.burst_size
                );
                annotate(&mut frame, &text);
            }

            // Return both the frame and it's index so we can later save
            (i, frame)
        });

    // Save and count frames, this consumes/runs the above iterator
    let num_frames = frames
        .map(|(i, frame)| {
            // Throw error if we cannot save.
            let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
            frame
                .save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
        })
        .count();

    // Finally, make a call to ffmpeg to assemble to video
    let pbar = ProgressBar::new(num_frames as u64).with_style(pbar_style);
    let cmd = format!(
        concat!(
            "-framerate {fps} -f image2 -i {pattern} ",
            "-y -vcodec libx264 -crf 22 -pix_fmt yuv420p {outfile}"
        ),
        fps = args.fps,
        pattern = Path::new(&img_dir).join("frame%06d.png").display(),
        outfile = args.output
    );

    let mut ffmpeg_runner = FfmpegCommand::new().args(cmd.split(' ')).spawn().unwrap();
    ffmpeg_runner
        .iter()
        .unwrap()
        .filter_progress()
        .for_each(|progress| pbar.set_position(progress.frame as u64));
    pbar.finish_and_clear();

    // Return successful!
    Ok(())
}
