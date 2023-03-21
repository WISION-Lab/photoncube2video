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
use indicatif::{ParallelProgressIterator, ProgressStyle};

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
    npy_path: String,

    /// Output directory to save PNGs in
    #[arg(short, long, default_value = "frames/")]
    out_path: String,

    /// Path of color filter array to use for demosaicing
    #[arg(short, long, default_value = None)]
    cfa_path: Option<String>,

    /// Number of frames to average together
    #[arg(short, long, default_value_t = 256)]
    burst_size: usize,

    /// Apply transformations to each frame
    #[arg(short, long, value_enum, num_args(0..))]
    transform: Vec<Transform>,

    /// If enabled, add bitplane indices to images
    #[arg(short, long, action)]
    annotate: bool,
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

fn array2image(frame: &Array2<u8>) -> RgbImage {
    // Create garyscale image
    let img = GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.as_slice().unwrap().to_vec(),
    )
    .unwrap();

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

fn process_colorspad(frame: &mut RgbImage) -> RgbImage {
    // p = p[..., 2:, :496].to(torch.float32)
    // p[..., :, 252:264] = p[..., :, [260, 261, 262, 263, 256, 257, 258, 259, 252, 253, 254, 255]]

    // Crop dead regions around edges
    imageops::crop(frame, 0, 2, 496, 255).to_image()
}

fn interpolate_where_mask<T>(frame: &Array2<T>, mask: &Array2<bool>, dither: bool) -> Array2<T>
where
    T: From<bool> + Into<f32> + Copy + 'static,
    f32: AsPrimitive<T>,
{
    let mut rng = rand::thread_rng();

    Array2::from_shape_fn(frame.dim(), |(i, j)| {
        if mask[(i, j)] {
            let mut counter = 0.0;
            let mut value = 0.0;

            for ki in [i - 1, i + 1] {
                if let Some(v) = frame.get((ki, j)) {
                    counter += 1.0;
                    value += T::into(*v);
                }
            }
            for kj in [j - 1, j + 1] {
                if let Some(v) = frame.get((i, kj)) {
                    counter += 1.0;
                    value += T::into(*v);
                }
            }

            if dither {
                (rng.gen_range(0.0..1.0) < (value / counter)).into()
            } else {
                ((value / counter).round()).as_()
            }
        } else {
            frame[(i, j)]
        }
    })
}

fn main() -> Result<()> {
    // Parse arguments defined in struct
    let args = Args::parse();

    // Error out if the numpy file does not exist
    if !Path::new(&args.npy_path).exists() {
        eprintln!("Photon cube data not found at {}!", args.npy_path);
        process::exit(exitcode::OSFILE);
    }
    create_dir_all(&args.out_path).ok();

    // Read in the npy file, we expext it to have ndim==3, and of type u8, error if it is not.
    let cube: Array3<u8> = read_npy(args.npy_path).unwrap();
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

    // Create chunked iterator over all data
    let groups = cube.axis_chunks_iter(Axis(0), args.burst_size);
    let pbar_style = ProgressStyle::with_template(
        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
    )
    .unwrap();

    // Functional style loop over all chunks of frames
    // The for-loop body is an anonymous function allowing it to be called
    // by mutyiple threads in parallel.
    groups
        // Make it parallel
        .into_par_iter()
        // Add progress bar
        .progress_count(t / args.burst_size as u64)
        .with_style(pbar_style)
        .enumerate()
        .for_each(|(i, group)| {
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
            let frame = frame.mapv(|x| x as u8);

            let frame = if let Some(mask) = &cfa_mask {
                interpolate_where_mask(&frame, mask, false)
            } else {
                frame
            };

            let mut frame = array2image(&frame);
            let frame = process_colorspad(&mut frame);
            let mut frame = apply_transform(frame, &args.transform);

            if args.annotate {
                annotate(
                    &mut frame,
                    &format!(
                        "{:06}:{:06}",
                        i * args.burst_size,
                        (i + 1) * args.burst_size
                    ),
                );
            }

            // Throw error if we cannot save.
            let path = Path::new(&args.out_path).join(format!("frame{:06}.png", i));
            frame
                .save(&path)
                .unwrap_or_else(|_| panic!("Could not save frame at {}!", &path.display()));
        });

    // Return successful!
    Ok(())
}
