use rusttype::{Font, Scale};
use std::path::Path;
use std::process;

use clap::Parser;
use image::{GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::map::map_pixels;
use indicatif::ParallelProgressIterator;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::Slice;
use ndarray_npy::read_npy;

/// Convert a photon cube (npy file) to a video preview (mp4) by naively averaging frames.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to photon cube (npy file)
    #[arg(short, long)]
    npy_path: String,

    /// Number of frames to average together
    #[arg(short, long, default_value_t = 256)]
    burst_size: usize,

    /// If enabled, add bitplane indices to images
    #[arg(short, long, action)]
    annotate: bool,
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

fn main() {
    // Parse arguments defined in struct
    let args = Args::parse();

    // Error out if the numpy file does not exist
    if !Path::new(&args.npy_path).exists() {
        eprintln!("Photon cube data not found at {}!", args.npy_path);
        process::exit(exitcode::OSFILE);
    }

    // Read in the npy file, we expext it to have ndim==3, and of type u8, error if it is not.
    let cube: Array3<u8> = read_npy(args.npy_path).unwrap();
    let t = cube.len_of(Axis(0)) as u64;

    // Create chunked iterator over all data
    let groups = cube.axis_chunks_iter(Axis(0), args.burst_size);

    // Functional style loop over all chunks of frames
    // The for-loop body is an anonymous function allowing it to be called
    // by mutyiple threads in parallel.
    groups
        .into_par_iter()
        .enumerate()
        .progress_count(t / args.burst_size as u64)
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
            let mut frame = array2image(&frame);

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

            frame.save(format!("frames/frame{:06}.png", i)).ok();
        });

    // Return successful!
    process::exit(exitcode::OK)
}
