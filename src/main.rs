use rand::Rng;
use rusttype::{Font, Scale};
use std::convert::{From, Into};
use std::fs::create_dir_all;
use std::process;

use image::{imageops, GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::map::map_pixels;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use tempfile::tempdir;

use ndarray::parallel::prelude::*;
use ndarray::Slice;
use num_traits::AsPrimitive;

mod cli;
mod io;
mod utils;

use crate::cli::*;
use crate::io::*;

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
    // Note: We reverse the shift to account for endianness
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
    // Note: if we don't shadow `frame` as a mut, we cannot override it in the loop
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
fn interpolate_where_mask<T>(frame: Array2<T>, mask: &Array2<bool>, dither: bool) -> Array2<T>
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
    // Parse arguments defined in struct
    let args = Args::parse();

    // Load all the neccesary files
    let mut cube: Array3<u8> = try_load_cube(args.input, Some((256, 512)))?;
    let inpaint_mask: Option<Array2<bool>> = try_load_mask(args.inpaint_path)?;
    let cfa_mask: Option<Array2<bool>> = try_load_mask(args.cfa_path)?;
    ensure_ffmpeg(true);

    // Get img path or tempdir, ensure it exists.
    let img_dir = args
        .img_dir
        .unwrap_or(tempdir()?.path().to_str().unwrap().to_owned());
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
    let frames = groups
        // Make it parallel
        .into_par_iter()
        // Add progress bar
        .progress_count(cube.len_of(Axis(0)) as u64 / args.burst_size as u64)
        .with_style(pbar_style.clone())
        .enumerate()
        .map(|(i, group)| {
            let frame = group
                // Unpack every frame in group
                .axis_iter(Axis(0))
                .map(|bitplane| unpack_single(&bitplane, 1))
                // Convert all frames to f32 to avoid overflows when summing
                .map(|bitplane| bitplane.mapv(|x| x as f32))
                // Sum frames together, use reduce not `.sum` as it's not
                // implemented for this type, maybe use `accumulate_axis_inplace`
                .reduce(|acc, e| acc + e)
                .unwrap();

            // Normalize by burst_size, then convert to u8
            let frame = frame / (args.burst_size as f32) * 255.0;
            let mut frame = frame.mapv(|x| x as u8);

            // Apply any frame-level fixes (only for ColorSPAD at the moment)
            if args.colorspad_fix {
                frame = process_colorspad(frame);
            }

            // Demosaic frame by interpolating white pixels
            if let Some(mask) = &cfa_mask {
                frame = interpolate_where_mask(frame, mask, false)
            }

            // Inpaint any hot/dead pixels
            if let Some(mask) = &inpaint_mask {
                frame = interpolate_where_mask(frame, mask, false)
            }

            // Convert to image and rotate/flip as needed
            let frame = array2rgbimage(frame.to_owned());
            let mut frame = apply_transform(frame, &args.transform);

            if args.annotate {
                let text = format!(
                    "{:06}:{:06}",
                    i * args.burst_size + (start_offset as usize),
                    (i + 1) * args.burst_size + (start_offset as usize)
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
    make_video(
        Path::new(&img_dir).join("frame%06d.png").to_str().unwrap(),
        &args.output,
        args.fps,
        num_frames as u64,
        Some(pbar_style),
    );

    Ok(())
}
