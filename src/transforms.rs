use std::convert::{From, Into};
use std::process;

use rand::Rng;
use rusttype::{Font, Scale};

use image::{imageops, GrayImage, Rgb, RgbImage};
use imageproc::drawing::{draw_text_mut, text_size};
use imageproc::map::map_pixels;

use ndarray::Slice;
use ndarray_stats::{interpolate::Linear, QuantileExt};
use noisy_float::types::n64;
use num_traits::AsPrimitive;

use crate::cli::*;
use crate::io::*;

pub fn unpack_single(bitplane: &ArrayView2<'_, u8>, axis: usize) -> Array2<u8> {
    // This pattern is cluncky, is there an easier one that's as readable as unpacking?
    let [h_orig, w_orig, ..] = bitplane.shape() else {
        process::exit(exitcode::DATAERR)
    };
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

pub fn array2grayimage(frame: Array2<u8>) -> GrayImage {
    GrayImage::from_raw(
        frame.len_of(Axis(1)) as u32,
        frame.len_of(Axis(0)) as u32,
        frame.into_raw_vec(),
    )
    .unwrap()
}

pub fn array2rgbimage(frame: Array2<u8>) -> RgbImage {
    // Create grayscale image
    let img = array2grayimage(frame);

    // Convert it to rgb by duplicating each pixel
    map_pixels(&img, |_x, _y, p| Rgb([p[0], p[0], p[0]]))
}

pub fn annotate(frame: &mut RgbImage, text: &str) {
    let font = Vec::from(include_bytes!("DejaVuSans.ttf") as &[u8]);
    let font = Font::try_from_vec(font).unwrap();
    let scale = Scale { x: 25.0, y: 25.0 };

    draw_text_mut(frame, Rgb([252, 186, 3]), 5, 5, scale, &font, text);
    text_size(scale, &font, text);
}

pub fn apply_transform(frame: RgbImage, transform: &[Transform]) -> RgbImage {
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

pub fn linearrgb_to_srgb(mut frame: Array2<f32>) -> Array2<f32> {
    // https://github.com/blender/blender/blob/master/source/blender/blenlib/intern/math_color.c
    frame.par_mapv_inplace(|x| {
        if x < 0.0031308 {
            if x < 0.0 {
                return 0.0;
            } else {
                return x * 12.92;
            }
        }
        1.055 * x.powf(1.0 / 2.4) - 0.055
    });
    frame
}

pub fn binary_avg_to_rgb(
    mut frame: Array2<f32>,
    factor: f32,
    quantile: Option<f32>,
) -> Array2<f32> {
    // Invert the process by which binary frames are simulated. The result can be either
    // linear RGB values or sRGB values depending on how the binary frames were constructed.
    // Assuming each binary patch was created by a Bernoulli process with p=1-exp(-factor*rgb),
    // then the average of binary frames tends to p. We can therefore recover the original rgb
    // values as -log(1-bin)/factor.

    frame.par_mapv_inplace(|v| -(1.0 - v).clamp(1e-6, 1.0).ln() / factor);

    if let Some(quantile_val) = quantile {
        let val = Array::from_iter(frame.iter().cloned())
            .quantile_axis_skipnan_mut(Axis(0), n64(quantile_val as f64), &Linear)
            .unwrap()
            .mean()
            .unwrap();
        frame.par_mapv_inplace(|v| (v / val).clamp(0.0, 1.0));
    } else {
        frame.par_mapv_inplace(|v| v.clamp(0.0, 1.0));
    };

    frame
}

pub fn process_colorspad(mut frame: Array2<u8>) -> Array2<u8> {
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
pub fn interpolate_where_mask<T>(
    frame: Array2<T>,
    mask: &Array2<bool>,
    dither: bool,
) -> Result<Array2<T>>
where
    T: Into<f32> + Copy + 'static,
    f32: AsPrimitive<T>,
    bool: AsPrimitive<T>,
{
    let (h, w) = frame.dim();
    let (mask_h, mask_w) = mask.dim();

    if (mask_h < h) || (mask_w < w) {
        return Err(anyhow!(
            "Frame has size {:?} but interpolation mask has size {:?}",
            frame.dim(),
            mask.dim()
        ));
    }

    let mut rng = rand::thread_rng();
    Ok(Array2::from_shape_fn((h, w), |(i, j)| {
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
    }))
}
