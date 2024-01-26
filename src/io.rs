use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, Result};
use image::{
    imageops::{resize, FilterType},
    GrayImage,
};
use memmap2::Mmap;
use ndarray::{Array, Array3, ArrayView3, Axis, Slice};
use ndarray_npy::{ViewNpyError, ViewNpyExt};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use image::io::Reader as ImageReader;
use ndarray::prelude::*;
use ndarray_npy::read_npy;
use nshare::ToNdarray2;

use crate::{
    cli::Transform,
    transforms::{apply_transform, array2_to_grayimage, process_colorspad, unpack_single},
    utils::sorted_glob,
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct PhotonCube<'a> {
    path: &'a str,
    bit_depth: u32,
    _storage: PhotonCubeStorage,
}

// These are needed to keep the underlying object's data in scope
// otherwise we get a use-after-free error.
// We use an enum here as either an array OR a memap object is needed.
#[derive(Debug)]
enum PhotonCubeStorage {
    ArrayStorage(Array3<u8>),
    MmapStorage(Mmap),
}

impl<'a> PhotonCube<'a> {
    pub fn view(&self) -> Result<ArrayView3<u8>, ViewNpyError> {
        match &self._storage {
            PhotonCubeStorage::ArrayStorage(arr) => Ok(arr.view()),
            PhotonCubeStorage::MmapStorage(mmap) => ArrayView3::<u8>::view_npy(mmap),
        }
    }

    pub fn open(path_str: &'a str) -> Result<Self> {
        let path = Path::new(path_str);

        if !path.exists() {
            // This should probably be a specific IO error?
            Err(anyhow!("File not found at {}!", path_str))
        } else if path.is_dir() {
            let (h, w) = (256, 512);
            let paths = sorted_glob(path, "**/*.bin")?;

            if paths.is_empty() {
                return Err(anyhow!("No .bin files found in {}!", path_str));
            }

            let mut buffer = Vec::new();

            for p in paths {
                let mut f = File::open(p)?;
                f.read_to_end(&mut buffer)?;
            }

            let t = buffer.len() / (h * w / 8);
            let arr = Array::from_vec(buffer)
                .into_shape((t, h, w / 8))?
                .mapv(|v| v.reverse_bits());

            Ok(Self {
                path: path_str,
                bit_depth: 1,
                _storage: PhotonCubeStorage::ArrayStorage(arr),
            })
        } else {
            let ext = path.extension().unwrap().to_ascii_lowercase();

            if ext != "npy" {
                // TODO: This should probably be a specific IO error?
                return Err(anyhow!(
                    "Expected numpy array with `.npy` extension, got {:?}.",
                    ext
                ));
            }

            let file = File::open(path_str)?;
            let mmap = unsafe { Mmap::map(&file)? };

            Ok(Self {
                path: path_str,
                bit_depth: 1,
                _storage: PhotonCubeStorage::MmapStorage(mmap),
            })
        }
    }

    pub fn load(
        &self,
        start: isize,
        stop: isize,
        step: usize,
        downscale: f32,
        transform: &[Transform],
    ) -> Result<Vec<GrayImage>> {
        let cube_view = self.view()?;
        let cube_view = cube_view.slice_axis(Axis(0), Slice::new(start, Some(stop), 1));

        // Create parallel iterator over all chunks of frames and process them
        let virtual_exposures: Vec<_> = cube_view
            .axis_chunks_iter(Axis(0), step)
            // Make it parallel
            .into_par_iter()
            .map(|group| {
                let (num_frames, _, _) = group.dim();
                let frame = group
                    // Iterate over all bitplanes in group
                    .axis_iter(Axis(0))
                    // Unpack every frame in group as a f32 array
                    .map(|bitplane| unpack_single::<f32>(&bitplane, 1).unwrap())
                    // Sum frames together (.sum not implemented for this type)
                    .reduce(|acc, e| acc + e)
                    .unwrap()
                    // Compute mean values and save as uint8's
                    .mapv(|v| (v / (num_frames as f32) * 255.0).round() as u8);

                // Convert to image and resize/transform
                let mut img = array2_to_grayimage(process_colorspad(frame));

                if downscale != 1.0 {
                    img = resize(
                        &img,
                        (img.width() as f32 / downscale).round() as u32,
                        (img.height() as f32 / downscale).round() as u32,
                        FilterType::CatmullRom,
                    );
                }

                apply_transform(img, transform)
            })
            // Force iterator to run to completion to get correct ordering
            .collect();

        Ok(virtual_exposures)
    }
}

/// Load either a 2D NPY file or an intensity-only image file as an array of booleans.
/// Note: For the image, any pure white pixels are false, all others are true.
///       This is contrary to what you might expect but enables us to load in the
///       colorSPAD's cfa array and have a mask representing the colored pixels.
pub fn try_load_mask(path: Option<String>) -> Result<Option<Array2<bool>>> {
    if path.is_none() {
        return Ok(None);
    }

    let path_str = path.unwrap();
    let path = Path::new(&path_str);
    let ext = path.extension().unwrap().to_ascii_lowercase();

    if !path.exists() {
        // This should probably be a specific IO error?
        Err(anyhow!("File not found at {}!", path_str))
    } else if ext == "npy" || ext == "npz" {
        let arr: Array2<bool> = read_npy(path_str)?;
        Ok(Some(arr))
    } else {
        let arr = ImageReader::open(path_str)?
            .decode()?
            .into_luma8()
            .into_ndarray2()
            .mapv(|v| v != 0);
        Ok(Some(arr))
    }
}
