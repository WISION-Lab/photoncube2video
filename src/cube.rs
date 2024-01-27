use std::{fs::File, io::Read, ops::BitOr, path::Path};

use anyhow::{anyhow, Result};
use image::{io::Reader as ImageReader, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use memmap2::Mmap;
use ndarray::{prelude::*, Array, Array3, ArrayView3, Axis, Slice};
use ndarray_npy::{read_npy, ViewNpyError, ViewNpyExt};
use nshare::ToNdarray2;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::{
    cli::Transform,
    transforms::{
        annotate, apply_transform, array2_to_grayimage, binary_avg_to_rgb, gray_to_rgbimage,
        interpolate_where_mask, linearrgb_to_srgb, process_colorspad, unpack_single,
    },
    utils::sorted_glob,
};

#[allow(dead_code)]
#[derive(Debug)]
pub struct PhotonCube<'a> {
    path: &'a str,
    bit_depth: u32,
    cfa_mask: Option<Array2<bool>>,
    inpaint_mask: Option<Array2<bool>>,
    start: isize,
    end: Option<isize>,
    step: Option<usize>,
    _storage: PhotonCubeStorage,
    _slice: Option<ArrayView3<'a, u8>>,
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
    /// Open a photoncube from a file, either a memmapped .npy file or a directory
    /// of .bin files is accepted, with the later being entirely loaded into memory.
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
                cfa_mask: None,
                inpaint_mask: None,
                start: 0,
                end: None,
                step: None,
                _storage: PhotonCubeStorage::ArrayStorage(arr),
                _slice: None,
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
                cfa_mask: None,
                inpaint_mask: None,
                start: 0,
                end: None,
                step: None,
                _storage: PhotonCubeStorage::MmapStorage(mmap),
                _slice: None,
            })
        }
    }

    /// Access the underlying data as a ArrayView3.
    pub fn view(&self) -> Result<ArrayView3<u8>, ViewNpyError> {
        match &self._storage {
            PhotonCubeStorage::ArrayStorage(arr) => Ok(arr.view()),
            PhotonCubeStorage::MmapStorage(mmap) => ArrayView3::<u8>::view_npy(mmap),
        }
    }

    /// Set the range of the photoncube, this is used for slicing and frame preview
    /// where `step` will be the number of frames to average together.
    pub fn set_range(&mut self, start: isize, end: Option<isize>, step: usize) {
        self.start = start;
        self.end = end;
        self.step = Some(step);
    }

    /// Total number of bitplanes in cube (independent of `set_range`).
    pub fn len(&self) -> usize {
        self.view().expect("Cannot load photoncube").len_of(Axis(0))
    }

    /// Load either a 2D NPY file or an intensity-only image file as an array of booleans.
    /// Note: For the image, any pure white pixels are false, all others are true.
    ///       This is contrary to what you might expect but enables us to load in the
    ///       colorSPAD's cfa array and have a mask representing the colored pixels.
    pub fn _try_load_mask(path: &str) -> Result<Array2<bool>> {
        let path_obj = Path::new(&path);
        let ext = path_obj.extension().unwrap().to_ascii_lowercase();

        if !path_obj.exists() {
            // This should probably be a specific IO error?
            Err(anyhow!("File not found at {}!", path))
        } else if ext == "npy" || ext == "npz" {
            let arr: Array2<bool> = read_npy(path)?;
            Ok(arr)
        } else {
            let arr = ImageReader::open(path)?
                .decode()?
                .into_luma8()
                .into_ndarray2()
                .mapv(|v| v != 0);
            Ok(arr)
        }
    }

    /// Load the color-filter array associated with the photoncube, if applicable.
    /// Will be used for processing if loaded.
    pub fn load_cfa(&mut self, path: &str) -> Result<()> {
        self.cfa_mask = Some(Self::_try_load_mask(path)?);
        Ok(())
    }

    /// Load an inpainting mask. This can be called multiple times, the masks will
    /// simply be ORed together (interpolate where mask is true/white).
    pub fn load_mask(&mut self, path: &str) -> Result<()> {
        let mut new_mask = Self::_try_load_mask(path)?;

        if let Some(mask) = &self.inpaint_mask {
            new_mask = mask.bitor(new_mask);
        }
        self.inpaint_mask = Some(new_mask);
        Ok(())
    }

    /// Return function to process a single virtual exposure, depends on any masks (cfa/inpaint).
    pub fn process_single(
        &'a self,
        invert_response: bool,
        tonemap2srgb: bool,
        colorspad_fix: bool,
    ) -> impl Fn(Array2<f32>) -> Result<Array2<f32>> + 'a {
        move |mut frame| {
            // Invert SPAD response
            if invert_response {
                frame = binary_avg_to_rgb(frame, 1.0, None);
            }

            // Apply sRGB tonemapping
            if tonemap2srgb {
                frame = linearrgb_to_srgb(frame);
            }

            // Apply any frame-level fixes (only for ColorSPAD at the moment)
            if colorspad_fix {
                frame = process_colorspad(frame);
            }

            // Demosaic frame by interpolating white pixels
            if let Some(mask) = &self.cfa_mask {
                frame = interpolate_where_mask(frame, mask, false)?;
            }

            // Inpaint any hot/dead pixels
            if let Some(mask) = &self.inpaint_mask {
                frame = interpolate_where_mask(frame, mask, false)?;
            }
            Ok(frame)
        }
    }

    /// Create parallel iterator over all virtual exposures (bitplane averages of depth `step`)
    pub fn par_virtual_exposures<'b>(
        &'b self,
        view: &'b ArrayView3<u8>,
        drop_last: bool,
    ) -> impl IndexedParallelIterator<Item = Array2<f32>> + 'b {
        let num_frames = if drop_last {
            self.len() / self.step.unwrap()
        } else {
            (self.len() as f32 / self.step.unwrap() as f32).ceil() as usize
        };

        view.axis_chunks_iter(
            Axis(0),
            self.step
                .expect("Step must be set before virtual exposures can be created!"),
        )
        // Make it parallel
        .into_par_iter()
        .map(|group| {
            let (num_frames, _, _) = group.dim();
            let mut frame = group
                // Iterate over all bitplanes in group
                .axis_iter(Axis(0))
                // Unpack every frame in group as a f32 array
                .map(|bitplane| unpack_single::<f32>(&bitplane, 1).unwrap())
                // Sum frames together (.sum not implemented for this type)
                .reduce(|acc, e| acc + e)
                .unwrap();
            // Compute mean values
            frame.mapv_inplace(|v| v / (num_frames as f32));
            frame
        })
        .take(num_frames)
    }

    /// Save all virtual exposures to a folder.
    pub fn save_images<F>(
        &'a self,
        img_dir: &str,
        process_fn: Option<F>,
        annotate_frames: bool,
        transform: &[Transform],
        message: Option<&str>,
    ) -> Result<isize>
    where
        F: Fn(Array2<f32>) -> Result<Array2<f32>> + Send + Sync,
    {
        if self.step.is_none() {
            return Err(anyhow!(
                "Step must be set before virtual exposures can be created!"
            ));
        }

        // Create virtual exposures iterator over all data
        let view = self.view()?;
        let slice = view.slice_axis(Axis(0), Slice::new(self.start, self.end, 1));
        let virtual_exps = self.par_virtual_exposures(&slice, true);

        // Conditionally setup a pbar
        let pbar = if let Some(msg) = message {
            ProgressBar::new((slice.len_of(Axis(0)) / self.step.unwrap()) as u64)
                .with_style(ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )?)
                .with_message(msg.to_owned())
        } else {
            ProgressBar::hidden()
        };

        // Create parallel iterator over all chunks of frames and process them
        let num_frames = virtual_exps
            .enumerate()
            // Process data, save and count frames, this consumes/runs the iterator to completion
            // We use a try_fold followed by a try_reduce as a sort of try_map here.
            .try_fold(
                || 0,
                |count, (i, mut frame)| {
                    // Perform all Array2<f32> -> Array2<f32> preprocessing
                    if let Some(preprocess) = &process_fn {
                        frame = preprocess(frame)?;
                    }

                    // Convert to uint8 in [0, 255] range
                    let frame = frame.mapv(|x| (x * 255.0) as u8);

                    // Convert to image and rotate/flip as needed
                    let frame = array2_to_grayimage(frame.to_owned());
                    let frame = apply_transform(frame, transform);
                    let mut frame = gray_to_rgbimage(&frame);

                    if annotate_frames {
                        let start_idx = i * self.step.unwrap() + (self.start as usize);
                        let text =
                            format!("{:06}:{:06}", start_idx, start_idx + self.step.unwrap());
                        annotate(&mut frame, &text, Rgb([252, 186, 3]));
                    }

                    // Throw error if we cannot save, and stop all processing.
                    let path = Path::new(&img_dir).join(format!("frame{:06}.png", i));
                    frame.save(&path)?;
                    pbar.inc(1);

                    Ok::<isize, anyhow::Error>(count + 1)
                },
            )
            .try_reduce(|| 0, |acc, x| Ok(acc + x))?;
        Ok(num_frames)
    }
}
