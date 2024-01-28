use std::{
    fs::{File, OpenOptions},
    io::Read,
    ops::BitOr,
    path::Path,
};

use anyhow::{anyhow, Error, Result};
use image::{io::Reader as ImageReader, Rgb};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::multizip;
use memmap2::{Mmap, MmapMut};
use ndarray::{prelude::*, Array, Array3, ArrayView3, Axis, Slice};
use ndarray_npy::{read_npy, write_zeroed_npy, ViewMutNpyExt, ViewNpyError, ViewNpyExt};
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
type PhotonCubeView<'a> = ArrayView3<'a, u8>;

trait VirtualExposure {
    /// Create parallel iterator over all virtual exposures (bitplane averages of depth `step`)
    fn par_virtual_exposures(
        &self,
        step: usize,
        drop_last: bool,
    ) -> impl IndexedParallelIterator<Item = Array2<f32>>;

    /// Create a sequential iterator over all virtual exposures (bitplane averages of depth `step`)
    fn virtual_exposures(&self, step: usize, drop_last: bool) -> impl Iterator<Item = Array2<f32>>;
}

impl<'a> VirtualExposure for PhotonCubeView<'a> {
    fn par_virtual_exposures(
        &self,
        step: usize,
        drop_last: bool,
    ) -> impl IndexedParallelIterator<Item = Array2<f32>> {
        let num_frames = if drop_last {
            self.len_of(Axis(0)) / step
        } else {
            (self.len_of(Axis(0)) as f32 / step as f32).ceil() as usize
        };

        self.axis_chunks_iter(Axis(0), step)
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

    fn virtual_exposures(&self, step: usize, drop_last: bool) -> impl Iterator<Item = Array2<f32>> {
        let num_frames = if drop_last {
            self.len_of(Axis(0)) / step
        } else {
            (self.len_of(Axis(0)) as f32 / step as f32).ceil() as usize
        };

        self.axis_chunks_iter(Axis(0), step)
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
}

impl<'a> PhotonCube<'a> {
    /// Open a photoncube from a memmapped `.npy`` file.
    /// Note: Loading from a directory of .bin files has been deprecated
    /// as it is non-trivial to do when using the full-array and requires
    /// entirely loading the photoncube into memory. Convert to `.npy` first.
    pub fn open(path_str: &'a str) -> Result<Self> {
        let path = Path::new(path_str);
        let ext = path.extension().unwrap().to_ascii_lowercase();

        if !path.exists() || !path.is_file() || ext != "npy" {
            // This should probably be a specific IO error?
            return Err(anyhow!("No `.npy` file found at {}!", path_str));
        }

        let file = File::open(path_str)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            path: path_str,
            cfa_mask: None,
            inpaint_mask: None,
            start: 0,
            end: None,
            step: None,
            _storage: PhotonCubeStorage::MmapStorage(mmap),
            _slice: None,
        })
    }

    /// Convert a photon cube stored as a set of `.bin` files to a `.npy` one. This is done
    /// in a streaming manner and ovoids loading all the data to memory.
    /// The `.npy` format enables memory mapping the photon cube and is usually faster.
    /// Note: Function assumes either 256x512 of 512x512 frames that are bitpacked along width dim.
    pub fn convert_to_npy(
        src: &str,
        dst: &str,
        is_full_array: bool,
        message: Option<&str>,
    ) -> Result<()> {
        let path = Path::new(src);

        if !path.exists() || !path.is_dir() {
            // This should probably be a specific IO error?
            Err(anyhow!("Directory of `.bin` files not found at {}!", src))
        } else {
            let paths = sorted_glob(path, "**/*.bin")?;
            if paths.is_empty() {
                return Err(anyhow!("No .bin files found in {}!", src));
            }

            // Create a (sparse if supported) file of zeroed data.
            // Estimate shape of final array, one bin file is 512 raw
            // half-frames, or 256 full array frames
            let batch_size = if is_full_array { 256 } else { 512 };
            let (h, w) = if is_full_array {
                (512, 512 / 8)
            } else {
                (256, 512 / 8)
            };
            let t = paths.len() * batch_size;
            let file = File::create(dst)?;
            write_zeroed_npy::<u8, _>(&file, (t, h, w))?;

            // Memory-map the file and create the mutable view.
            let file = OpenOptions::new().read(true).write(true).open(dst)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            let mut view_mut = ArrayViewMut3::<u8>::view_mut_npy(&mut mmap)?;

            // Modify the array, and write data to it in a streaming manner.
            // TODO: Make this parallel? Not sure any speedup is possible...
            let pbar = if let Some(msg) = message {
                ProgressBar::new(paths.len() as u64)
                    .with_style(ProgressStyle::with_template(
                        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                    )?)
                    .with_message(msg.to_owned())
            } else {
                ProgressBar::hidden()
            };

            // Read all files and assign to Memmapped npy
            (paths, view_mut.axis_chunks_iter_mut(Axis(0), batch_size))
                .into_par_iter()
                .progress_with(pbar)
                .try_for_each(|(p, mut chunk)| {
                    let mut buffer = Vec::new();
                    let mut f = File::open(p)?;
                    f.read_to_end(&mut buffer)?;

                    if is_full_array {
                        // Raw data is saved as 1/4th of the top array (h, w/4) then a quarter of the
                        // bottom array, but flipped up/down, and repeat. We read out all data in a
                        // buffer that's (h/2, w*2), meaning there's 8 interleaved zones:
                        // Top (1/4), Flipped Btm (1/4), Top (2/4), Flipped Btm (2/4), etc...
                        let flat_data = Array::from_iter(buffer.iter().map(|v| v.reverse_bits()));
                        let data = flat_data.to_shape((batch_size, h / 2, w * 2))?.into_owned();

                        // Iterate over array quarters and assign as needed.
                        // Data: (256, 1024) -> src_chunk: (256, 256) -> top/btm_src: (256, 128)
                        // Chunk: (512, 512) -> dst_chunk: (512, 128) -> top_btm_dst: (256, 128)
                        multizip((
                            data.axis_chunks_iter(Axis(2), w / 2),
                            chunk.axis_chunks_iter_mut(Axis(2), w / 4),
                        ))
                        .for_each(|(src_chunk, mut dst_chunk)| {
                            let (top_src, btm_src) = src_chunk.split_at(Axis(2), w / 4);
                            let (mut top_dst, mut btm_dst) = dst_chunk.multi_slice_mut((
                                s![.., ..(h / 2), ..],
                                s![.., (h/2)..;-1, ..], // We flip the btm array here!
                            ));

                            top_dst.assign(&top_src);
                            btm_dst.assign(&btm_src);
                        });
                    } else {
                        chunk.assign(
                            &Array::from_iter(buffer.iter().map(|v| v.reverse_bits()))
                                .to_shape((batch_size, h, w))?,
                        );
                    }

                    Ok::<(), Error>(())
                })?;
            Ok(())
        }
    }

    /// Access the underlying data as a ArrayView3.
    pub fn view(&self) -> Result<PhotonCubeView, ViewNpyError> {
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
        let virtual_exps = slice.par_virtual_exposures(self.step.unwrap(), true);

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
