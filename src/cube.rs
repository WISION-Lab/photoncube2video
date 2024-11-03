use std::{
    collections::HashMap,
    fs::{create_dir_all, File, OpenOptions},
    io::Read,
    ops::BitOr,
    path::{Path, PathBuf},
    str::FromStr,
};

use anyhow::{anyhow, Error, Result};
use image::{ImageReader, Rgb};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use itertools::multizip;
use memmap2::{Mmap, MmapMut};
use ndarray::{prelude::*, Array, ArrayView3, Axis, Slice};
use ndarray_npy::{read_npy, write_zeroed_npy, ViewMutNpyExt, ViewNpyError, ViewNpyExt};
use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pyo3::{prelude::*, types::PyType};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use tempfile::tempdir;

use crate::{
    ffmpeg::{ensure_ffmpeg, make_video, Preset},
    signals::DeferredSignal,
    transforms::{
        annotate, apply_transforms, array2_to_grayimage, binary_avg_to_rgb, gray_to_rgbimage,
        grayimage_to_array2, interpolate_where_mask, linearrgb_to_srgb, pack_single,
        process_colorspad, process_grayspad, unpack_single, Transform,
    },
    utils::sorted_glob,
};

#[pyclass]
#[derive(Debug)]
pub struct PhotonCube {
    #[pyo3(get)]
    pub path: PathBuf,

    // Custom getter/setter implemented below
    pub cfa_mask: Option<Array2<bool>>,

    // Custom getter/setter implemented below
    pub inpaint_mask: Option<Array2<bool>>,

    #[pyo3(get, set)]
    pub start: isize,

    #[pyo3(get, set)]
    pub end: Option<isize>,

    #[pyo3(get, set)]
    pub burst_size: Option<usize>,

    #[pyo3(get)]
    pub quantile: Option<f32>,

    #[pyo3(get)]
    pub transforms: Vec<Transform>,

    _storage: Mmap,
}
pub type PhotonCubeView<'a> = ArrayView3<'a, u8>;

pub trait VirtualExposure {
    /// Create parallel iterator over all virtual exposures (bitplane averages of depth `burst_size`)
    fn par_virtual_exposures(
        &self,
        burst_size: usize,
        step: usize,
        drop_last: bool,
    ) -> impl IndexedParallelIterator<Item = Array2<f32>>;

    /// Create a sequential iterator over all virtual exposures (bitplane averages of depth `burst_size`)
    fn virtual_exposures(
        &self,
        burst_size: usize,
        step: usize,
        drop_last: bool,
    ) -> impl Iterator<Item = Array2<f32>>;
}

impl<'a> VirtualExposure for PhotonCubeView<'a> {
    fn par_virtual_exposures(
        &self,
        burst_size: usize,
        step: usize,
        drop_last: bool,
    ) -> impl IndexedParallelIterator<Item = Array2<f32>> {
        let num_frames = if drop_last {
            self.len_of(Axis(0)) / burst_size
        } else {
            (self.len_of(Axis(0)) as f32 / burst_size as f32).ceil() as usize
        };

        self.axis_chunks_iter(Axis(0), burst_size)
            // Make it parallel
            .into_par_iter()
            .step_by(step)
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

    fn virtual_exposures(
        &self,
        burst_size: usize,
        step: usize,
        drop_last: bool,
    ) -> impl Iterator<Item = Array2<f32>> {
        let num_frames = if drop_last {
            self.len_of(Axis(0)) / burst_size
        } else {
            (self.len_of(Axis(0)) as f32 / burst_size as f32).ceil() as usize
        };

        self.axis_chunks_iter(Axis(0), burst_size)
            .step_by(step)
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

// Note: Methods in this `impl` block aren't exposed to python
impl PhotonCube {
    /// Open a photoncube from a memmapped `.npy` file.
    /// For more see `open_py`, the python analogue to this method.
    pub fn open<P>(path: P) -> Result<Self>
    where
        P: AsRef<Path>,
    {
        let path = path.as_ref();
        let ext = path
            .extension()
            .expect("File should have valid filename")
            .to_ascii_lowercase();

        if !path.exists() || !path.is_file() || ext != "npy" {
            // TODO: This should probably be a specific IO error?
            return Err(anyhow!("No `.npy` file found at {}!", path.display()));
        }

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            path: path.to_path_buf(),
            cfa_mask: None,
            inpaint_mask: None,
            start: 0,
            end: None,
            burst_size: None,
            quantile: None,
            transforms: vec![],
            _storage: mmap,
        })
    }

    /// Convert a photon cube stored as a set of `.bin` files to a `.npy` one.
    /// For more see `convert_to_npy_py`, the python analogue to this method.
    pub fn convert_to_npy<P: AsRef<Path>>(
        src: P,
        dst: P,
        is_full_array: bool,
        message: Option<&str>,
    ) -> Result<()> {
        let path = src.as_ref();

        if !path.exists() || !path.is_dir() {
            // This should probably be a specific IO error?
            Err(anyhow!(
                "Directory of `.bin` files not found at {}!",
                path.display()
            ))
        } else {
            let paths = sorted_glob(path, "**/*.bin")?;
            if paths.is_empty() {
                return Err(anyhow!("No .bin files found in {}!", path.display()));
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
            let file = File::create(dst.as_ref())?;
            write_zeroed_npy::<u8, _>(&file, (t, h, w)).map_err(|e| anyhow!(e))?;

            // Memory-map the file and create the mutable view.
            let file = OpenOptions::new().read(true).write(true).open(dst)?;
            let mut mmap = unsafe { MmapMut::map_mut(&file)? };
            let mut view_mut =
                ArrayViewMut3::<u8>::view_mut_npy(&mut mmap).map_err(|e| anyhow!(e))?;

            // Modify the array, and write data to it in a streaming manner.
            let pbar = if let Some(msg) = message {
                ProgressBar::new(paths.len() as u64)
                    .with_style(ProgressStyle::with_template(
                        "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                    ).map_err(|e| anyhow!(e))?)
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
        ArrayView3::<u8>::view_npy(&self._storage)
    }

    /// Equivalent to setting `cube.transforms` directly.
    /// Method mirrors python API.
    pub fn set_transforms(&mut self, transforms: Vec<Transform>) {
        self.transforms = transforms;
    }

    /// Equivalent to setting `cube.quantile` directly.
    /// Method mirrors python API.
    pub fn set_quantile(&mut self, quantile: Option<f32>) {
        self.quantile = quantile;
    }

    /// Load either a 2D NPY file or an intensity-only image file as an array of booleans.
    /// Note: For the image, any pure white pixels are false, all others are true.
    ///       This is contrary to what you might expect but enables us to load in the
    ///       colorSPAD's cfa array and have a mask representing the colored pixels.
    pub fn _try_load_mask<P: AsRef<Path>>(path: P) -> Result<Array2<bool>> {
        let path_obj = path.as_ref();
        let ext = path_obj.extension().unwrap().to_ascii_lowercase();

        if !path_obj.exists() {
            // This should probably be a specific IO error?
            Err(anyhow!("File not found at {}!", path_obj.display()))
        } else if ext == "npy" || ext == "npz" {
            let arr: Array2<bool> = read_npy(path)?;
            Ok(arr)
        } else {
            let arr = grayimage_to_array2(ImageReader::open(path)?.decode()?.into_luma8())
                .mapv(|v| v != 255);
            Ok(arr)
        }
    }

    /// Return function to process a single virtual exposure, depends on any masks (cfa/inpaint).
    pub fn process_single(
        &self,
        invert_response: bool,
        tonemap2srgb: bool,
        colorspad_fix: bool,
        grayspad_fix: bool,
    ) -> Result<impl Fn(Array2<f32>) -> Result<Array2<f32>> + '_> {
        if colorspad_fix && grayspad_fix {
            return Err(anyhow!(
                "Cannot use both `colorspad_fix` and `grayspad_fix`."
            ));
        };
        Ok(move |mut frame| {
            // Invert SPAD response
            if invert_response {
                frame = binary_avg_to_rgb(frame, 1.0, None);
            }

            // Apply sRGB tonemapping
            if tonemap2srgb {
                frame = linearrgb_to_srgb(frame);
            }

            // Apply any frame-level fixes
            if colorspad_fix {
                frame = process_colorspad(frame);
            }
            if grayspad_fix {
                frame = process_grayspad(frame);
            }

            // Demosaic frame by interpolating white pixels
            if let Some(mask) = &self.cfa_mask {
                frame = interpolate_where_mask(&frame, mask, false)?;
            }

            // Inpaint any hot/dead pixels
            if let Some(mask) = &self.inpaint_mask {
                frame = interpolate_where_mask(&frame, mask, false)?;
            }
            Ok(frame)
        })
    }

    /// Apply corrections such as inpainting, rotating/flipping and any fixes directly
    /// to every bitplane and save results as new `.npy` file.
    pub fn process_cube<P: AsRef<Path>>(
        &self,
        dst: P,
        colorspad_fix: bool,
        grayspad_fix: bool,
        message: Option<&str>,
    ) -> Result<(usize, usize, usize)> {
        let path = dst.as_ref();
        let ext = path.extension().unwrap().to_ascii_lowercase();

        if path.is_file() || ext != "npy" {
            // This should probably be a specific IO error?
            return Err(anyhow!("Destination exists, or is not a `.npy` file."));
        }

        if colorspad_fix && grayspad_fix {
            return Err(anyhow!(
                "Cannot use both `colorspad_fix` and `grayspad_fix`."
            ));
        };

        // Load all data
        let view = self.view()?;
        let slice = view.slice_axis(Axis(0), Slice::new(self.start, self.end, 1));

        // Processing closure applied to every bitplane
        let process_bitplane = |src_frame| {
            let mut frame = unpack_single::<u8>(&src_frame, 1).unwrap();

            // Apply any frame-level fixes
            if colorspad_fix {
                frame = process_colorspad(frame);
            }
            if grayspad_fix {
                frame = process_grayspad(frame);
            }

            // Demosaic frame by interpolating white pixels
            if let Some(mask) = &self.cfa_mask {
                frame = interpolate_where_mask(&frame, mask, true).unwrap();
            }

            // Inpaint any hot/dead pixels
            if let Some(mask) = &self.inpaint_mask {
                frame = interpolate_where_mask(&frame, mask, true).unwrap();
            }

            // Convert to image, apply transforms and convert back
            let frame = array2_to_grayimage(frame);
            let frame = apply_transforms(frame, &self.transforms[..]);
            grayimage_to_array2(frame)
        };

        // Create empty dst file
        let file = File::create(dst.as_ref())?;
        let probe_bitplane = process_bitplane(slice.slice(s![0, .., ..]));
        let (full_h, full_w) = probe_bitplane.dim();
        let (out_h, out_w) = pack_single(&probe_bitplane.view(), 1)?.dim();
        write_zeroed_npy::<u8, _>(&file, (slice.len_of(Axis(0)), out_h, out_w))
            .map_err(|e| anyhow!(e))?;

        // Memory-map the file and create the mutable view.
        let file = OpenOptions::new().read(true).write(true).open(dst)?;
        let mut mmap = unsafe { MmapMut::map_mut(&file)? };
        let mut view_mut = ArrayViewMut3::<u8>::view_mut_npy(&mut mmap).map_err(|e| anyhow!(e))?;

        // Conditionally setup a pbar
        // TODO: Refactor this conditional pbar BS
        let pbar = if let Some(msg) = message {
            ProgressBar::new(slice.len_of(Axis(0)) as u64)
                .with_style(ProgressStyle::with_template(
                    "{msg} ETA:{eta}, [{elapsed_precise}] {wide_bar:.cyan/blue} {pos:>6}/{len:6}",
                )?)
                .with_message(msg.to_owned())
        } else {
            ProgressBar::hidden()
        };

        // Actually iterate and transform the data
        (slice.axis_iter(Axis(0)), view_mut.axis_iter_mut(Axis(0)))
            .into_par_iter()
            .progress_with(pbar)
            .for_each(|(src_frame, mut dst_frame)| {
                // Process, bitpack it and save
                let frame = process_bitplane(src_frame);
                dst_frame.assign(&pack_single(&frame.view(), 1).unwrap());
            });

        Ok((slice.len_of(Axis(0)), full_h, full_w))
    }

    /// Save all virtual exposures to a folder.
    pub fn save_images<P: AsRef<Path>>(
        &self,
        img_dir: P,
        process_fn: Option<impl Fn(Array2<f32>) -> Result<Array2<f32>> + Send + Sync>,
        annotate_frames: bool,
        message: Option<&str>,
        step: usize,
    ) -> Result<isize> {
        // Do some quick validation
        let img_dir_path = img_dir.as_ref();
        if self.burst_size.is_none() {
            return Err(anyhow!(
                "Parameter `burst_size` must be set before virtual exposures can be created!"
            ));
        }
        create_dir_all(img_dir_path).ok();

        // Create virtual exposures iterator over all data
        let view = self.view()?;
        let slice = view.slice_axis(Axis(0), Slice::new(self.start, self.end, 1));
        let virtual_exps = slice.par_virtual_exposures(self.burst_size.unwrap(), step, true);

        // Conditionally setup a pbar
        let pbar = if let Some(msg) = message {
            ProgressBar::new((slice.len_of(Axis(0)) / self.burst_size.unwrap()) as u64)
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
                    let frame = apply_transforms(frame, &self.transforms[..]);
                    let mut frame = gray_to_rgbimage(&frame);

                    if annotate_frames {
                        let start_idx = i * self.burst_size.unwrap() + (self.start as usize);
                        let text = format!(
                            "{:06}:{:06}",
                            start_idx,
                            start_idx + self.burst_size.unwrap()
                        );
                        annotate(&mut frame, &text, Rgb([252, 186, 3]));
                    }

                    // Throw error if we cannot save, and stop all processing.
                    let path = img_dir_path.join(format!("frame{:06}.png", i));
                    frame.save(&path)?;
                    pbar.inc(1);

                    Ok::<isize, anyhow::Error>(count + 1)
                },
            )
            .try_reduce(|| 0, |acc, x| Ok(acc + x))?;
        Ok(num_frames)
    }

    /// Save all virtual exposures as a video (and optionally images).
    #[allow(clippy::too_many_arguments)]
    pub fn save_video<P: AsRef<Path>>(
        &self,
        output: P,
        fps: u64,
        img_dir: Option<P>,
        process_fn: Option<impl Fn(Array2<f32>) -> Result<Array2<f32>> + Send + Sync>,
        annotate_frames: bool,
        crf: Option<u32>,
        preset: Option<Preset>,
        message: Option<&str>,
        metadata: Option<HashMap<&str, &str>>,
        step: usize,
    ) -> Result<isize> {
        // Get img path or tempdir, ensure it exists.
        let tmp_dir = tempdir()?;
        let img_dir = img_dir
            .map(|p| {
                let path: &Path = p.as_ref();
                path.to_owned()
            })
            .unwrap_or(tmp_dir.path().to_path_buf());
        let pattern = img_dir.join("frame%06d.png");
        create_dir_all(img_dir.clone()).ok();

        // Generate preview frames
        let num_frames = self.save_images(img_dir, process_fn, annotate_frames, message, step)?;

        // Assemble them into a video
        ensure_ffmpeg(true);
        make_video(
            pattern.to_str().expect("Pattern should be valid"),
            output,
            fps,
            num_frames as u64,
            crf,
            preset,
            message,
            metadata,
        );

        tmp_dir.close()?;
        Ok(num_frames)
    }
}

// Note: Methods in this `impl` block are exposed to python.
#[pymethods]
impl PhotonCube {
    /// Open a photoncube from a memmapped `.npy` file.
    /// Note: Loading from a directory of .bin files has been deprecated
    /// as it is non-trivial to do when using the full-array and requires
    /// entirely loading the photoncube into memory. Convert to `.npy` first.
    #[classmethod]
    #[pyo3(name = "open", signature = (path))]
    pub fn open_py(_: &Bound<'_, PyType>, path: PathBuf) -> Result<Self> {
        Self::open(path)
    }

    /// Convert a photon cube stored as a set of `.bin` files to a `.npy` one. This is done
    /// in a streaming manner and avoids loading all the data to memory.
    /// The `.npy` format enables memory mapping the photon cube and is usually faster.
    /// Note: Function assumes either 256x512 of 512x512 frames that are bitpacked along width dim.
    #[staticmethod]
    #[pyo3(
        name = "convert_to_npy",
        signature = (src, dst, is_full_array=false, message=None)
    )]
    pub fn convert_to_npy_py(
        py: Python,
        src: PathBuf,
        dst: PathBuf,
        is_full_array: bool,
        message: Option<&str>,
    ) -> PyResult<()> {
        let _defer = DeferredSignal::new(py, "SIGINT")?;
        Self::convert_to_npy(src, dst, is_full_array, message).map_err(|e| e.into())
    }

    /// Apply corrections such as inpainting, rotating/flipping and any fixes directly
    /// to every bitplane and save results as new `.npy` file.
    #[pyo3(
        name = "process_cube",
        signature = (dst, colorspad_fix=false, grayspad_fix=false, message=None)
    )]
    pub fn process_cube_py<'py>(
        &'py self,
        py: Python<'py>,
        dst: PathBuf,
        colorspad_fix: bool,
        grayspad_fix: bool,
        message: Option<&str>,
    ) -> Result<(usize, usize, usize)> {
        let _defer = DeferredSignal::new(py, "SIGINT")?;
        self.process_cube(dst, colorspad_fix, grayspad_fix, message)
    }

    #[getter(inpaint_mask)]
    pub fn inpaint_mask_getter<'py>(&'py self, py: Python<'py>) -> Result<Option<Py<PyAny>>> {
        // See: https://github.com/PyO3/rust-numpy/issues/408
        self.inpaint_mask
            .as_ref()
            .map(|a| -> Result<Py<PyAny>> {
                let py_arr = a.to_pyarray_bound(py).to_owned().into_py(py);
                py_arr
                    .getattr(py, "setflags")?
                    .call1(py, (false, None::<bool>, None::<bool>))?;
                Ok(py_arr)
            })
            .transpose()
    }

    #[setter(inpaint_mask)]
    pub fn inpaint_mask_setter(&mut self, arr: Option<&Bound<'_, PyArray2<bool>>>) -> Result<()> {
        self.inpaint_mask = arr.map(|a| a.to_owned().to_owned_array());
        Ok(())
    }

    #[getter(cfa_mask)]
    pub fn cfa_mask_getter<'py>(&'py self, py: Python<'py>) -> Result<Option<Py<PyAny>>> {
        // See: https://github.com/PyO3/rust-numpy/issues/408
        self.cfa_mask
            .as_ref()
            .map(|a| -> Result<Py<PyAny>> {
                let py_arr = a.to_pyarray_bound(py).to_owned().into_py(py);
                py_arr
                    .getattr(py, "setflags")?
                    .call1(py, (false, None::<bool>, None::<bool>))?;
                Ok(py_arr)
            })
            .transpose()
    }

    #[setter(cfa_mask)]
    pub fn cfa_mask_setter(&mut self, arr: Option<&Bound<'_, PyArray2<bool>>>) -> Result<()> {
        self.cfa_mask = arr.map(|a| a.to_owned().to_owned_array());
        Ok(())
    }

    /// Load the color-filter array associated with the photoncube, if applicable.
    /// Will be used for processing if loaded.
    #[pyo3(signature = (path))]
    pub fn load_cfa(&mut self, path: PathBuf) -> Result<()> {
        self.cfa_mask = Some(Self::_try_load_mask(path)?);
        Ok(())
    }

    /// Load an inpainting mask. This can be called multiple times, the masks will
    /// simply be ORed together (interpolate where mask is true/white).
    #[pyo3(signature = (path))]
    pub fn load_mask(&mut self, path: PathBuf) -> Result<()> {
        let mut new_mask = Self::_try_load_mask(path)?;

        if let Some(mask) = &self.inpaint_mask {
            new_mask = mask.bitor(new_mask);
        }
        self.inpaint_mask = Some(new_mask);
        Ok(())
    }

    /// Set the range of the photoncube, this is used for slicing and frame preview
    /// where `burst_size` will be the number of frames to average together.
    #[pyo3(signature = (start=0, end=None, burst_size=None))]
    pub fn set_range(&mut self, start: isize, end: Option<isize>, burst_size: Option<usize>) {
        self.start = start;
        self.end = end;
        self.burst_size = burst_size;
    }

    /// Define which transforms to apply to virtual exposures. Tranforms are applied
    /// sequentially and can thus be composed (i.e: Rot90+Rot90=Rot180).
    /// Options are: "Identity", "Rot90", "Rot180", "Rot270", "FlipUD", "FlipLR"
    #[pyo3(name = "set_transforms", signature=(transforms))]
    pub fn set_transforms_py(&mut self, transforms: Vec<Transform>) -> Result<()> {
        self.transforms = transforms;
        Ok(())
    }

    /// Set quantile to use when inverting SPAD response, this is only used
    /// when `invert_response` is set to True. Quantile must be in 0-1 range.
    #[pyo3(name = "set_quantile", signature=(quantile=None))]
    pub fn set_quantile_py(&mut self, quantile: Option<f32>) -> Result<()> {
        if let Some(qtl) = quantile {
            if !(0.0..=1.0).contains(&qtl) {
                return Err(anyhow!("Quantile value must be in 0-1 range."));
            }
        }
        self.quantile = quantile;
        Ok(())
    }

    /// Save all virtual exposures to a folder.
    #[pyo3(
        name = "save_images",
        signature = (img_dir, invert_response=false, tonemap2srgb=false, colorspad_fix=false, grayspad_fix=false, annotate_frames=false, message=None, step=1)
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn save_images_py(
        &self,
        py: Python,
        img_dir: PathBuf,
        invert_response: bool,
        tonemap2srgb: bool,
        colorspad_fix: bool,
        grayspad_fix: bool,
        annotate_frames: bool,
        message: Option<&str>,
        step: usize,
    ) -> Result<isize> {
        let _defer = DeferredSignal::new(py, "SIGINT")?;
        let process =
            self.process_single(invert_response, tonemap2srgb, colorspad_fix, grayspad_fix)?;
        self.save_images(img_dir, Some(process), annotate_frames, message, step)
    }

    /// Save all virtual exposures as a video (and optionally images).
    #[pyo3(
        name = "save_video",
        signature = (output, fps=24, img_dir=None, invert_response=false, tonemap2srgb=false, colorspad_fix=false, grayspad_fix=false, annotate_frames=false, crf=28, preset="ultrafast", message=None, step=1)
    )]
    #[allow(clippy::too_many_arguments)]
    pub fn save_video_py(
        &self,
        py: Python,
        output: PathBuf,
        fps: Option<u64>,
        img_dir: Option<PathBuf>,
        invert_response: bool,
        tonemap2srgb: bool,
        colorspad_fix: bool,
        grayspad_fix: bool,
        annotate_frames: bool,
        crf: u32,
        preset: Option<&str>,
        message: Option<&str>,
        step: usize,
    ) -> Result<isize> {
        let _defer = DeferredSignal::new(py, "SIGINT")?;
        let process =
            self.process_single(invert_response, tonemap2srgb, colorspad_fix, grayspad_fix)?;

        self.save_video(
            output,
            fps.unwrap_or(28),
            img_dir,
            Some(process),
            annotate_frames,
            Some(crf),
            preset.map(Preset::from_str).transpose()?,
            message,
            None,
            step,
        )
    }

    /// Total number of bitplanes in cube (independent of `set_range`).
    pub fn len(&self) -> usize {
        self.view().expect("Cannot load photoncube").len_of(Axis(0))
    }

    /// True if the photon cube is empty or not yet loaded.
    pub fn is_empty(&self) -> bool {
        self.view().map_or(true, |p| p.len_of(Axis(0)) == 0)
    }

    /// Get shape of photoncube with possible bitpacking
    #[getter]
    pub fn shape(&self) -> (usize, usize, usize) {
        self.view().expect("Cannot load photoncube").dim()
    }

    /// Same as `self.len()`
    pub fn __len__(&self) -> usize {
        self.len()
    }
}
