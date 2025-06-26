### Overview
Command line utility and python/rust library to preview a photon cube as a video or a series of frames and/or convert it to a different format.

Features:
- Zero runtime dependencies, except for FFmpeg, which will be automatically downloaded if not found on path.
- Multi-threaded and (despite very unoptimized code) blazing fast, usually an order of magnitude or two faster than equivalent cpu-based numpy+numba code. 
- Supports a collection of .bin or .npy file. Assumes bitpacking in the width axis in all cases.
- Demosaicing, inpainting, tonemapping, inverting SPAD response, and transforms (rotate/flip) are also supported.

Limitations:
- Not all features are exposed to python yet.

### Getting Started 
To compile it locally, simply clone the repository, `cd` into it and run:
```
pip install -v . 
```
Ensure you have an up-to-date pip, and an adequate rust toolchain installed ([install from here](https://rustup.rs/)), else this might fail. This should work for python >= 3.8 (tested with 3.8 and py 3.12). You might have to update your rust toolchain with `rustup update` (MSRV: 1.74.0).


This should pull in any rust dependencies and compile bindings that are compatible with your machine's env. It will create both a CLI and package.  
If this is your first time compiling a rust project, this may take a few minutes.


### CLI Usage:

Three main functions are available via the CLI: `preview`, `process`, and `convert`.

```
$ photoncube -h

Convert a photon cube (npy file/directory of bin files) between formats or to a video preview (mp4) by naively averaging frames

Usage: photoncube <COMMAND>

Commands:
  convert  Convert photoncube from a collection of .bin files to a .npy file
  preview  Extract and preview virtual exposures from a photoncube
  process  Apply corrections directly to every bitplane and save results as new .npy file
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

For more info, run `photoncube <COMMAND> --help`.


#### Preview

To preview a photoncube as a video or series of frames (either averaged or single bitplanes), the photon cube needs to be saved as a npy file. See `convert` for how to convert a folder of `.bin` files to an `.npy` file.


Here we preview a photoncube as a video and apply a rotation by 270 degrees then flip frames left-right:
```
photoncube preview -i binary.npy -t=rot270 -t=flip-lr -o video.mp4
```
_Note:_ Transforms can be composed arbitrarily, and order matters. Valid transforms are enumerated [here](./src/transforms.rs) and are serialized in kebab-case, i.e: Transform::FlipLR is "flip-lr" and so on. 


You can annotate frames and correct for cfa arrays and inpaint like so:
```
photoncube preview -i binary.npy -a --cfa-path=rgbw_oh_bn_color_ss2_corrected.png --inpaint-path=colorspad_inpaint_mask.npy -o video.mp4
```
_Note:_ Multiple `inpaint-path`s can be specified. These masks will be OR'd together then used.


Individual frames can be saved using the `--img-dir` option, and a range of frames, as well as the window over which we aggregate frames, can be specified:
```
photoncube preview -i binary.npy --img-dir=frames/ --start=20000 --end=50000 --burst-size=100
```
_Note:_ You can, of course, save a video and frames at the same time!


In particular, if you wish to view individual bitplanes, you can set `burst-size` to one, and use the `step` argument to skip over frames as to not create a huge video. For a photoncube captured at 15kHz, a realtime playback of individual bitplanes can be shown:
```
photoncube preview -i binary.npy --batch-size=1 --step=625 -o video.mp4
```
_Note:_ Here we set step=625 because the default playback speed is 24 fps (i.e: 15k/24 == 625), but the fps can also be changed using `--fps`, although using a very large or small fps will cause ffmpeg issues.


#### Process

Many of the operations detailed above can be applied directly to a photoncube at the bitplane level, notably all transforms, slicing operations (start/end) and inpainting/cfa-correction can be applied. The later will use a dithering-like approach to perform stochastic inpainting. Again, only numpy files are supported.

Here we process a photoncube by applying cfa corrections and a few transforms:
```
photoncube process -i binary.npy --cfa-path=rgbw_oh_bn_color_ss2_corrected.png --inpaint-path=colorspad_inpaint_mask.npy --start=0 --end=100000 -t flip-lr -o processed.npy
```


#### Convert

You can convert from a collection of .bin files to a single .npy file like so:
```
photoncube convert -i <DIR OF BINS> -o <OUTPUT>
```

By default this assumes half array frames (i.e: 256x512), you can specify `--full-array` for the whole array. 


### Library Usage (Python)

The python API mirrors the rust and CLI functionality, for more info on how to use it please see the above section.

```python
from photoncube import PhotonCube, Transform

PhotonCube.convert_to_npy(
    # Directory containing `.bin` files
    "full-array/binary/16kHz/", 
    "test.npy", 
    is_full_array=True, 
    message="Converting..."
)

# Open cube 
pc = PhotonCube.open("test.npy")

# Get a frame
frame = pc[10]

# Load inpainting masks, both .png and .npy supported.
pc.load_mask("dead_pixels.png")
pc.load_mask("hot_pixels.npy")

# Inpainting masks will be OR'ed together
print(pc.inpaint_mask)

# If using colorspad, you can specify the color filter array 
# and all non-white pixels will be interpolated out
pc.load_cfa("cfa.png")

# Generate 100 preview images with bit-depth of 256 
pc.set_range(0, 25600, 256)
pc.save_images(
    "tmp/", 
    invert_response=False,
    tonemap2srgb=False,
    colorspad_fix=False,
    grayspad_fix=False,
    annotate_frames=False,
    message="Saving Images..."  # If not None, a progress bar will be drawn
)

# Make video preview instead, but transform frames first, and invert the SPAD
# response, and normalize to a 95% quantile, for better low-light performance
pc.set_transforms([Transform.Rot90, Transform.FlipUD])
pc.set_quantile(0.95)
pc.save_video(
    "output.mp4", fps=24, 
    # If specified, images are also saved
    img_dir=None,
    invert_response=True,
    # Options from `save_images` can be used here too:
    message="Making video..." 
) 

# Save a new photoncube that has been processed with any transforms,
# cfas, masks or color/grayspad fixes  such as cropping dead regions or column swapping.
pc.process_cube("processed.npy")
```
For the full python API and up-to-date typing, see [photoncube.pyi](./photoncube.pyi).


### Development

We use [maturin](https://www.maturin.rs/) as a build system, which enables this package to be built as a python extension using [pyo3](https://pyo3.rs) bindings. Some other tools are needed for development work, which can be installed using `pip install -v .[dev]`.

There are both rust tests, and python ones, run them with:
```
cargo test && pytest . 
```

#### Code Quality

We use `rustfmt` to format the codebase, we use some customizations (i.e for import sorting) which require nightly. First ensure you have the nightly toolchain installed with:
```
rustup toolchain install nightly
```

Then you can format the code using:

```
cargo +nightly fmt 
```

Similarly we use `black` to format the python parts of the project. 


To keep the project lean, it's recommended to check for unused dependencies [using this tool](https://github.com/est31/cargo-udeps), or [this one](https://github.com/bnjbvr/cargo-machete), like so: 

```
cargo +nightly udeps
cargo machete --with-metadata
```
