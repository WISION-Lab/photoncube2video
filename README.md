### Overview
Command line utility to preview a photon cube as a video or a series of frames.

Features:
- Zero runtime dependencies, except for FFmpeg, which will be automatically downloaded if not found on path.
- Multi-threaded and (despite very unoptimized code) blazing fast, usually an order of magnitude faster than equivalent cpu-based numpy+numba code. 
- Multiple inputs types are supported: bin files, npy files (which will be memmapped) and npy.gz files. Assumes bitpacking in the width axis in all cases.
- Demosaicing, inpainting, tonemapping, inverting SPAD response, and transforms (rotate/flip) are also supported.

Limitations:
- If the photoncube is compressed with GZ, it will need to be read into memory and decompressed before any processing happens, so if the decompressed data is larger than RAM, it will not work.


### Getting Started 
To compile it locally, simply make sure you have an adequate rust toolchain installed ([install from here](https://rustup.rs/)).

You can install directly using
```
cargo install --git="https://github.com/WISION-Lab/photoncube2video"
```

Which will create an executable and put in on your path (i.e: in ~/.cargo/bin on linux by default). Or you can clone this project, `cd` into it, and run:

```
cargo install --path .
```

Which will behave like the above command, just not using a tmp folder, or with

```
cargo build --release
```

Which will put the executable in `target/release`.


If this is your first time compiling a rust project, this may take a few minutes.


You can also compile it for a [different platform](https://rust-lang.github.io/rustup/cross-compilation.html), this, for instance, will compile the project for use on a 64bit windows machine:
```
cargo build --release --target=x86_64-pc-windows-msvc
```

### Examples:

Transforms can be composed. Here we apply a rotation by 270 degrees then flip frames left-right:
```
photoncube2video -i binary.npy -t=rot270 -t=flip-lr -o video.mp4
```


You can annotate frames and correct for cfa arrays and inpaint like so:
```
photoncube2video -i binary.npy -a --cfa-path=rgbw_oh_bn_color_ss2_corrected.png --inpaint-path=colorspad_inpaint_mask.npy -o video.mp4
```
Multiple `inpaint-path`s can be specified. These masks will be OR'd together then used.


Individual frames can be saved using the `--img-dir` option, and a range of frames can also be specified:
```
photoncube2video -i binary.npy --img-dir=frames/ --start=20000 --end=50000
```

### Features
Here's the current help section: 
```
$ target/release/photoncube2video -h

Convert a photon cube (npy file/directory of bin files) to a video preview (mp4) by naively averaging frames

Usage: photoncube2video.exe [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>
          Path to photon cube (npy file or directory of .bin files)
  -o, --output <OUTPUT>
          Path of output video
  -d, --img-dir <IMG_DIR>
          Output directory to save PNGs in
      --cfa-path <CFA_PATH>
          Path of color filter array to use for demosaicing
      --inpaint-path [<INPAINT_PATH>...]
          Path of inpainting mask to use for filtering out dead/hot pixels
  -b, --burst-size <BURST_SIZE>
          Number of frames to average together [default: 256]
      --fps <FPS>
          Frame rate of resulting video [default: 25]
  -t, --transform [<TRANSFORM>...]
          Apply transformations to each frame (these can be composed) [possible values: identity, rot90, rot180, rot270, flip-ud, flip-lr]
  -a, --annotate
          If enabled, add bitplane indices to images
      --colorspad-fix
          If enabled, swap columns that are out of order and crop to 254x496
  -s, --start <START>
          Index of binary frame at which to start the preview from (inclusive)
  -e, --end <END>
          Index of binary frame at which to end the preview at (exclusive)
      --invert-response
          If enabled, invert the SPAD's response (Bernoulli process)
      --tonemap2srgb
          If enabled, apply sRGB tonemapping to output
  -h, --help
          Print help
  -V, --version
          Print version
```

### Development

#### Code Quality

We use `rustfmt` to format the codebase, we use some customizations (i.e for import sorting) which require nightly, use:
```
cargo +nightly fmt 
```

To keep the project lean, it's recommended to check for unused dependencies [using this tool](https://github.com/est31/cargo-udeps), like so: 

```
cargo +nightly udeps
```
