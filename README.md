### Overview
Command line utility to preview a photon cube as a video.

Features:
- Zero runtime dependencies, except for FFmpeg, which will be automatically downloaded if not found on path.
- Multi-threaded and (despite very unoptimized code) blazing fast, at least as much as numpy+numba equivalent. 
- Can read both bin files and npy files (assumes bitpacking in the width axis).
- Demosaicing, inpainting, and transforms (rotate/flip) are also supported.

Limitations:
- No support for mem-mapped NPY files yet ([PR here](https://github.com/jturner314/ndarray-npy/pull/45)), so if a npy file is larger than RAM, it will not work.


### Getting Started 
To compile it locally, simply make sure you have an adequate rust toolchain installed ([install from here](https://rustup.rs/)), clone this project, `cd` into it, and run:

```
cargo build --release
```

If this is your first time compiling any rust, this may take a few minutes.
The executable will then be located in `target/release`.

You can also compile it for a [different platform](https://rust-lang.github.io/rustup/cross-compilation.html), this, for instance, will compile the project for use on a 64bit windows machine:
```
cargo build --release --target=x86_64-pc-windows-msvc
```


### Features
Here's the current help section: 
```
$ target/release/photoncube2video.exe -h

Convert a photon cube (npy file/directory of bin files) to a video preview (mp4) by naively averaging frames

Usage: photoncube2video.exe [OPTIONS] --input <INPUT>

Options:
  -i, --input <INPUT>                Path to photon cube (npy file or directory of .bin files)
  -o, --output <OUTPUT>              Path of output video [default: out.mp4]
      --img-dir <IMG_DIR>            Output directory to save PNGs in
      --cfa-path <CFA_PATH>          Path of color filter array to use for demosaicing
      --inpaint-path <INPAINT_PATH>  Path of inpainting mask to use for filtering out dead/hot pixels
  -b, --burst-size <BURST_SIZE>      Number of frames to average together [default: 256]
      --fps <FPS>                    Frame rate of resulting video [default: 25]
  -t, --transform [<TRANSFORM>...]   Apply transformations to each frame (these can be composed) [possible values: identity, rot90, rot180, rot270, flip-ud, flip-lr]
  -a, --annotate                     If enabled, add bitplane indices to images
      --colorspad-fix                If enabled, swap columns that are out of order and crop to 254x496
  -s, --start <START>                Index of binary frame at which to start the preview from (inclusive)
  -e, --end <END>                    Index of binary frame at which to end the preview at (exclusive)
  -h, --help                         Print help
  -V, --version                      Print version
```