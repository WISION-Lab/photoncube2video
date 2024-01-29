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
pip install . 
```
Ensure you have an up-to-date pip, else this might fail. This should work for python >= 3.6.


This should pull in any rust dependencies and compile bindings that are compatible with your machine's env. It will create both a CLI and package.  
If this is your first time compiling a rust project, this may take a few minutes.

### Library Usage (Python)

```python
from photoncube2video import PhotonCube

PhotonCube.convert_to_npy(
        "full-array/binary/16kHz/", 
        "test.npy", 
        is_full_array=True, 
        message="Converting..."
)
```

### CLI Usage:

Two main functions are available via the CLI: `preview` and `convert`.

```
$ photoncube2video -h

Convert a photon cube (npy file/directory of bin files) between formats or to a video preview (mp4) by naively averaging frames

Usage: photoncube2video <COMMAND>

Commands:
  convert  Convert photoncube from a collection of .bin files to a .npy file
  preview  Extract and preview virtual exposures from a photoncube
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version
```

#### Preview

Transforms can be composed. Here we apply a rotation by 270 degrees then flip frames left-right:
```
photoncube2video preview -i binary.npy -t=rot270 -t=flip-lr -o video.mp4
```


You can annotate frames and correct for cfa arrays and inpaint like so:
```
photoncube2video preview -i binary.npy -a --cfa-path=rgbw_oh_bn_color_ss2_corrected.png --inpaint-path=colorspad_inpaint_mask.npy -o video.mp4
```
Multiple `inpaint-path`s can be specified. These masks will be OR'd together then used.


Individual frames can be saved using the `--img-dir` option, and a range of frames can also be specified:
```
photoncube2video preview -i binary.npy --img-dir=frames/ --start=20000 --end=50000
```

#### Convert

You can convert from a collection of .bin files to a single .npy file like so:
```
photoncube2video convert -i <DIR OF BINS> -o <OUTPUT>
```

By default this assumes half array frames (i.e: 256x512), you can specify `--full-array` for the whole array. 

### Development

We use [maturin](https://www.maturin.rs/) as a build system, which enables this package to be built as a python extension using [pyo3](https://pyo3.rs) bindings.

#### Code Quality

We use `rustfmt` to format the codebase, we use some customizations (i.e for import sorting) which require nightly, use:
```
cargo +nightly fmt 
```

To keep the project lean, it's recommended to check for unused dependencies [using this tool](https://github.com/est31/cargo-udeps), like so: 

```
cargo +nightly udeps
```
