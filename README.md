# greycard-denoise-raw

Raw file in, linear float32 TIFF or linear DNG out, through the learned
raw-domain denoiser from [greycard](https://github.com/jessolmstead/greycard)
(`tools/denoise`; weights at
[Hugging Face](https://huggingface.co/jessolmstead/greycard-denoise)). The
network demosaics *and* denoises a Bayer mosaic; this repo re-implements the
code around it (level normalisation, noise estimation, variance-stabilising
transform, tiling) in Python. GPL-3.0-or-later, like its sources.

    uv venv --python 3.12
    uv pip install numpy rawpy onnxruntime tifffile

    # tier "balanced", best provider found
    python greycard_denoise_raw.py photo.ARW -o photo.tif
    python greycard_denoise_raw.py photo.ARW -m best --provider cpu

    # needs: uv pip install torch onnx onnx2torch
    python greycard_denoise_raw.py photo.ARW --backend torch

Models (4.9 / 9 / 20 MB) are fetched on first use into `./models` or
`~/.cache/greycard-denoise-raw` (`GREYCARD_DENOISE_RAW_MODELS` overrides) and
SHA-256 checked.

## Output

* float32, H x W x 3, **linear**, **camera-native RGB**: black level removed,
  divided by (white level - black level). No white balance, no colour matrix,
  no tone curve. `--wb camera` multiplies by the as-shot gains (G = 1). Values
  may exceed 1 slightly and may be a hair below 0 (noise is not clipped).
* Rotated upright per the file's orientation flag (`--no-rotate` keeps sensor
  orientation).
* The TIFF description holds JSON: model, estimated noise model, scaling.

## DNG output

    # 16-bit LinearRaw DNG (default)
    python greycard_denoise_raw.py photo.ARW -o photo.dng

    # float32 DNG 1.4
    python greycard_denoise_raw.py photo.ARW -o photo.dng --dng-bits 32

A linear DNG (`PhotometricInterpretation = LinearRaw`) that raw converters
treat as a raw file: camera Make/Model + `UniqueCameraModel` (so DCP/ICC lookup
by camera works), `ColorMatrix1` (XYZ D65 -> camera, from LibRaw),
`AsShotNeutral` (as-shot white balance), black 0 / white 1, sensor orientation
kept and flagged in the Orientation tag, and the EXIF (exposure, ISO, lens,
date) copied from the raw if `exiftool` is installed. The pixels are the same
denoised, camera-native linear RGB as the TIFF.

**ART caveat:** ART's float-DNG loader reads single-plane (CFA) float data only
(`packed_dng_load_raw`, `float_raw_image`), so a 3-sample float DNG is
*mis-rendered* by ART (about 4x too dark, wrong colours). `--dng-bits 16`, the
default, works in ART: verified there to pick the same `rgb_cam` matrix and
white balance as the original raw and to apply the camera's DCP, rendering
within a few percent of the ARW. 16 bits clips negative noise and quantises the
deep shadows (1/65535); float32 is correct for readers that support it
(exiftool and LibRaw open it fine).

## Accelerators

* **onnxruntime + CoreML** (macOS, default): static 608x608 shape; ~4.5x CPU
  on Apple Silicon, fp32-exact to ~3e-6.
* **onnxruntime + CUDA / ROCm / DirectML**: `onnxruntime-gpu` etc.; picked by
  `auto`; **untested here**.
* **onnxruntime CPU**: fallback; a failing GPU provider falls back to it with
  a message.
* **`--backend torch`** (cuda, mps, xpu, cpu): via onnx2torch; MPS ~1.3x slower
  than CoreML, ~10 s start-up; CPU 2x slower than ORT.

## Limits

* Bayer sensors only (X-Trans is not supported by the model). Non-RGB filters
  are refused.
* `tile + 2*margin` must be a multiple of 16 (the U-Net's strides); default
  1024 / 96.
* The noise model is *estimated from the frame* (port of greycard's
  `estimate_noise`); use `--noise` to override. Hot pixels, defects and
  chromatic aberration are not handled.

## Tests

`pytest` -- VST round trip, noise-model recovery from synthetic
Poisson-Gaussian data, CFA phase handling, tile-seam invariance with a
stand-in network, DNG round trips.
