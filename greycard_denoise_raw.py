#!/usr/bin/env python3
"""Raw file -> linear float32 TIFF/DNG through greycard's raw-domain denoiser.

The network (https://github.com/jessolmstead/greycard, tools/denoise; weights at
https://huggingface.co/jessolmstead/greycard-denoise) takes a Bayer mosaic and
returns demosaiced, denoised RGB at the mosaic's resolution. This is a Python
re-implementation of the code around it (greycard-ai/src/denoise.rs and
greycard-core/src/develop/{noise,denoise}.rs): level normalisation, noise
estimation, variance-stabilising transform, tiling.

Output is linear, camera-native RGB (no white balance, no colour matrix) unless
--wb camera is given, which only multiplies the result by the as-shot gains.

Licence: GPL-3.0-or-later, like the code and weights it is derived from.
"""

import argparse
import hashlib
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Models

HF = "https://huggingface.co/jessolmstead/greycard-denoise/resolve/main"
MODELS = {
    # tier: (file, bytes, sha256)
    "fast": (
        "denoise-v19.onnx",
        4_932_834,
        "319d024a39d3e31c95e99c91bb5c089691707cbf13f4f077c1c6e49d10235c49",
    ),
    "balanced": (
        "denoise-v15.onnx",
        9_037_296,
        "f5845119e0b0ee63c57aad38129aed1c8925e60ec9b7838c6b7a0fa23a84a546",
    ),
    "best": (
        "denoise-v20.onnx",
        20_297_204,
        "c20aed71a45b2a984570e09d8456c54ce8eff43a334b54c7ebb480c21518f9be",
    ),
}


def cache_dir():
    base = os.environ.get("GREYCARD_DENOISE_RAW_MODELS")
    if base:
        return Path(base)
    here = Path(__file__).resolve().parent / "models"
    if here.is_dir():
        return here
    xdg = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(xdg) / "greycard-denoise-raw"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_model(tier):
    name, size, digest = MODELS[tier]
    d = cache_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    if path.is_file() and sha256(path) == digest:
        return path
    print(
        f"fetching {name} ({size / 1e6:.1f} MB) from Hugging Face",
        file=sys.stderr,
    )
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(f"{HF}/{name}", tmp)
    if sha256(tmp) != digest:
        tmp.unlink()
        raise RuntimeError(f"{name}: SHA-256 mismatch after download")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------
# Variance-stabilising transform (greycard-core `Vst`, tools/denoise/vst.py)

SILENT_B = 1e-8


class Vst:
    """var = a x + b -> unit variance, zero at x = 0 (generalised Anscombe)."""

    def __init__(self, a, b):
        if a <= 0 and b <= 0:  # a silent channel still needs a finite transform
            a, b = 0.0, SILENT_B
        self.a = np.float32(a)
        self.s0 = np.float32(np.sqrt(3.0 / 8.0 * a * a + b))

    def forward(self, x):
        a, s0 = self.a, self.s0
        return 2.0 * x / (np.sqrt(np.maximum(a * x + s0 * s0, 0.0)) + s0)

    def inverse_clean(self, d):
        return self.a * d * d / 4.0 + d * self.s0


# ---------------------------------------------------------------------------
# Noise model estimation (greycard-core `estimate_noise`, Bayer path)

BLOCK = 16
BINS = 32
MIN_BLOCKS_PER_BIN = 8
MIN_SAMPLES = 32
CLIP = 0.98
RESIDUAL_GAIN = 1.25


def estimate_noise(x, rgb_idx):
    """Noise model var = a*level + b per RGB channel.

    `x`: float32 HxW, levels normalised to 0..1 (black removed), before white
    balance. `rgb_idx`: 2x2 array of 0/1/2 (R/G/B) giving the CFA phase.
    Returns (a[3], b[3], info) -- both greens share one fit, as in greycard.
    """
    h, w = x.shape
    a = np.zeros(3)
    b = np.zeros(3)
    info = {"blocks": [0, 0, 0], "bins": [0, 0, 0]}
    if w < BLOCK + 4 or h < BLOCK + 4:
        return a, b, info

    # Squared residual against the mean of the four same-colour neighbours
    # two samples away; defined for the interior [2, n-2).
    r = x[2:-2, 2:-2] - 0.25 * (
        x[2:-2, :-4] + x[2:-2, 4:] + x[:-4, 2:-2] + x[4:, 2:-2]
    )
    r2 = (r * r).astype(np.float64)

    # Block grid: origins 2, 18, ... such that the block ends before n-2.
    nby = len(range(2, h - 2 - BLOCK + 1, BLOCK))
    nbx = len(range(2, w - 2 - BLOCK + 1, BLOCK))
    if nby == 0 or nbx == 0:
        return a, b, info
    v = x[2 : 2 + nby * BLOCK, 2 : 2 + nbx * BLOCK].astype(np.float64)
    # r2 is offset by 2 relative to x, so block origin 2 is r2 origin 0.
    q = r2[: nby * BLOCK, : nbx * BLOCK]

    def blocks(m):
        return m.reshape(nby, BLOCK, nbx, BLOCK)

    vb, qb = blocks(v), blocks(q)
    clipped = (vb >= CLIP).any(axis=(1, 3))  # (nby, nbx)
    # Block origins are even, so the colour of (y, x) in a block is
    # rgb_idx[y & 1, x & 1].
    cmap = np.tile(np.asarray(rgb_idx), (BLOCK // 2, BLOCK // 2))  # 16x16
    for c in range(3):
        m = (cmap == c)[None, :, None, :]
        n = int(m.sum())
        if n < MIN_SAMPLES:
            continue
        mean = (vb * m).sum(axis=(1, 3)) / n
        var = (qb * m).sum(axis=(1, 3)) / n / RESIDUAL_GAIN
        keep = ~clipped
        mean, var = mean[keep], var[keep]
        info["blocks"][c] = int(mean.size)
        bin_ = np.minimum(
            (np.sqrt(np.maximum(mean, 0.0)) * BINS).astype(int), BINS - 1
        )
        sw = swx = swy = swxx = swxy = 0.0
        used = 0
        for k in range(BINS):
            sel = bin_ == k
            cnt = int(sel.sum())
            if cnt < MIN_BLOCKS_PER_BIN:
                continue
            bm, bv = mean[sel], var[sel]
            order = np.argsort(bv, kind="stable")[: max(cnt // 4, 2)]
            qm, qv = bm[order].mean(), bv[order].mean()
            bias = 1.0 - 1.271 * np.sqrt(2.0 / n)
            qv /= max(bias, 0.5)
            wt = float(cnt)
            sw += wt
            swx += wt * qm
            swy += wt * qv
            swxx += wt * qm * qm
            swxy += wt * qm * qv
            used += 1
        info["bins"][c] = used
        if used == 0:
            continue
        det = sw * swxx - swx * swx
        if used >= 2 and abs(det) > 1e-18:
            ac = (sw * swxy - swx * swy) / det
            bc = (swxx * swy - swx * swxy) / det
        else:
            ac, bc = 0.0, swy / sw
        if ac < 0.0:
            ac, bc = 0.0, swy / sw
        if bc < 0.0:
            bc = 0.0
            ac = swxy / swxx if swxx > 0.0 else 0.0
        a[c], b[c] = ac, bc
    return a, b, info


# ---------------------------------------------------------------------------
# Tiled inference (greycard-ai `Denoiser::run`)

DEFAULT_TILE = 1024
DEFAULT_MARGIN = 96
PLAUSIBLE = 3.0
PLAUSIBLE_RANGE = 0.05
RGGB = np.array([[0, 1], [1, 2]])


def rggb_offset(rgb_idx):
    """(dx, dy) at which the 2x2 pattern reads RGGB, or None."""
    p = np.asarray(rgb_idx)
    for dy in range(2):
        for dx in range(2):
            if all(
                p[(i + dy) % 2, (j + dx) % 2] == RGGB[i, j]
                for i in range(2)
                for j in range(2)
            ):
                return dx, dy
    return None


def mirror_index(i, n):
    """Reflection about the edge samples (keeps the Bayer phase).

    -1 -> 1, n -> n-2.
    """
    period = 2 * (n - 1)
    i = np.mod(i, period)
    return np.where(i >= n, period - i, i)


def tiling(tile, margin):
    """Normalise a tile/margin pair the way greycard does, and check that
    the network takes it."""
    tile += tile & 1
    margin += margin & 1
    tile, margin = max(tile, 32), max(margin, 2)
    side = tile + 2 * margin
    if (side // 2) % 8:
        # The U-Net halves its input three times; other sizes fail in a Concat.
        raise ValueError(f"tile + 2*margin = {side} must be a multiple of 16")
    return tile, margin


def denoise_mosaic(
    x,
    rgb_idx,
    a,
    b,
    run,
    tile=DEFAULT_TILE,
    margin=DEFAULT_MARGIN,
    progress=None,
):
    """Demosaic + denoise a Bayer mosaic.

    `x`: float32 HxW levels (>= 0, any common scale). `a`, `b`: noise model for
    those units, per R/G/B. `run(packed)`: float32 [1,4,s,s] -> [1,3,2s,2s].
    Returns float32 HxWx3 in the same units.
    """
    h, w = x.shape
    if h < 4 or w < 4:
        raise ValueError(f"{w}x{h} is too small")
    off = rggb_offset(rgb_idx)
    if off is None:
        raise ValueError(
            f"not a 2x2 Bayer pattern: {np.asarray(rgb_idx).tolist()}"
        )
    dx, dy = off
    tile, margin = tiling(tile, margin)
    side = tile + 2 * margin
    half = side // 2
    vst = [Vst(a[c], b[c]) for c in range(3)]
    plane_vst = [vst[0], vst[1], vst[1], vst[2]]  # R, G1, G2, B

    out = np.zeros((h, w, 3), np.float32)
    ys_tiles = list(range(-2 * dy, h, tile))
    xs_tiles = list(range(-2 * dx, w, tile))
    total, done = len(ys_tiles) * len(xs_tiles), 0
    for ty in ys_tiles:
        for tx in xs_tiles:
            y_in = ty + dy - margin
            x_in = tx + dx - margin
            ys = mirror_index(y_in + np.arange(side), h)
            xs = mirror_index(x_in + np.arange(side), w)
            win = np.maximum(x[np.ix_(ys, xs)], 0.0)
            packed = np.empty((1, 4, half, half), np.float32)
            for p, (py, px) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                packed[0, p] = plane_vst[p].forward(win[py::2, px::2])
            rgb = np.asarray(run(packed))
            if rgb.shape != (1, 3, side, side):
                raise RuntimeError(
                    f"network returned {rgb.shape}, "
                    f"wanted (1, 3, {side}, {side})"
                )
            rgb = rgb[0]
            _check_plausible(packed[0], rgb, side, margin)
            # Write the core: samples the margin does not cover.
            y0, y1 = max(y_in + margin, 0), min(y_in + side - margin, h)
            x0, x1 = max(x_in + margin, 0), min(x_in + side - margin, w)
            if y0 < y1 and x0 < x1:
                sl = (slice(y0 - y_in, y1 - y_in), slice(x0 - x_in, x1 - x_in))
                for c in range(3):
                    out[y0:y1, x0:x1, c] = vst[c].inverse_clean(rgb[c][sl])
            done += 1
            if progress:
                progress(done, total)
    return out


def _check_plausible(packed, rgb, side, margin):
    """greycard's per-tile sanity check: finite, and not far from the input."""
    core = slice(margin, side - margin)
    sub = rgb[:, core, core]
    if not np.isfinite(sub).all():
        raise RuntimeError("the network produced non-finite values")
    # Input value at each core sample, from its own phase's plane.
    s = side - 2 * margin
    inp = np.empty((s, s), np.float32)
    res = np.empty((s, s), np.float32)
    # margin is even, so core sample (yy, xx) has the phase of
    # (yy-margin, xx-margin).
    for p, (py, px) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        c = (0, 1, 1, 2)[p]
        plane = packed[p][
            margin // 2 : (side - margin) // 2,
            margin // 2 : (side - margin) // 2,
        ]
        inp[py::2, px::2] = plane
        res[py::2, px::2] = sub[c][py::2, px::2]
    mean = float(np.abs(res - inp).mean())
    bound = PLAUSIBLE + PLAUSIBLE_RANGE * float(inp.max() - inp.min())
    if mean > bound:
        raise RuntimeError(
            f"a tile whose answer sits {mean:.1f} stabilised units from its "
            "input"
        )


# ---------------------------------------------------------------------------
# ONNX Runtime backend


def make_runner(model_path, provider="auto", verbose=False, fixed_side=None):
    import onnxruntime as ort

    avail = ort.get_available_providers()
    names = {
        "coreml": "CoreMLExecutionProvider",
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
        "rocm": "ROCMExecutionProvider",
        "directml": "DmlExecutionProvider",
        "openvino": "OpenVINOExecutionProvider",
        "cpu": "CPUExecutionProvider",
    }
    if provider == "auto":
        order = [
            "CUDAExecutionProvider",
            "ROCMExecutionProvider",
            "DmlExecutionProvider",
            "CoreMLExecutionProvider",
            "OpenVINOExecutionProvider",
        ]
        chain = [p for p in order if p in avail]
    else:
        if names.get(provider, provider) not in avail:
            raise SystemExit(f"provider {provider} not available; have {avail}")
        chain = [names.get(provider, provider)]
    chain = [p for p in chain if p != "CPUExecutionProvider"] + [
        "CPUExecutionProvider"
    ]

    def options(p):
        if p == "CoreMLExecutionProvider":
            return {
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": str(
                    cache_dir()
                    / "coreml-cache"
                    / f"{Path(model_path).stem}-{fixed_side or 'dyn'}"
                ),
            }
        return {}

    def session_for(prov):
        so = ort.SessionOptions()
        so.log_severity_level = 0 if verbose else 3
        if fixed_side:
            # Accelerators want static shapes (greycard's CoreML session is
            # fixed too).
            for dim in ("h", "w"):
                so.add_free_dimension_override_by_name(dim, fixed_side)
        return ort.InferenceSession(
            str(model_path), so, providers=[(p, options(p)) for p in prov]
        )

    state = {"prov": chain}
    state["sess"] = session_for(chain)

    def run(packed):
        try:
            return state["sess"].run(["rgb"], {"packed": packed})[0]
        except (
            Exception
        ) as e:  # a GPU provider that cannot run the model: use the CPU
            if state["prov"] == ["CPUExecutionProvider"]:
                raise
            print(
                f"{state['prov'][0]} failed ({str(e).splitlines()[0]}); "
                "falling back to CPU",
                file=sys.stderr,
            )
            state["prov"] = ["CPUExecutionProvider"]
            state["sess"] = session_for(state["prov"])
            return state["sess"].run(["rgb"], {"packed": packed})[0]

    run.state = state
    return run


# ---------------------------------------------------------------------------
# PyTorch backend (optional: `pip install torch onnx onnx2torch`)


def make_torch_runner(model_path, device="auto"):
    """The same network through onnx2torch: CUDA / ROCm / MPS / XPU / CPU by
    one code path."""
    import onnx
    import torch
    from onnx2torch import convert

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
    net = convert(onnx.load(str(model_path))).eval().to(device)

    def run(packed):
        with torch.inference_mode():
            return (
                net(torch.from_numpy(packed).to(device)).float().cpu().numpy()
            )

    run.state = {"prov": [f"torch:{device}"]}
    return run


# ---------------------------------------------------------------------------
# Raw I/O

_DESC_TO_RGB = {"R": 0, "G": 1, "B": 2}


def load_raw(path):
    """-> (levels float32 HxW in 0..1, 2x2 R/G/B index array, info dict)."""
    import rawpy

    with rawpy.imread(str(path)) as raw:
        desc = raw.color_desc.decode()
        if set(desc) - set("RGB") or len(desc) != 4:
            raise SystemExit(
                f"unsupported colour filter {desc!r}: "
                "the model reads RGB Bayer only"
            )
        cfa = np.asarray(raw.raw_colors_visible)
        if cfa.shape[0] < 4 or cfa.shape[1] < 4:
            raise SystemExit("raw image too small")
        idx = np.array(
            [
                [_DESC_TO_RGB[desc[cfa[i, j]]] for j in range(2)]
                for i in range(2)
            ]
        )
        # Verify the 2x2 phase repeats over the whole visible area (Bayer, not
        # X-Trans).
        tiled = np.tile(
            cfa[:2, :2], (cfa.shape[0] // 2 + 1, cfa.shape[1] // 2 + 1)
        )
        if not (tiled[: cfa.shape[0], : cfa.shape[1]] == cfa).all():
            raise SystemExit(
                "not a 2x2 Bayer mosaic (X-Trans is not supported by the model)"
            )
        if rggb_offset(idx) is None:
            raise SystemExit(f"unsupported Bayer arrangement {idx.tolist()}")
        data = np.asarray(raw.raw_image_visible).astype(np.float32)
        black = np.asarray(raw.black_level_per_channel, np.float32)[cfa]
        white = float(raw.white_level)
        # Per-pixel black from its own CFA channel; scale by (white - black of
        # that pixel).
        scale = np.float32(white) - black
        levels = np.maximum((data - black) / scale, 0.0).astype(np.float32)
        wb = np.asarray(raw.camera_whitebalance, np.float64)
        gains = np.ones(3)
        if wb[1] > 0 and wb[0] > 0:
            gains = np.array(
                [wb[desc.index("R")], wb[desc.index("G")], wb[desc.index("B")]]
            )
            gains = gains / gains[1]
        info = {
            "file": os.path.basename(str(path)),
            "size": [int(levels.shape[1]), int(levels.shape[0])],
            "cfa": desc,
            "black": black[:2, :2].tolist(),
            "white": white,
            "flip": int(raw.sizes.flip),
            "camera_wb": gains.tolist(),
            "xyz_cam": np.asarray(raw.rgb_xyz_matrix)[:3].tolist(),
        }
    return levels, idx, info


def apply_flip(rgb, flip):
    if flip == 3:
        return np.rot90(rgb, 2)
    if flip == 5:
        return np.rot90(rgb, 1)
    if flip == 6:
        return np.rot90(rgb, -1)
    return rgb


# ---------------------------------------------------------------------------
# Linear DNG output (DNG 1.4, floating point LinearRaw)

# TIFF field types
_BYTE, _ASCII, _SHORT, _LONG, _RATIONAL, _SRATIONAL = 1, 2, 3, 4, 5, 10
_FMT = {_BYTE: "B", _SHORT: "H", _LONG: "I"}
_LIBRAW_FLIP_TO_TIFF = {0: 1, 3: 3, 5: 8, 6: 6}


def camera_names(path):
    """(make, model) of a raw file, via exiftool if present, else a
    TIFF-based raw's own tags."""
    import shutil
    import subprocess

    if shutil.which("exiftool"):
        try:
            out = subprocess.run(
                ["exiftool", "-j", "-Make", "-Model", str(path)],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            tags = json.loads(out)[0]
            if tags.get("Make") and tags.get("Model"):
                return str(tags["Make"]), str(tags["Model"])
        except (
            subprocess.CalledProcessError,
            ValueError,
            IndexError,
            KeyError,
        ):
            pass
    try:
        import tifffile

        with tifffile.TiffFile(str(path)) as t:
            tg = t.pages[0].tags
            return str(tg["Make"].value), str(tg["Model"].value)
    except Exception:
        return None, None


def _entry(tag, typ, values):
    """A TIFF IFD entry as (tag, type, count, payload bytes)."""
    if typ == _ASCII:
        data = values.encode("ascii", "replace") + b"\0"
        return tag, typ, len(data), data
    if typ in (_RATIONAL, _SRATIONAL):
        fmt = "<II" if typ == _RATIONAL else "<ii"
        return (
            tag,
            typ,
            len(values),
            b"".join(struct.pack(fmt, n, dn) for n, dn in values),
        )
    return (
        tag,
        typ,
        len(values),
        struct.pack("<%d%s" % (len(values), _FMT[typ]), *values),
    )


def _rational(x, den=10000):
    return int(round(x * den)), den


def write_linear_dng(
    path,
    rgb,
    *,
    make,
    model,
    xyz_cam,
    neutral,
    orientation=1,
    software="greycard-denoise-raw",
    bits=32,
):
    """Write an HxWx3 float32 linear image as a LinearRaw DNG: 32-bit float
    (`bits=32`) or 16-bit unsigned integer (`bits=16`, white = 65535,
    negatives clipped).

    `xyz_cam`: 3x3 XYZ(D65) -> camera RGB matrix (DNG ColorMatrix1);
    `neutral`: the camera's response to the as-shot neutral (AsShotNeutral),
    i.e. the reciprocal of the white-balance gains. The data are camera RGB
    in 0..1 with black at 0 and white at 1, so BlackLevel is 0, WhiteLevel 1.
    """
    if bits == 16:
        rgb = np.ascontiguousarray(
            np.rint(np.clip(rgb, 0.0, 1.0) * 65535.0), dtype="<u2"
        )
    else:
        rgb = np.ascontiguousarray(rgb, dtype="<f4")
    h, w, _ = rgb.shape
    unique = (
        f"{make} {model}".strip() if make else (model or "greycard-denoise-raw")
    )
    e = [
        _entry(254, _LONG, [0]),  # NewSubfileType: main image
        _entry(256, _LONG, [w]),
        _entry(257, _LONG, [h]),
        _entry(258, _SHORT, [bits] * 3),
        _entry(259, _SHORT, [1]),  # no compression
        _entry(262, _SHORT, [34892]),  # LinearRaw
        _entry(271, _ASCII, make or ""),
        _entry(272, _ASCII, model or "greycard-denoise-raw"),
        _entry(274, _SHORT, [orientation]),
        _entry(277, _SHORT, [3]),
        _entry(278, _LONG, [h]),  # one strip
        _entry(284, _SHORT, [1]),  # chunky
        _entry(305, _ASCII, software),
        _entry(
            339, _SHORT, [3 if bits == 32 else 1] * 3
        ),  # IEEE float / unsigned int
        _entry(50706, _BYTE, [1, 4, 0, 0]),  # DNGVersion (float needs 1.4)
        _entry(50707, _BYTE, [1, 4, 0, 0]),  # DNGBackwardVersion
        _entry(50708, _ASCII, unique),  # UniqueCameraModel
        _entry(50714, _RATIONAL, [(0, 1)] * 3),  # BlackLevel
        _entry(
            50717, _LONG, [1 if bits == 32 else 65535]
        ),  # WhiteLevel (float data are 0..1)
        _entry(
            50721,
            _SRATIONAL,
            [_rational(v) for v in np.asarray(xyz_cam).reshape(9)],
        ),
        _entry(
            50728, _RATIONAL, [_rational(v, 1000000) for v in neutral]
        ),  # AsShotNeutral
        _entry(50778, _SHORT, [21]),  # CalibrationIlluminant1 = D65
    ]
    # StripOffsets / StripByteCounts need the final layout: IFD, then the
    # out-of-line values, then the pixel data (4-aligned).
    e += [
        (273, _LONG, 1, b"\0\0\0\0"),
        (279, _LONG, 1, struct.pack("<I", rgb.nbytes)),
    ]
    e.sort(key=lambda t: t[0])
    ifd_size = 2 + 12 * len(e) + 4
    extra = sum((len(d) + 1) & ~1 for _, _, _, d in e if len(d) > 4)
    data_off = (8 + ifd_size + extra + 3) & ~3
    out = bytearray(b"II*\0" + struct.pack("<I", 8) + struct.pack("<H", len(e)))
    spill = bytearray()
    spill_base = 8 + ifd_size
    for tag, typ, count, data in e:
        if tag == 273:
            data = struct.pack("<I", data_off)
        if len(data) <= 4:
            out += struct.pack("<HHI", tag, typ, count) + data.ljust(4, b"\0")
        else:
            out += struct.pack(
                "<HHII", tag, typ, count, spill_base + len(spill)
            )
            spill += data + (b"\0" if len(data) & 1 else b"")
    out += struct.pack("<I", 0)
    out += spill
    out += b"\0" * (data_off - len(out))
    with open(path, "wb") as f:
        f.write(out)
        rgb.tofile(f)


def copy_exif(src, dst):
    """Copy the shooting metadata (exposure, ISO, lens, date...) from the raw
    to the DNG."""
    import shutil
    import subprocess

    if not shutil.which("exiftool"):
        return False
    r = subprocess.run(
        [
            "exiftool",
            "-q",
            "-overwrite_original",
            "-TagsFromFile",
            str(src),
            "-ExifIFD:all",
            "-GPS:all",
            "-IFD0:Artist",
            "-IFD0:Copyright",
            "-IFD0:DateTime",
            "-x",
            "ExifIFD:MakerNotes",
            "-x",
            "ExifIFD:MakerNote",
            "-x",
            "ExifIFD:ComponentsConfiguration",
            "-x",
            "ExifIFD:ExifImageWidth",
            "-x",
            "ExifIFD:ExifImageHeight",
            "-x",
            "ExifIFD:PixelXDimension",
            "-x",
            "ExifIFD:PixelYDimension",
            str(dst),
        ],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


# ---------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("raw", help="input raw file (Bayer)")
    ap.add_argument("-o", "--output", help="output .tif (default: <raw>.tif)")
    ap.add_argument(
        "-m",
        "--model",
        default="balanced",
        choices=sorted(MODELS),
        help="tier to use (default balanced); or --model-file",
    )
    ap.add_argument("--model-file", help="use this ONNX file instead of a tier")
    ap.add_argument(
        "--backend",
        choices=["onnxruntime", "torch"],
        default="onnxruntime",
        help="onnxruntime (default; CoreML/CUDA/DirectML/CPU) or torch "
        "(cuda/mps/xpu/cpu)",
    )
    ap.add_argument(
        "--provider",
        default="auto",
        help="onnxruntime: auto, cpu, coreml, cuda, tensorrt, rocm, directml, "
        "openvino; "
        "torch: auto, cpu, cuda, mps, xpu",
    )
    ap.add_argument(
        "--noise",
        metavar="aR,bR,aG,bG,aB,bB",
        help="noise model (levels 0..1) instead of estimating it",
    )
    ap.add_argument(
        "-f",
        "--format",
        choices=["tiff", "dng"],
        help="output format; default from the -o extension, else tiff. dng "
        "writes a LinearRaw DNG (camera matrix, as-shot neutral, Make/Model) "
        "that raw converters treat as a raw file, so camera profiles apply; "
        "sensor orientation is kept and flagged, and --wb is ignored",
    )
    ap.add_argument(
        "--dng-bits",
        type=int,
        choices=[32, 16],
        default=16,
        help="dng sample type: 16 = unsigned integer (default; clips "
        "negatives, quantises to 1/65535); 32 = IEEE float (DNG 1.4). "
        "NOTE: ART's float-DNG loader only reads single-plane (CFA) float, "
        "so 32 is mis-rendered by ART",
    )
    ap.add_argument(
        "--wb",
        choices=["none", "camera"],
        default="none",
        help="multiply the result by the as-shot gains (G=1); default none",
    )
    ap.add_argument(
        "--no-rotate", action="store_true", help="keep sensor orientation"
    )
    ap.add_argument("--tile", type=int, default=DEFAULT_TILE)
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument(
        "--progress",
        action="store_true",
        help="print integer percentages, one per line, on stdout "
        "(ART-dialogs --progress)",
    )
    args = ap.parse_args(argv)
    try:
        args.tile, args.margin = tiling(args.tile, args.margin)
    except ValueError as e:
        ap.error(str(e))

    import tifffile

    last = [-1]

    def report(pct):
        if args.progress and int(pct) > last[0]:
            last[0] = int(pct)
            print(last[0], flush=True)

    fmt = args.format or (
        "dng"
        if args.output and args.output.lower().endswith(".dng")
        else "tiff"
    )
    out = (
        Path(args.output)
        if args.output
        else Path(args.raw).with_suffix(".dng" if fmt == "dng" else ".tif")
    )
    t0 = time.time()
    levels, idx, info = load_raw(args.raw)
    t_load = time.time() - t0
    report(3)

    if args.noise:
        v = [float(s) for s in args.noise.split(",")]
        if len(v) != 6:
            raise SystemExit("--noise wants six numbers")
        a, b, ninfo = np.array(v[0::2]), np.array(v[1::2]), {"given": True}
    else:
        a, b, ninfo = estimate_noise(levels, idx)
    report(6)
    if args.verbose:
        print(f"noise a={a.tolist()} b={b.tolist()} {ninfo}", file=sys.stderr)

    model_path = (
        Path(args.model_file) if args.model_file else fetch_model(args.model)
    )
    if args.backend == "torch":
        run = make_torch_runner(model_path, args.provider)
    else:
        run = make_runner(
            model_path,
            args.provider,
            args.verbose,
            (args.tile + 2 * args.margin) // 2,
        )
    print(
        f"{info['file']} {info['size'][0]}x{info['size'][1]}  "
        f"model {model_path.name}",
        file=sys.stderr,
    )
    report(10)

    def progress(i, n):
        if args.progress:
            report(10 + 88 * i / n)
        else:
            print(f"\r  tile {i}/{n}", end="", file=sys.stderr, flush=True)

    t1 = time.time()
    rgb = denoise_mosaic(
        levels, idx, a, b, run, args.tile, args.margin, progress
    )
    print(
        f"  {time.time() - t1:.1f} s on {run.state['prov'][0]}", file=sys.stderr
    )

    if fmt == "dng":
        make, model = camera_names(args.raw)
        if not make:
            print(
                "warning: camera Make/Model unknown; raw converters cannot "
                "pick a camera profile",
                file=sys.stderr,
            )
        write_linear_dng(
            out,
            rgb,
            make=make,
            model=model,
            xyz_cam=info["xyz_cam"],
            neutral=1.0 / np.asarray(info["camera_wb"]),
            orientation=_LIBRAW_FLIP_TO_TIFF.get(info["flip"], 1),
            bits=args.dng_bits,
        )
        if not copy_exif(args.raw, out):
            print(
                "note: exiftool not available or failed; exposure/lens "
                "metadata not copied",
                file=sys.stderr,
            )
        print(f"wrote {out}  (total {time.time() - t0:.1f} s)", file=sys.stderr)
        report(100)
        return
    if args.wb == "camera":
        rgb = rgb * np.asarray(info["camera_wb"], np.float32)
    if not args.no_rotate:
        rgb = apply_flip(rgb, info["flip"])
    meta = {
        "source": info["file"],
        "model": model_path.name,
        "noise_a": a.tolist(),
        "noise_b": b.tolist(),
        "wb_applied": args.wb == "camera",
        "levels": "linear, black removed, divided by (white - black); "
        "camera RGB",
    }
    tifffile.imwrite(
        out,
        np.ascontiguousarray(rgb, dtype=np.float32),
        photometric="rgb",
        description=json.dumps(meta),
        software="greycard-denoise-raw",
    )
    print(
        f"wrote {out}  (load {t_load:.1f} s, total {time.time() - t0:.1f} s)",
        file=sys.stderr,
    )
    report(100)


if __name__ == "__main__":
    main()
