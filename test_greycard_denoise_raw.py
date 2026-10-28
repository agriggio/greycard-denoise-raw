import numpy as np
import pytest

import greycard_denoise_raw as d

RGGB = np.array([[0, 1], [1, 2]])


def test_vst_round_trip():
    a = [3e-4, 1e-2, 0.0]
    b = [5e-8, 1e-4, 1e-4]
    x = np.linspace(0, 1.2, 1000, dtype=np.float32)
    for ai, bi in zip(a, b):
        v = d.Vst(ai, bi)
        np.testing.assert_allclose(v.inverse_clean(v.forward(x)), x, atol=1e-6)


def test_vst_gives_unit_noise():
    rng = np.random.default_rng(0)
    a, b = 1e-3, 1e-5
    v = d.Vst(a, b)
    for level in (0.01, 0.1, 0.5):
        noisy = a * rng.poisson(level / a, 1_000_000) + np.sqrt(
            b
        ) * rng.standard_normal(1_000_000)
        assert abs(v.forward(noisy.astype(np.float32)).std() - 1.0) < 0.03


def test_silent_channel_is_finite():
    v = d.Vst(0.0, 0.0)
    assert np.isfinite(v.forward(np.float32([0, 0.5, 1]))).all()


def bayer(h, w, pattern, rng, a, b, ramp=True):
    """A noisy mosaic of a smooth scene with Poisson-Gaussian noise (a, b per
    colour)."""
    yy, xx = np.mgrid[0:h, 0:w]
    clean = (
        (0.02 + 0.8 * (0.5 + 0.5 * np.sin(xx / 400.0) * np.cos(yy / 300.0)))
        if ramp
        else 0.3 + 0 * xx
    )
    cmap = np.tile(np.asarray(pattern), (h // 2 + 1, w // 2 + 1))[:h, :w]
    aa, bb = np.asarray(a)[cmap], np.asarray(b)[cmap]
    noisy = aa * rng.poisson(clean / aa) + np.sqrt(bb) * rng.standard_normal(
        (h, w)
    )
    return clean.astype(np.float32), np.clip(noisy, 0, 0.97).astype(np.float32)


def test_noise_estimate_recovers_model():
    rng = np.random.default_rng(1)
    a_true, b_true = [4e-4, 3e-4, 5e-4], [3e-6, 2e-6, 4e-6]
    _, x = bayer(1600, 2400, RGGB, rng, a_true, b_true)
    a, b, info = d.estimate_noise(x, RGGB)
    # The model is a line fit through noisy bins: accept 30% on a, and the
    # variance at mid-grey (what matters) to 15%.
    for c in range(3):
        assert a[c] == pytest.approx(a_true[c], rel=0.3)
        v_true = a_true[c] * 0.18 + b_true[c]
        assert a[c] * 0.18 + b[c] == pytest.approx(v_true, rel=0.15)


def test_noise_estimate_is_phase_independent():
    rng = np.random.default_rng(2)
    a_true, b_true = [4e-4, 3e-4, 5e-4], [3e-6, 2e-6, 4e-6]
    pat = np.array([[2, 1], [1, 0]])  # BGGR
    _, x = bayer(800, 1200, pat, rng, a_true, b_true)
    a, b, _ = d.estimate_noise(x, pat)
    assert a[0] == pytest.approx(a_true[0], rel=0.4)
    assert a[2] == pytest.approx(a_true[2], rel=0.4)


def test_clipped_blocks_are_left_out():
    x = np.full((200, 200), 0.99, np.float32)
    a, b, info = d.estimate_noise(x, RGGB)
    assert info["blocks"] == [0, 0, 0] and not a.any() and not b.any()


@pytest.mark.parametrize(
    "pat,off",
    [
        ([[0, 1], [1, 2]], (0, 0)),
        ([[1, 0], [2, 1]], (1, 0)),
        ([[1, 2], [0, 1]], (0, 1)),
        ([[2, 1], [1, 0]], (1, 1)),
    ],
)
def test_rggb_offset_finds_each_phase(pat, off):
    assert d.rggb_offset(np.array(pat)) == off


def test_rggb_offset_rejects_other_patterns():
    assert d.rggb_offset(np.array([[0, 1], [0, 1]])) is None


def test_mirror_keeps_the_phase():
    i = np.arange(-4, 14)
    m = d.mirror_index(i, 10)
    assert m.tolist() == [4, 3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5]
    assert ((m & 1) == (i & 1)).all()


def fake_net(packed):
    """Stand-in for the network: each plane is smoothed by a 5x5-wide box and
    nearest-upsampled to the full-res RGB, G averaged -- a function with a
    finite receptive field, so a margin that covers it must give no seams."""
    _, _, s, _ = packed.shape
    p = packed[0]
    k = 2
    pad = np.pad(p, ((0, 0), (k, k), (k, k)), mode="edge")
    sm = (
        sum(
            pad[:, i : i + s, j : j + s]
            for i in range(2 * k + 1)
            for j in range(2 * k + 1)
        )
        / 25.0
    )
    rgb = np.stack([sm[0], 0.5 * (sm[1] + sm[2]), sm[3]])
    return np.repeat(np.repeat(rgb, 2, axis=1), 2, axis=2)[None]


@pytest.mark.parametrize(
    "pat",
    [[[0, 1], [1, 2]], [[1, 0], [2, 1]], [[1, 2], [0, 1]], [[2, 1], [1, 0]]],
)
def test_tiling_has_no_seams_and_covers_every_pixel(pat):
    rng = np.random.default_rng(3)
    h, w = 203, 261  # odd sizes on purpose
    _, x = bayer(h, w, pat, rng, [4e-4] * 3, [3e-6] * 3)
    a = np.full(3, 4e-4)
    b = np.full(3, 3e-6)
    one = d.denoise_mosaic(
        x, pat, a, b, fake_net, tile=1024, margin=96
    )  # one tile
    many = d.denoise_mosaic(
        x, pat, a, b, fake_net, tile=64, margin=16
    )  # many tiles
    assert (one != 0).all()
    # The fake net's reach is 2 packed samples = 4 mosaic samples << margin.
    np.testing.assert_allclose(many, one, atol=1e-5)


@pytest.mark.parametrize(
    "pat",
    [[[0, 1], [1, 2]], [[1, 0], [2, 1]], [[1, 2], [0, 1]], [[2, 1], [1, 0]]],
)
def test_network_always_sees_rggb(pat):
    """Light only the sensor's red sites: whatever the sensor's phase, the
    network's R plane (0) must carry the signal and the others stay dark."""
    pat = np.array(pat)
    h = w = 128
    cmap = np.tile(pat, (h // 2, w // 2))
    x = np.where(cmap == 0, 0.5, 0.0).astype(np.float32)
    seen = []

    def spy(packed):
        seen.append(packed.copy())
        return fake_net(packed)

    d.denoise_mosaic(
        x, pat, np.full(3, 4e-4), np.full(3, 3e-6), spy, tile=64, margin=16
    )
    assert seen
    for packed in seen:
        planes = packed[0]
        assert planes[0].min() > 0.1  # R plane: all lit
        assert planes[1:].max() == 0.0  # G1, G2, B: dark


def test_window_size_must_suit_the_network():
    x = np.zeros((64, 64), np.float32)
    with pytest.raises(ValueError, match="multiple of 16"):
        d.denoise_mosaic(
            x, RGGB, np.ones(3), np.ones(3), fake_net, tile=64, margin=2
        )


XYZ_CAM = [
    [0.746, -0.2365, -0.0588],
    [-0.5687, 1.3442, 0.2474],
    [-0.0624, 0.1156, 0.6584],
]


@pytest.mark.parametrize("bits", [32, 16])
def test_linear_dng_round_trips(tmp_path, bits):
    import tifffile

    rng = np.random.default_rng(5)
    rgb = rng.random((37, 53, 3)).astype(np.float32)
    path = tmp_path / "x.dng"
    d.write_linear_dng(
        path,
        rgb,
        make="SONY",
        model="ILCE-7M4",
        xyz_cam=XYZ_CAM,
        neutral=[0.42, 1.0, 0.6],
        orientation=8,
        bits=bits,
    )
    with tifffile.TiffFile(path) as t:
        tags = t.pages[0].tags
        assert (
            tags["Make"].value == "SONY" and tags["Model"].value == "ILCE-7M4"
        )
        assert tags["PhotometricInterpretation"].value == 34892
        assert tags["Orientation"].value == 8
        assert tuple(tags["BitsPerSample"].value) == (bits,) * 3
        assert tuple(tags["DNGVersion"].value) == (1, 4, 0, 0)
        assert tags["UniqueCameraModel"].value == "SONY ILCE-7M4"
        assert tags["WhiteLevel"].value == (1 if bits == 32 else 65535)
        cm = np.array(tags["ColorMatrix1"].value, float).reshape(-1, 2)
        np.testing.assert_allclose(
            cm[:, 0] / cm[:, 1], np.ravel(XYZ_CAM), atol=1e-4
        )
        an = np.array(tags["AsShotNeutral"].value, float).reshape(-1, 2)
        np.testing.assert_allclose(
            an[:, 0] / an[:, 1], [0.42, 1.0, 0.6], atol=1e-6
        )
        data = t.pages[0].asarray()
    assert data.shape == rgb.shape
    if bits == 32:
        assert data.dtype == np.float32
        np.testing.assert_array_equal(data, rgb)
    else:
        assert data.dtype == np.uint16
        np.testing.assert_allclose(data / 65535.0, rgb, atol=1 / 65535)


def test_linear_dng_16bit_clips_out_of_range(tmp_path):
    import tifffile

    rgb = np.array([[[-0.1, 0.5, 1.2]]], np.float32)
    d.write_linear_dng(
        tmp_path / "x.dng",
        rgb,
        make="A",
        model="B",
        xyz_cam=XYZ_CAM,
        neutral=[1, 1, 1],
        bits=16,
    )
    assert tifffile.imread(tmp_path / "x.dng").ravel().tolist() == [
        0,
        32768,
        65535,
    ]
