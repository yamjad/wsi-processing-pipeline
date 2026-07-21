"""
Stream D - Color / stain / intensity processing for H&E tiles.

Milestone M4: tiles normalized to a common stain reference, with the transform
toolkit implemented and understood.

Orienting fact: in H&E, hematoxylin stains nuclei blue-purple, eosin stains
cytoplasm pink. This stream measures and standardizes those two stains.

Contents
  D1  optical_density / od_to_rgb        - the OD space stain math lives in
  D2  deconvolve_hed, adaptive note      - separate H & E channels
  D3  StainNormalizer (staintools wrap)  - reproduce Panoptes normalization
  D4  to_hsv/to_lab, channel_histograms, - color-space QC readout
      qc_readout

Figures are produced by the plot_* helpers so the module stays importable without
a display. All functions take/return uint8 RGB arrays unless noted.
"""

import numpy as np
import torchstain

def optical_density(rgb, i0=255.0, eps=1e-6):
    """Convert an RGB image (uint8 or float) to optical density.

    OD = -log10(I / I0), from Beer-Lambert: stain *concentration* is linear in OD,
    not in transmitted intensity (RGB). I0 is the illumination of blank slide
    (255 for 8-bit; or an estimated per-slide white point).

    Guards: intensities are clipped to [eps, i0] before the log so that I=0 does
    not produce +inf and pixels brighter than the white point do not go negative.
    Returns float OD with the same H x W x 3 shape (OD >= 0).
    """
    I = np.asarray(rgb, dtype=np.float64)
    I = np.clip(I, eps, i0)                 # avoid log(0) and I > I0
    return -np.log10(I / i0)


def od_to_rgb(od, i0=255.0):
    """Inverse of optical_density: I = I0 * 10**(-OD). Returns uint8 RGB."""
    I = i0 * np.power(10.0, -np.asarray(od, dtype=np.float64))
    return np.clip(I, 0, 255).astype(np.uint8)


def deconvolve_hed(rgb):
    """Separate an RGB H&E tile into H, E, DAB channels with a FIXED stain matrix.

    Uses skimage.color.rgb2hed (Ruifrok & Johnston fixed vectors). Returns the
    three float channels (hematoxylin, eosin, residual/DAB). Fixed vectors are
    fast and deterministic but do not adapt to scanner/lab color shifts - which is
    exactly why Panoptes commits to adaptive Vahadane in D3.
    """
    from skimage.color import rgb2hed
    rgb = _as_float01(rgb)
    hed = rgb2hed(rgb)
    return hed[:, :, 0], hed[:, :, 1], hed[:, :, 2]


class StainNormalizer:
    """Adaptive stain normalization, reproducing Panoptes Slicer.normalization().

    Panoptes used staintools' Vahadane, which depends on SPAMS - painful to build on
    modern Python. This wrapper prefers `torchstain` (maintained, no SPAMS, py3.13)
    and falls back to staintools only if torchstain is absent. Same interface either
    way; the reference image is swappable for Analysis A5.

        norm = StainNormalizer(method="macenko").fit_from_path("colorstandard.png")
        out  = norm.transform(tile)          # uint8 RGB in, uint8 RGB out

    Backends:
      method="macenko"  -> torchstain numpy backend (adaptive; needs only numpy)
      method="vahadane" -> torchstain torch backend if available (needs torch), else
                           staintools (needs SPAMS). Panoptes' exact choice.

    Note: Macenko and Vahadane are both *adaptive* (image-estimated) stain matrices,
    so the D3 point (adaptive > fixed rgb2hed across scanners) holds for either.

    Unlike Panoptes (which re-fit per tile), fit once and reuse - fit is the cost.
    """

    def __init__(self, method="macenko", backend="numpy", reference_path=None):
        self.method = method
        self.backend = backend
        self._fitted = False
        self._kind = None
        try:
            if method == "macenko":
                self._impl = torchstain.normalizers.MacenkoNormalizer(backend=backend)
            elif method == "vahadane":
                # Vahadane in torchstain needs the torch backend
                self._impl = torchstain.normalizers.VahadaneNormalizer(backend="torch" if backend == "numpy" else backend)
                self.backend = "torch"
            else:
                raise ValueError("method must be 'macenko' or 'vahadane'")
            self._kind = "torchstain"
        except Exception:
            import staintools
            self._st = staintools
            self._lum = staintools.LuminosityStandardizer
            self._impl = staintools.StainNormalizer(
                method="vahadane" if method == "vahadane" else "macenko")
            self._kind = "staintools"
        if reference_path:
            self.fit_from_path(reference_path)

    def _to_backend(self, rgb):
        """uint8 HxWxC RGB -> the array/tensor form the active backend expects."""
        arr = np.asarray(rgb, dtype=np.uint8)
        if self._kind == "torchstain" and self.backend == "torch":
            import torch
            return torch.from_numpy(arr).permute(2, 0, 1).float()   # CxHxW, 0-255
        return arr                                                   # HxWxC, 0-255

    def fit(self, reference_rgb):
        if self._kind == "torchstain":
            self._impl.fit(self._to_backend(reference_rgb))
        else:
            self._impl.fit(self._lum.standardize(np.asarray(reference_rgb, np.uint8)))
        self._fitted = True
        return self

    def fit_from_path(self, path):
        from PIL import Image
        return self.fit(np.array(Image.open(path).convert("RGB")))

    def transform(self, rgb):
        if not self._fitted:
            raise RuntimeError("call fit() / fit_from_path() before transform()")
        if self._kind == "torchstain":
            out = self._impl.normalize(I=self._to_backend(rgb), stains=False)
            norm = out[0] if isinstance(out, (tuple, list)) else out
            arr = norm.numpy() if hasattr(norm, "numpy") else np.asarray(norm)
            return np.clip(arr, 0, 255).astype(np.uint8)
        img = self._lum.standardize(np.asarray(rgb, np.uint8))
        return self._impl.transform(img)


def to_hsv(rgb):
    """RGB -> HSV float. HSV separates hue (stain identity) from brightness."""
    from skimage.color import rgb2hsv
    return rgb2hsv(_as_float01(rgb))


def to_lab(rgb):
    """RGB -> CIELAB float. LAB is perceptually uniform (good for QC)."""
    from skimage.color import rgb2lab
    return rgb2lab(_as_float01(rgb))


def channel_histograms(img, bins=64, value_range=None):
    """Per-channel histograms of an H x W x C image. Returns (counts, edges) per ch."""
    img = np.asarray(img, dtype=np.float64)
    out = []
    for c in range(img.shape[2]):
        rng = value_range[c] if value_range else (img[:, :, c].min(), img[:, :, c].max())
        counts, edges = np.histogram(img[:, :, c].ravel(), bins=bins, range=rng)
        out.append((counts, edges))
    return out


def qc_readout(rgb, white_thr=220, dark_thr=35, clip_frac=0.02):
    """One-tile QC summary: well-stained? over/under? clipped?

    Heuristics on OD and intensity:
      - mean_od          : overall stain amount (higher = darker/more stain)
      - h_frac / e_frac  : share of tissue pixels dominated by H vs E (balance)
      - frac_white       : blank/background share (under-stained if very high)
      - clipped_low/high : fraction of pixels pinned at 0 / 255 (sensor clipping)
    Returns a dict; 'flags' lists any triggered warnings.
    """
    rgb = np.asarray(rgb, dtype=np.uint8)
    od = optical_density(rgb)
    mean_od = float(od.mean())
    gray = rgb.mean(axis=2)

    frac_white = float((gray > white_thr).mean())
    frac_dark = float((gray < dark_thr).mean())
    clipped_low = float((rgb == 0).all(axis=2).mean())
    clipped_high = float((rgb == 255).all(axis=2).mean())

    # tissue = non-background; measure H vs E dominance there via deconvolution
    tissue = gray <= white_thr
    flags = []
    h_frac = e_frac = float("nan")
    if tissue.sum() > 0:
        try:
            H, E, _ = deconvolve_hed(rgb)
            h_dom = (H > E) & tissue
            e_dom = (E >= H) & tissue
            h_frac = float(h_dom.sum() / tissue.sum())
            e_frac = float(e_dom.sum() / tissue.sum())
        except Exception:
            pass

    if frac_white > 0.85:
        flags.append("under-stained / mostly background")
    if mean_od > 0.9:
        flags.append("possibly over-stained (very high OD)")
    if clipped_low > clip_frac:
        flags.append("dark clipping (pixels at 0)")
    if clipped_high > clip_frac:
        flags.append("bright clipping (pixels at 255)")

    return {"mean_od": mean_od, "h_frac": h_frac, "e_frac": e_frac,
            "frac_white": frac_white, "frac_dark": frac_dark,
            "clipped_low": clipped_low, "clipped_high": clipped_high,
            "flags": flags or ["looks ok"]}


def _as_float01(rgb):
    arr = np.asarray(rgb)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    return np.clip(arr.astype(np.float64), 0, 1)