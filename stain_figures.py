"""
Stream D figure helpers - one function per task deliverable.

Kept separate from stain_wsidicom.py so the transforms stay importable without a
display. Each returns a matplotlib Figure (also saves if out_path given).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import stain_wsidicom as S


def plot_d1_od(rgb, out_path=None, bins=64):
    od = S.optical_density(rgb)
    recon = S.od_to_rgb(od)

    fig, ax = plt.subplots(2, 3, figsize=(12, 7))
    ax[0, 0].imshow(rgb);                 ax[0, 0].set_title("original RGB");        ax[0, 0].axis("off")
    ax[0, 1].imshow(od.mean(2), cmap="magma"); ax[0, 1].set_title("OD (mean over channels)"); ax[0, 1].axis("off")
    ax[0, 2].imshow(recon);               ax[0, 2].set_title("reconstructed (od_to_rgb)"); ax[0, 2].axis("off")

    colors = ("r", "g", "b")
    for c, col in enumerate(colors):
        ax[1, 0].hist(rgb[:, :, c].ravel(), bins=bins, range=(0, 255),
                      color=col, alpha=0.5, label=col.upper())
    ax[1, 0].set_title("RGB intensity histogram"); ax[1, 0].set_xlabel("intensity 0-255"); ax[1, 0].legend()

    for c, col in enumerate(colors):
        ax[1, 1].hist(od[:, :, c].ravel(), bins=bins, color=col, alpha=0.5, label=col.upper())
    ax[1, 1].set_title("OD histogram"); ax[1, 1].set_xlabel("optical density"); ax[1, 1].legend()

    # scatter showing the log relationship between intensity and OD
    samp = np.random.default_rng(0).choice(rgb[:, :, 0].size, size=min(4000, rgb[:, :, 0].size), replace=False)
    ax[1, 2].scatter(rgb[:, :, 0].ravel()[samp], od[:, :, 0].ravel()[samp], s=2, alpha=0.3)
    ax[1, 2].set_title("intensity vs OD (nonlinear)"); ax[1, 2].set_xlabel("R intensity"); ax[1, 2].set_ylabel("R OD")

    fig.tight_layout()
    if out_path: fig.savefig(out_path, dpi=120)
    return fig


def plot_d2_hed(rgb, out_path=None):
    H, E, D = S.deconvolve_hed(rgb)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(rgb);              ax[0].set_title("original");                 ax[0].axis("off")
    ax[1].imshow(H, cmap="magma");  ax[1].set_title("Hematoxylin (nuclei)");     ax[1].axis("off")
    ax[2].imshow(E, cmap="magma");  ax[2].set_title("Eosin (cytoplasm)");        ax[2].axis("off")
    fig.suptitle("Fixed-matrix deconvolution (rgb2hed) - compare to adaptive Vahadane in D3")
    fig.tight_layout()
    if out_path: fig.savefig(out_path, dpi=120)
    return fig


def plot_d3_normalization(tiles_by_slide, normalizer, out_path=None):
    """tiles_by_slide: dict {slide_name: rgb_tile}. Shows raw (top) vs normalized (bottom)."""
    names = list(tiles_by_slide)
    n = len(names)
    fig, ax = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1: ax = ax.reshape(2, 1)
    for j, name in enumerate(names):
        raw = tiles_by_slide[name]
        ax[0, j].imshow(raw); ax[0, j].set_title(name, fontsize=9); ax[0, j].axis("off")
        try:
            norm = normalizer.transform(raw)
        except Exception as e:
            norm = raw; ax[1, j].set_title("(norm failed)", fontsize=8)
        ax[1, j].imshow(norm); ax[1, j].axis("off")
    ax[0, 0].set_ylabel("raw", fontsize=11)
    ax[1, 0].set_ylabel("normalized", fontsize=11)
    fig.suptitle("Stain normalization to one reference (raw top / normalized bottom)")
    fig.tight_layout()
    if out_path: fig.savefig(out_path, dpi=120)
    return fig


def plot_d4_qc(rgb, out_path=None, bins=64):
    hsv = S.to_hsv(rgb)
    lab = S.to_lab(rgb)
    qc = S.qc_readout(rgb)

    fig, ax = plt.subplots(3, 4, figsize=(15, 10))
    spaces = [("RGB", rgb, ("R", "G", "B"), (0, 255)),
              ("HSV", hsv, ("H", "S", "V"), (0, 1)),
              ("LAB", lab, ("L", "a", "b"), None)]
    for row, (nm, img, chans, rng) in enumerate(spaces):
        # show a representative view in col 0
        view = img if nm == "RGB" else (img / [1, 1, 1] if nm == "HSV" else None)
        if nm == "RGB":
            ax[row, 0].imshow(rgb)
        elif nm == "HSV":
            ax[row, 0].imshow(hsv[:, :, 0], cmap="hsv")
        else:
            ax[row, 0].imshow(lab[:, :, 0], cmap="gray")
        ax[row, 0].set_title(nm + " view"); ax[row, 0].axis("off")
        for c in range(3):
            a = ax[row, c + 1]
            data = np.asarray(img)[:, :, c].ravel()
            a.hist(data, bins=bins, range=rng if rng else (data.min(), data.max()), color="0.3")
            a.set_title("{} {}".format(nm, chans[c]), fontsize=9)

    flag_txt = "QC: mean_OD={:.3f}  white={:.0%}  clipHi={:.1%}  ->  {}".format(
        qc["mean_od"], qc["frac_white"], qc["clipped_high"], "; ".join(qc["flags"]))
    fig.suptitle(flag_txt, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    if out_path: fig.savefig(out_path, dpi=120)
    return fig
