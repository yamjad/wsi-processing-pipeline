import os
import traceback

from wsidicom import WsiDicom
from slicer_wsidicom import tile

import re
import glob
import pandas as pd


def available_downsamples(slide_path):
    """Return [(list_index, downsample)] for the slide's available levels."""
    with WsiDicom.open(slide_path) as slide:
        return [(i, 2 ** lvl.level) for i, lvl in enumerate(slide.levels)]


def build_ladder(slide_path, targets=(2, 4, 8)):
    """Map tier m -> (list_index, ft) achieving each target effective downsample.

    For each target, choose the coarsest available level whose downsample divides
    the target evenly (minimises the read window), then ft = target // downsample.
    Raises if a target can't be reached with an integer ft from any level.
    """
    levels = available_downsamples(slide_path)      # [(idx, ds), ...] ds ascending
    ladder = {}
    for m, t in enumerate(targets, start=1):
        candidates = [(idx, ds) for idx, ds in levels if ds <= t and t % ds == 0]
        if not candidates:
            raise ValueError(
                "no available level reaches effective {}x with an integer ft; "
                "available downsamples = {}".format(t, [ds for _, ds in levels]))
        idx, ds = max(candidates, key=lambda p: p[1])   # coarsest that fits
        ladder[m] = (idx, t // ds)
    return ladder


def cutter(slide_path, outdirr, dp=None, targets=(2, 4, 8),
           std_path=None, parallel=False,
           white_thr=200, black_thr=50, blank_max=0.8,
           edge="pad", pad_color=(255, 255, 255)):
    """Cut one slide into len(targets) magnifications under outdirr/level1, level2, ...

    targets    : effective downsample factors relative to full resolution.
                 (2, 4, 8) reproduces Panoptes' 20x branch (10x/5x/2.5x if the slide
                 is natively 20x).
    slide_path : folder holding the DICOM instances for one slide.
    std_path   : optional stain reference; reserved for Stream D (tile() ignores
                 std_img for now), loaded only if provided.

    Returns {m: {"list_level","downsample","ft","eff","n_x","n_y","count","outdir"}}
    for the tiers that succeeded. A failed tier is reported with its traceback and
    skipped.
    """
    os.makedirs(outdirr, exist_ok=True)

    std = None
    if std_path is not None:
        import staintools
        std = staintools.LuminosityStandardizer.standardize(
            staintools.read_image(std_path))

    ladder = build_ladder(slide_path, targets)
    avail = dict(available_downsamples(slide_path))
    print("available downsamples (list index -> factor):", avail)
    print("ladder:", ", ".join(
        "level{}=(list {}, ft {}, eff {}x)".format(m, idx, ft, avail[idx] * ft)
        for m, (idx, ft) in ladder.items()))

    results = {}
    for m, (list_level, ft) in ladder.items():
        ds = avail[list_level]
        eff = ds * ft
        otdir = "{}/level{}".format(outdirr, m)
        os.makedirs(otdir, exist_ok=True)
        print("\nlevel{} -> list level {} (downsample {}x), ft {}, effective {}x  {}"
              .format(m, list_level, ds, ft, eff, otdir))
        try:
            n_x, n_y, raw, ct = tile(
                slide_path, otdir, list_level, std_img=std, ft=ft, dp=dp,
                white_thr=white_thr, black_thr=black_thr, blank_max=blank_max,
                parallel=parallel, edge=edge, pad_color=pad_color)
            results[m] = {"list_level": list_level, "downsample": ds, "ft": ft,
                          "eff": eff, "n_x": n_x, "n_y": n_y, "count": ct,
                          "outdir": otdir}
        except Exception:
            print("  !! level{} failed - skipping this tier:".format(m))
            traceback.print_exc()

    print("\ndone. tiles per tier:", {m: r["count"] for m, r in results.items()})
    return results
 
 
def _ids_from_dir(level_dir, fac):
    """Parse (x, y, path) from tile filenames in a level folder, quantized by fac.
 
    Filenames look like region_x-{X}-y-{Y}.png or region_x-{X}-y-{Y}_{dp}.png; X, Y
    are level-0 coordinates. Quantizing by fac maps every tile to a coarse grid cell
    so that fine tiles and the coarse tile covering them share an (x, y) index.
    """
    rows = []
    for p in sorted(glob.glob(os.path.join(level_dir, "region_x-*-y-*.png"))):
        m = re.search(r"x-(\d+)-y-(\d+)", os.path.basename(p))
        if m:
            X, Y = int(m.group(1)), int(m.group(2))
            rows.append((X // fac, Y // fac, p))
    return pd.DataFrame(rows, columns=["x", "y", "path"])
 
 
def paired_tile_ids_in(l0_dir, l1_dir, l2_dir, fac=1000):
    """Return rows (x, y, L0path, L1path, L2path) present at ALL three levels.
 
    l0_dir / l1_dir / l2_dir are the fine->coarse tier folders
    (cut_out/level1, level2, level3). fac is the quantization cell in level-0
    pixels; Panoptes uses resolution*50 (= 1000 for a 20x slide), and here it
    equals the middle tier's tile step so L1 maps 1:1 to cells. The %2 snap then
    doubles the cell to the coarse (L2) grid before the final merge.
    """
    idsa = _ids_from_dir(l0_dir, fac).rename(columns={"path": "L0path"})
    idsb = _ids_from_dir(l1_dir, fac).rename(columns={"path": "L1path"})
    idsc = _ids_from_dir(l2_dir, fac).rename(columns={"path": "L2path"})
 
    idsa = pd.merge(idsa, idsb, on=["x", "y"], how="left")   # L0 + L1
    idsa["x"] = idsa["x"] - (idsa["x"] % 2)                  # snap to coarse grid
    idsa["y"] = idsa["y"] - (idsa["y"] % 2)
    idsa = pd.merge(idsa, idsc, on=["x", "y"], how="left")   # + L2
    idsa = idsa.dropna().reset_index(drop=True)              # all three present
    return idsa[["x", "y", "L0path", "L1path", "L2path"]]
 
 
def pair_region(cut_out, fac=1000):
    """Convenience wrapper: pair cut_out/level1,2,3 and report the surviving count."""
    paired = paired_tile_ids_in(cut_out + "/level1", cut_out + "/level2",
                                 cut_out + "/level3", fac=fac)
    print("locations present at all three levels:", len(paired))
    return paired