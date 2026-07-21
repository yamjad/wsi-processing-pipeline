# port of slicer.py from panoptes utilizing wsidicom 

import os
import glob
import multiprocessing as mp

import numpy as np
import pandas as pd
from PIL import Image
from wsidicom import WsiDicom

import os
import glob
import multiprocessing as mp
 
import numpy as np
import pandas as pd
from PIL import Image
from wsidicom import WsiDicom
 

def bgcheck(img, ts, white_thr=200, black_thr=50):
    """Return blank-pixel fraction of a tile.
 
    A pixel is 'blank' if all 3 channels are near-white (> white_thr) or all 3 are
    near-black (< black_thr). ts is the tile side length in pixels of the array that
    was read (i.e. full_width_region), so ts*ts is the pixel count.
    """
    arr = np.array(img)[:, :, :3]
    arr = np.nan_to_num(arr)
    white = (arr > white_thr).astype(np.uint8)
    black = (arr < black_thr).astype(np.uint8)
    white = white[:, :, 0] * white[:, :, 1] * white[:, :, 2]
    black = black[:, :, 0] * black[:, :, 1] * black[:, :, 2]
    blank_fraction = (white.sum() + black.sum()) / (ts * ts)
    return blank_fraction
 

def v_slide(slp, n_y, x, y, tile_size, stepsize, x0, outdir, level,
            dp=None, white_thr=200, black_thr=50, blank_max=0.8,
            edge="pad", pad_color=(255, 255, 255)):
    """Cut one vertical column (x0) of tiles. Returns list of kept-tile records.
 
    Edge handling: the grid's last column/row can extend past the level bounds.
    OpenSlide silently padded such reads; wsidicom raises WsiDicomOutOfBoundsError.
    edge="pad"  -> read the in-bounds part and pad up to tile_size (default).
    edge="skip" -> drop any tile that would cross the boundary.
    pad_color defaults to white (glass-like); Panoptes incidentally got black.
    """
    slide = WsiDicom.open(slp)
    imloc = []
 
    lvl = slide.levels[level]          # `level` is the LIST INDEX of an available level
    level_w, level_h = lvl.size.width, lvl.size.height   # this level's pixel grid
    pyr = lvl.level                    # PYRAMID index - what read_region wants. It
                                       # differs from the list index whenever pyramid
                                       # levels are missing (e.g. downsamples 1,4,16 ->
                                       # pyramid indices 0,2,4 at list indices 0,1,2).
    scale = 2 ** pyr                   # downsample vs base level = 2**pyramid_index
 
    read_x = x0 * stepsize + x         # level-relative x (in THIS level's pixel grid)
    base_x = read_x * scale            # level-0 x for the filename (pairing frame)
 
    y0 = 0
    while y0 < n_y:
        read_y = y0 * stepsize + y
        base_y = read_y * scale
 
        # clamp the read window to the level bounds
        avail_w = level_w - read_x
        avail_h = level_h - read_y
        if avail_w <= 0 or avail_h <= 0:
            y0 += 1
            continue
        read_w = min(tile_size, avail_w)
        read_h = min(tile_size, avail_h)
 
        if edge == "skip" and (read_w < tile_size or read_h < tile_size):
            y0 += 1
            continue
 
        # wsidicom read_region: location is in THIS level's pixel grid, size in px,
        # and the level argument is the PYRAMID index (pyr), not the list index.
        img = slide.read_region((read_x, read_y), pyr, (read_w, read_h))
        img = img.convert("RGB")       # reader returns RGB, but be explicit
 
        # pad a partial edge tile back up to the full tile footprint
        if (read_w, read_h) != (tile_size, tile_size):
            canvas = Image.new("RGB", (tile_size, tile_size), pad_color)
            canvas.paste(img, (0, 0))
            img = canvas
 
        wscore = bgcheck(img, tile_size, white_thr, black_thr)
        if wscore < blank_max:
            img = img.resize((299, 299))
            if dp:
                strr = outdir + "/region_x-{}-y-{}_{}.png".format(base_x, base_y, str(dp))
            else:
                strr = outdir + "/region_x-{}-y-{}.png".format(base_x, base_y)
            img.save(strr)
            imloc.append([x0, y0, base_x, base_y, strr])
        y0 += 1
 
    slide.close()
    return imloc
 

def tile(slide_path, outdir, level, std_img=None, dp=None, ft=1,
         white_thr=200, black_thr=50, blank_max=0.8,
         thumbnail_size=(1024, 1024), parallel=True,
         edge="pad", pad_color=(255, 255, 255)):
    """Tile one DICOM-WSI slide. Returns (n_x, n_y, lowres, ct).
 
    std_img is reserved for Stream D (normalization) and is unused here.
    Set parallel=False to run single-threaded, which surfaces real tracebacks
    from workers - do this first when validating on a new slide.
    """
    os.makedirs(outdir, exist_ok=True)
 
    slide = WsiDicom.open(slide_path)
    print("slide:", slide_path)
 
    # equivalent of OpenSlide slide.level_dimensions
    level_dimensions = [(l.size.width, l.size.height) for l in slide.levels]
    print("level_dimensions:", level_dimensions)
 
    lvl = slide.levels[level]
    bounds_width, bounds_height = lvl.size.width, lvl.size.height
    print("using list level {} -> pyramid index {}, downsample {}x, size {}x{}".format(
        level, lvl.level, 2 ** lvl.level, bounds_width, bounds_height))
 
    x = 0
    y = 0
    half_width_region = 49 * ft
    full_width_region = 299 * ft
    stepsize = full_width_region - half_width_region      # overlap = 49*ft
 
    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)
    print("grid: n_x={}, n_y={}, stepsize={}, tile={}".format(
        n_x, n_y, stepsize, full_width_region))
 
    # thumbnail: Panoptes built a manual low-res read; wsidicom has read_thumbnail,
    # which preserves aspect ratio within the given max box (see task A3).
    try:
        lowres = slide.read_thumbnail(thumbnail_size)
        lowres = np.array(lowres.convert("RGB"))
    except Exception as e:
        print("thumbnail failed:", e)
        lowres = None
 
    slide.close()   # each worker opens its own handle
 
    # one task per column
    tasks = [
        (slide_path, n_y, x, y, full_width_region, stepsize, x0, outdir, level,
         dp, white_thr, black_thr, blank_max, edge, pad_color)
        for x0 in range(n_x)
    ]
 
    if parallel:
        print("cpus:", mp.cpu_count())
        with mp.Pool(processes=mp.cpu_count()) as pool:
            temp = pool.starmap(v_slide, tasks)
    else:
        temp = [v_slide(*t) for t in tasks]
 
    # flatten, keeping only non-empty column results
    imloc = []
    for sub in filter(None, temp):
        imloc.extend(sub)
 
    imlocpd = pd.DataFrame(imloc, columns=["X_pos", "Y_pos", "X", "Y", "Loc"])
    imlocpd = (imlocpd.sort_values(["X_pos", "Y_pos"], ascending=[True, True])
                      .reset_index(drop=True)
                      .reset_index(drop=False))
    imlocpd.columns = ["Num", "X_pos", "Y_pos", "X", "Y", "Loc"]
 
    csv_path = outdir + ("/{}_dict.csv".format(dp) if dp else "/dict.csv")
    imlocpd.to_csv(csv_path, index=False)
 
    ct = len(imloc)
    print("tiles kept:", ct, "->", csv_path)
    return n_x, n_y, lowres, ct
 
 
def contact_sheet(tile_dir, out_path, n=50, cols=10, thumb=64):
    paths = sorted(glob.glob(os.path.join(tile_dir, "region_*.png")))[:n]
    if not paths:
        print("no tiles found for contact sheet in", tile_dir)
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
    for i, p in enumerate(paths):
        t = Image.open(p).convert("RGB").resize((thumb, thumb))
        sheet.paste(t, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)
    print("contact sheet:", out_path)
 

def scan_blank_fractions(slide_path, level, ft=1,
                         white_thr=200, black_thr=50, sample_stride=1):
    """Compute blank fraction for every candidate tile WITHOUT saving.
 
    Returns a list of floats. sample_stride>1 subsamples columns to run faster
    (e.g. 4 = every 4th column) when you just need the distribution shape.
    """
    slide = WsiDicom.open(slide_path)
    lvl = slide.levels[level]
    pyr = lvl.level                       # pyramid index for read_region
    bounds_width, bounds_height = lvl.size.width, lvl.size.height
 
    full_width_region = 299 * ft
    stepsize = full_width_region - 49 * ft
    n_x = int((bounds_width - 1) / stepsize)
    n_y = int((bounds_height - 1) / stepsize)
 
    fractions = []
    for x0 in range(0, n_x, sample_stride):
        for y0 in range(n_y):
            img = slide.read_region((x0 * stepsize, y0 * stepsize), pyr,
                                    (full_width_region, full_width_region)).convert("RGB")
            fractions.append(bgcheck(img, full_width_region, white_thr, black_thr))
    slide.close()
    return fractions

'''
if __name__ == "__main__":
    # Example. Point SLIDE at the FOLDER containing all DICOM instances for the slide.
    SLIDE = ("/Users/yahyaamjad/Downloads/Research/cptac_brca/01BR001/"
             "2.25.48791557373299768401597362411459861639")
    OUT = "./tiles_out"
 
    # 1) validate single-threaded on level 0 first
    n_x, n_y, lowres, ct = tile(SLIDE, OUT, level=0, parallel=False)
 
    # 2) deliverables
    contact_sheet(OUT, os.path.join(OUT, "contact_sheet.png"))
    fr = scan_blank_fractions(SLIDE, level=0, sample_stride=4)
    plot_blank_histogram(fr, os.path.join(OUT, "blank_hist.png"), blank_max=0.8)
'''