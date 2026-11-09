#!/usr/bin/env python3
"""Frames from raw FCI Level 1c (FDHSI) instead of EUMETView: for one 10-min
product (zip from the EUMETSAT Data Store), on the same 2048 px geostationary
grid as the EUMETView frames (lon 0, sweep y, +-5,570,000 m):

  geocolour/  true colour by day (satpy), night side built like GeoColour:
              infrared clouds over a dark background, plus static city lights
  fog/        fog / low-cloud RGB (night microphysics) from brightness temperatures
  ir/         10.5 um, cold = bright, same grey scale as EUMETView's ir105

  uv run --extra raw raw/fci_frame.py PRODUCT.zip OUT_DIR [--keep]

OUT_DIR gets geocolour/, fog/, ir/ with YYYYmmdd_HHMM.jpg (repeat-cycle start).
The extracted chunks are deleted afterwards (and the zip, unless --keep).
"""
import argparse, datetime as dt, glob, os, re, shutil, sys, zipfile

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import nightgrade as ng          # noqa: E402
import starfield as sf           # noqa: E402

HALF = sf.HALF
SIZE = sf.NATIVE_SIZE
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
LIGHTS = os.path.join(ASSETS, "citylights.png")            # city lights (warm emission)
BACKGROUND = os.path.join(ASSETS, "night_background.jpg")  # GeoColour's static night ground
NIGHT = (80.0, 88.0)             # solar zenith band where GeoColour switches to its night image

# Fog / low-cloud RGB (EUMETSAT night microphysics) from brightness temperatures (K),
# matched to EUMETView's rgb_fog (fitted on 2026-11-08 00:00, binned medians):
#   R = BT12.3 - BT10.5 over [-4, 2] and B = BT10.5 over [243, 293], both linear;
#   G = BT10.5 - BT3.8 through EUMETView's (strongly non-linear) curve.
FOG_R = (-4.0, 2.0)
FOG_B = (243.0, 293.0)
FOG_G = ([0.0, 0.1, 0.2, 0.4, 0.6, 1.0, 1.5, 2.1, 3.0, 5.5],
         [2, 5, 12, 37, 70, 98, 124, 148, 176, 255])
# GeoColour's night luminance against that IR grey (2026-11-08 00:00, binned medians):
# clear surface stays at the dark navy background, clouds brighten with cold tops
NIGHT_L = ([0, 65, 85, 105, 125, 145, 165, 185, 205, 225, 245, 255],
           [33.7, 33.7, 51, 58, 74, 94, 120, 148, 175, 196, 214, 222])
# 10.5 um grey, cold = bright, through EUMETView's ir105_hrfi curve (00:00 + 12:00 fit)
IR_GREY = ([193, 224, 240, 250, 258, 265, 270, 275, 280, 285, 290, 295, 298, 303],
           [255, 183, 146, 123, 105, 88, 78, 66, 55, 43, 32, 21, 15, 0])


def slot_time(path):
    """Repeat-cycle start from the product name (..._OPE_<start>_<end>_...)."""
    m = re.search(r"_OPE_(\d{12})\d{2}_", os.path.basename(path))
    t = dt.datetime.strptime(m.group(1), "%Y%m%d%H%M")
    return t.replace(minute=t.minute // 10 * 10)


def to_grid(a, extent):
    """Native geos array (rows north->south) with area extent (x0, y0, x1, y1)
    -> our SIZE x SIZE grid, area-averaged."""
    x0, y0, x1, y1 = extent
    h, w = a.shape[:2]
    px = (x1 - x0) / w
    f = max(1, int((2 * HALF / SIZE) / px / 1.5))       # integer pre-shrink, no aliasing
    if f > 1:
        a = cv2.resize(a, (w // f, h // f), interpolation=cv2.INTER_AREA)
        px *= w / (w // f)
    s = 2 * HALF / SIZE / px                             # source px per target px
    # target pixel centre (i + .5) -> source coordinate
    cx = (-HALF - x0) / px - 0.5 + 0.5 * s
    cy = (y1 - HALF) / px - 0.5 + 0.5 * s
    M = np.array([[s, 0, cx], [0, s, cy]], np.float32)
    return cv2.warpAffine(a, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def scale(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)


def build(zip_path, out_dir, keep=False):
    from satpy import Scene
    from satpy.writers import get_enhanced_image
    import dask
    # netCDF4/HDF5 wheels on Linux are not thread-safe (HDF errors, segfaults
    # with threaded reads): read single-threaded there
    if sys.platform.startswith("linux") or os.environ.get("FCI_SYNC"):
        dask.config.set(scheduler="synchronous")
    else:
        dask.config.set(scheduler="threads", num_workers=os.cpu_count())
    when = slot_time(zip_path)
    name = when.strftime("%Y%m%d_%H%M") + ".jpg"
    tmp = os.path.splitext(os.path.abspath(zip_path))[0]
    if not os.path.isdir(tmp):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp + ".part", [n for n in z.namelist() if n.endswith(".nc")])
        os.replace(tmp + ".part", tmp)
    files = sorted(f for f in glob.glob(os.path.join(tmp, "**", "*.nc"), recursive=True)
                   if "TRAIL" not in f)
    scn = Scene(filenames=files, reader="fci_l1c_nc", reader_kwargs={"cache_handle": True})

    # --- infrared: brightness temperatures (2 km) ---
    scn.load(["ir_38", "ir_105", "ir_123"], upper_right_corner="NE")
    bt = {}
    for ch in ("ir_38", "ir_105", "ir_123"):
        d = scn[ch]
        bt[ch] = to_grid(np.nan_to_num(np.asarray(d.values, np.float32), nan=0.0),
                         d.attrs["area"].area_extent)
    disc = ~np.isnan(ng.load_normals()[0])                # inside the Earth disc
    ir = np.interp(bt["ir_105"], *IR_GREY).astype(np.float32)
    fog = np.dstack([scale(bt["ir_123"] - bt["ir_105"], *FOG_R) * 255,
                     np.interp(bt["ir_105"] - bt["ir_38"], *FOG_G),
                     scale(bt["ir_105"], *FOG_B) * 255]).astype(np.float32)

    # --- day: satpy true colour (1 km visible) ---
    scn.load(["true_color"], upper_right_corner="NE")
    res = scn.resample(scn.coarsest_area(["true_color"]), resampler="native")
    img = res["true_color"]
    rgb = get_enhanced_image(img).finalize(fill_value=0)[0]
    day = np.asarray(rgb.transpose("y", "x", "bands").values)[..., :3]
    if day.dtype != np.uint8:
        day = np.clip(day * (255 if day.max() <= 1.5 else 1), 0, 255).astype(np.uint8)
    day = to_grid(day.astype(np.float32), img.attrs["area"].area_extent)

    # --- night, built like GeoColour's: IR clouds on a dark background + lights ---
    sky = sf.Sky(mag_limit=1.0)
    sun = sky.sun_dir(when.replace(tzinfo=dt.timezone.utc) + dt.timedelta(minutes=5))
    n = ng.night_weight_band(sun, *NIGHT)[..., None]
    c = np.maximum(np.interp(ir, *NIGHT_L) - ng.BASE_L, 0).astype(np.float32)
    night = ng.BASE_RGB + c[..., None] * np.array([0.92, 0.96, 1.0], np.float32)
    if os.path.exists(BACKGROUND):            # clouds over the static ground, as GeoColour
        night = np.maximum(night, np.asarray(Image.open(BACKGROUND).convert("RGB"), np.float32))
    if os.path.exists(LIGHTS):                # lights dimmed under cloud
        clear = 1 - ng.smoothstep(c, 30.0, 100.0)[..., None]
        night = night + np.asarray(Image.open(LIGHTS).convert("RGB"), np.float32) * clear
    geo = day * (1 - n) + night * n

    for sub, a in (("geocolour", geo), ("fog", fog), ("ir", ir)):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        a = np.clip(a, 0, 255)
        a = a * (disc[..., None] if a.ndim == 3 else disc)
        Image.fromarray(a.astype(np.uint8)).save(os.path.join(out_dir, sub, name), quality=92)
    shutil.rmtree(tmp, ignore_errors=True)
    if not keep:
        os.remove(zip_path)
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("zip")
    p.add_argument("out")
    p.add_argument("--keep", action="store_true", help="keep the product zip")
    a = p.parse_args()
    print("wrote", build(a.zip, a.out, a.keep), flush=True)


if __name__ == "__main__":
    main()
