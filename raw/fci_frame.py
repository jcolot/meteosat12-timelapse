"""Build a full-disc frame from raw FCI Level 1c (FDHSI) with satpy, on the
same 2048 px geostationary grid as the EUMETView frames (lon 0, sweep y,
+-5,570,000 m). Run in its own uv environment:
  uv run --with satpy --with h5netcdf --with netCDF4 --with pyspectral --with pillow \
      python raw/fci_frame.py PRODUCT.zip COMPOSITE OUT.png
COMPOSITE: e.g. true_color, geo_color; 'list' prints what satpy offers."""
import glob, os, sys, tempfile, zipfile
import numpy as np
from PIL import Image

HALF = 5_570_000.0
SIZE = 2048


def main():
    src, comp, out = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else None)
    from satpy import Scene
    import dask
    dask.config.set(scheduler="threads", num_workers=4)
    tmp = os.path.splitext(os.path.abspath(src))[0]          # extracted once, reused
    if not os.path.isdir(tmp):
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp + ".part", [n for n in z.namelist() if n.endswith(".nc")])
        os.replace(tmp + ".part", tmp)
    files = sorted(f for f in glob.glob(os.path.join(tmp, "**", "*.nc"), recursive=True)
                   if "TRAIL" not in f)
    print(len(files), "chunk files", flush=True)
    scn = Scene(filenames=files, reader="fci_l1c_nc", reader_kwargs={"cache_handle": True})
    if comp == "list":
        print(sorted(scn.available_composite_names()))
        return
    scn.load([comp], upper_right_corner="NE")
    # one common resolution (the coarsest input), still the native geos grid
    # keep `scn` alive: its file handlers close when it is garbage-collected
    res = scn.resample(scn.coarsest_area(), resampler="native")
    img = res[comp]
    area = img.attrs["area"]
    x0, y0, x1, y1 = area.area_extent
    print("native", area.width, "x", area.height, "extent", area.area_extent, flush=True)
    from satpy.writers import get_enhanced_image
    rgb = get_enhanced_image(img).finalize(fill_value=0)[0]
    a = np.asarray(rgb.transpose("y", "x", "bands").values)
    if a.dtype != np.uint8:
        a = np.clip(a * 255 if a.max() <= 1.5 else a, 0, 255).astype(np.uint8)
    a = a[..., :3]
    # resample to our grid: same projection, so only scale + offset
    import cv2
    px = (x1 - x0) / area.width
    n = int(round(area.width * px / (2 * HALF / SIZE)))
    small = cv2.resize(a, (n, n), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    o = (SIZE - n) // 2
    if o >= 0:
        canvas[o:o + n, o:o + n] = small
    else:
        canvas = small[-o:-o + SIZE, -o:-o + SIZE]
    Image.fromarray(canvas).save(out)
    print("wrote", out, "scaled", area.width, "->", n, flush=True)


if __name__ == "__main__":
    main()
