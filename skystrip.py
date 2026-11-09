"""
Pre-rendered sky strip: the starfield and Milky Way for a whole day, drawn
once, then looked up per frame.

Why it works: the camera is fixed to the rotating Earth, so every canvas
pixel has a fixed direction in Earth-fixed axes. In the true-equator-and-
equinox-of-date frame (TOD) the stars stay put for a day (nutation and
aberration change by far less than a pixel), and the Earth-fixed frame turns
about the polar axis by Greenwich apparent sidereal time (GAST). So

    RA_tod(pixel, t) = lon_itrs(pixel) + GAST(t),  Dec_tod(pixel) = lat_itrs(pixel)

and a frame is one lookup into an RA/Dec strip. In that frame the sky drifts
purely along RA, so star motion blur is a box filter along the strip's x axis,
applied once at build time.

The strip stores the stars' accumulated linear flux *before* the soft
saturation, exactly what starfield.render accumulates, so the per-frame tone
mapping (and the Moon and planets, drawn per frame on top) are unchanged.

Files (memory-mapped, shared by all workers; only the slice in view is read):
  stars: float32 (rows, cols, 3) at the canvas pixel scale (at the centre)
  mw:    float32 (rows/2, cols/2, 3), Milky Way in S10 (calibrated, desaturated)
"""
import datetime as dt
import hashlib
import json
import os

import cv2
import numpy as np

import milkyway as mwy
import starfield as sf

SIDEREAL_DEG_PER_MIN = 360.98564736629 / 1440.0
MW_DOWN = 2


def _key(params):
    return hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]


def canvas_lonlat(width, height, ppr):
    """Earth-fixed longitude/latitude (degrees) of every canvas ray."""
    d = mwy.canvas_rays(width, height, ppr)
    lon = (np.degrees(np.arctan2(d[1], d[0])) % 360.0).astype(np.float32)   # canvas looks along 180: no seam
    lat = np.degrees(np.arcsin(np.clip(d[2], -1, 1))).astype(np.float32)
    return lon, lat


def gast_deg(sky, when):
    return float(sky.ts.from_datetime(when).gast) * 15.0


class SkyStrip:
    def __init__(self, path):
        meta = json.load(open(path + ".json"))
        self.meta = meta
        self.ppd = meta["ppd"]
        self.dmax = meta["dmax"]
        self.cols = meta["cols"]
        self.rows = meta["rows"]
        self.stars = np.memmap(path + ".stars.f32", np.float32, "r", shape=(self.rows, self.cols, 3))
        self.mw = np.memmap(path + ".mw.f32", np.float32, "r",
                            shape=(self.rows // MW_DOWN, self.cols // MW_DOWN, 3))
        W, H, ppr = meta["W"], meta["H"], meta["ppr"]
        lon, lat = canvas_lonlat(W, H, ppr)
        self.lon = lon
        self.my = ((self.dmax - lat) * self.ppd - 0.5).astype(np.float32)
        self.span = float(lon.max() - lon.min())

    def _lookup(self, img, scale, sky, when):
        """Remap the strip `img` (pixel scale = ppd / scale) into the canvas."""
        g = gast_deg(sky, when)
        ra = self.lon + np.float32(g)                                    # degrees, TOD
        cols = img.shape[1]
        x = ((ra % 360.0) * (self.ppd / scale) - 0.5).astype(np.float32)
        # copy only the columns in view (wrapping around RA 0 if needed)
        lo = int(np.floor(((self.lon.min() + g) % 360.0) * self.ppd / scale)) - 4
        n = int(np.ceil(self.span * self.ppd / scale)) + 10
        lo %= cols
        if lo + n <= cols:
            sub = img[:, lo:lo + n]                 # a view: no copy, rows keep their stride
        else:                                       # straddles RA 0/360: stitch the two pieces
            sub = np.concatenate([img[:, lo:], img[:, :lo + n - cols]], axis=1)
        xs = ((x - lo) % cols).astype(np.float32)
        ys = (self.my + 0.5) / scale - 0.5 if scale != 1 else self.my
        return cv2.remap(sub, xs, ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def star_acc(self, sky, when):
        """(H, W, 3) accumulated star flux at `when` (before saturation)."""
        return self._lookup(self.stars, 1, sky, when)

    def mw_s10(self, sky, when):
        """(H, W, 3) Milky Way surface brightness (S10) at `when`."""
        return self._lookup(self.mw, MW_DOWN, sky, when)


def render(strip, sky, when, ppr, gain=1.0, contrast=0.75, psf=0.7, shutter_min=0.0, bodies=True,
           body_shutter=1.0, tone=True, **_):
    """Drop-in for starfield.render using the strip; Moon/planets per frame."""
    acc = strip.star_acc(sky, when)
    sigma = max(0.45, psf * ppr / sf.NATIVE_PPR)
    if bodies:
        H, W = acc.shape[:2]
        # same shutter sampling as starfield.render for moving bodies
        bs = shutter_min * body_shutter              # Moon/planets may get a shorter shutter
        L = bs * SIDEREAL_DEG_PER_MIN * strip.ppd
        Kb = max(1, min(48, int(np.ceil(L / 0.8)))) if bs > 0 else 1
        for k in range(Kb):
            f = (k + 0.5) / Kb - 0.5 if Kb > 1 else 0.0
            tb = sky.ts.from_datetime(when + dt.timedelta(minutes=bs * f))
            for b in sky.bodies(tb):
                sf.draw_body(acc, b, ppr, gain / Kb, sigma, contrast)
    if not tone:                                    # caller (e.g. the GPU) applies the curve
        return acc
    np.negative(acc, out=acc)
    np.exp(acc, out=acc)
    np.subtract(1.0, acc, out=acc)
    np.clip(acc, 0, 1, out=acc)
    np.power(acc, np.float32(1 / 2.2), out=acc)
    return acc


# --- building ----------------------------------------------------------------------
def build(day_mid, W, H, ppr, shutter_min, out_dir, gain=1.0, mag_limit=8.0, contrast=0.75, psf=0.7,
          tile_deg=15.0, log=print):
    """Render the strip for the day around `day_mid` (UTC). Returns its path
    prefix; reuses an existing strip with the same parameters."""
    params = dict(day=day_mid.strftime("%Y-%m-%dT%H:%M"), W=W, H=H, ppr=round(ppr, 6),
                  shutter=round(shutter_min, 6), gain=gain, mag=mag_limit, contrast=contrast, psf=psf,
                  mw_cal=mwy.MAP_TO_S10, mw_sat=mwy.SATURATION, v=1)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"skystrip_{day_mid:%Y%m%d}_{_key(params)}")
    if os.path.exists(path + ".json"):
        return path

    sky = sf.Sky(mag_limit=mag_limit)
    ppd = ppr * np.pi / 180.0                         # canvas pixels per degree at the centre
    lon, lat = canvas_lonlat(W, H, ppr)
    dmax = float(np.abs(lat).max()) + 1.0
    rows = int(np.ceil(2 * dmax * ppd))
    rows += rows % MW_DOWN
    cols = int(round(360.0 * ppd))
    cols -= cols % MW_DOWN
    ppd = cols / 360.0                                # exact wrap-around
    sigma = max(0.45, psf * ppr / sf.NATIVE_PPR)
    blur_px = shutter_min * SIDEREAL_DEG_PER_MIN * ppd
    log(f"sky strip: {cols} x {rows} px, Dec +-{dmax:.1f} deg, star blur {blur_px:.1f} px")

    # star positions in the true-of-date frame at mid-day (apparent, from the satellite)
    v, t = sky.directions(day_mid)
    g = gast_deg(sky, day_mid)
    lon_s = np.degrees(np.arctan2(v[1], v[0])) + g
    dec_s = np.degrees(np.arcsin(np.clip(v[2], -1, 1)))
    m = sky.vmag
    keep = (m <= mag_limit) & (np.abs(dec_s) < dmax + 0.5)
    ra = lon_s[keep] % 360.0
    dec = dec_s[keep]
    flux = sf.star_flux(m[keep], gain, contrast)
    rgb = (sf.bv_to_rgb(sky.bv[keep]) * flux[:, None]).astype(np.float32)
    bright = m[keep] < 3.0
    xs = ra * ppd - 0.5
    ys = (dmax - dec) * ppd - 0.5

    stars = np.lib.format.open_memmap(path + ".stars.tmp.npy", "w+", np.float32, (rows, cols, 3))
    tile = int(round(tile_deg * ppd))
    pad = int(np.ceil(blur_px)) + int(np.ceil(4 * sigma * 4)) + 4
    for x0 in range(0, cols, tile):
        x1 = min(cols, x0 + tile)
        w = x1 - x0 + 2 * pad
        acc = np.zeros((rows, w, 3), np.float32)
        # stars in this tile (with wrap-around copies near RA 0/360)
        for shift in (0.0, -cols, cols):
            xx = xs + shift - (x0 - pad)
            sel = (xx > -8) & (xx < w + 8)
            if sel.any():
                sf.splat(acc, xx[sel], ys[sel], rgb[sel], sigma)
                b = sel & bright
                if b.any():
                    sf.splat(acc, (xs + shift - (x0 - pad))[b], ys[b], rgb[b] * 0.25, sigma * 4)
        if blur_px > 1:
            kw = int(round(blur_px)) | 1                    # odd: centred, no half-pixel shift
            k = np.ones((1, kw), np.float32) / kw
            acc = cv2.filter2D(acc, -1, k, borderType=cv2.BORDER_CONSTANT)
        stars[:, x0:x1] = acc[:, pad:pad + (x1 - x0)]
        log(f"  stars {x1 / cols * 100:5.1f}%")
    stars.flush()
    del stars

    # Milky Way: J2000 map resampled into the TOD strip (half resolution)
    R = np.asarray(sky.ts.from_datetime(day_mid).M).T.astype(np.float32)    # TOD -> J2000
    mrows, mcols = rows // MW_DOWN, cols // MW_DOWN
    mw = mwy.MilkyWay.__new__(mwy.MilkyWay)
    mw.mh, mw.mw = np.load(mwy.MAP, mmap_mode="r").shape[:2]
    mmap = mwy.load_map(mwy.SOFTEN)
    out = np.lib.format.open_memmap(path + ".mw.tmp.npy", "w+", np.float32, (mrows, mcols, 3))
    dec_m = np.radians(dmax - (np.arange(mrows) + 0.5) / (ppd / MW_DOWN))
    for c0 in range(0, mcols, 2048):
        c1 = min(mcols, c0 + 2048)
        ra_m = np.radians((np.arange(c0, c1) + 0.5) / (ppd / MW_DOWN))
        RA, DE = np.meshgrid(ra_m, dec_m)
        vt = np.stack([np.cos(DE) * np.cos(RA), np.cos(DE) * np.sin(RA), np.sin(DE)]).astype(np.float32)
        vj = np.einsum("ij,jhw->ihw", R, vt)
        raj = np.arctan2(vj[1], vj[0]); dej = np.arcsin(np.clip(vj[2], -1, 1))
        mx = np.mod((0.5 - raj / (2 * np.pi)) * mw.mw - 0.5, mw.mw).astype(np.float32)
        my = ((0.5 - dej / np.pi) * mw.mh - 0.5).astype(np.float32)
        smp = cv2.remap(mmap, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * mwy.MAP_TO_S10
        lum = smp.mean(axis=2, keepdims=True)
        out[:, c0:c1] = lum + mwy.SATURATION * (smp - lum)
    out.flush()
    del out, mmap

    # raw float32 files + metadata, then publish atomically
    for name in ("stars", "mw"):
        a = np.load(path + f".{name}.tmp.npy", mmap_mode="r")
        a.tofile(path + f".{name}.f32")
        os.remove(path + f".{name}.tmp.npy")
    meta = dict(params, ppd=ppd, dmax=dmax, rows=rows, cols=cols, W=W, H=H, ppr=ppr)
    json.dump(meta, open(path + ".json.tmp", "w"), indent=1)
    os.replace(path + ".json.tmp", path + ".json")
    return path
