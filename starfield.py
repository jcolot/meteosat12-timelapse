#!/usr/bin/env python3
"""
Put the real sky behind EUMETView full-disc frames.

The camera is MTG-I1 in geostationary orbit over 0N 0E, looking at Earth's
centre. For each frame time we compute the apparent direction of every
Hipparcos star (plus Moon, planets and Sun) as seen *from the satellite*,
express it in the Earth-fixed frame the image is tied to, and project it with
the same geostationary (GEOS) projection the WMS server uses for the frame.

Commands
  render     one star layer for a given UTC time
  composite  stars behind every frame in a folder (time taken from filename)
  validate   check our projection against the server's own graticule/coastlines

Examples
  .venv/bin/python starfield.py render --time 2026-11-06T12:00 -o stars.png
  .venv/bin/python starfield.py composite frames/ -o frames_sky/
  .venv/bin/python starfield.py validate

Note: a camera exposed for the sunlit Earth would not record stars. Positions
are true; brightness is enhanced (see --gain / --mag-limit).
"""
import argparse
import datetime as dt
import os
import re
import sys

import numpy as np
from PIL import Image
from skyfield.api import Star, load, wgs84
from skyfield.framelib import itrs

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# --- Projection of the WMS frames (EPSG:900003 on view.eumetsat.int) ---------
# Fitted against the server's graticule layer with `validate` (see README).
H_SAT = 35_785_831.0   # satellite height above the equator, m
A_EQ = 6_378_137.0     # WGS84 equatorial radius, m
B_POL = 6_356_752.314  # WGS84 polar radius, m
SWEEP = "y"            # GEOS sweep axis ("y" = MSG/MTG convention)
HALF = 5_570_000.0     # half-width of the frame BBOX in projection metres
SAT_LON = 0.0          # sub-satellite longitude, degrees
SCAN_OFFSET_MIN = 5.0  # frame time -> time the sky is rendered (mid-scan)


# --- Geometry ----------------------------------------------------------------
def scan_angles(vx, vy, vz, sweep=SWEEP):
    """Direction from the satellite (Earth-fixed axes, x toward the
    sub-satellite point's meridian, z north) -> GEOS scan angles, radians.
    `tmp` is the component pointing toward Earth (i.e. -x)."""
    tmp = -vx
    if sweep == "x":
        ax = np.arctan2(vy, np.hypot(vz, tmp))
        ay = np.arctan2(vz, tmp)
    else:
        ax = np.arctan2(vy, tmp)
        ay = np.arctan2(vz, np.hypot(vy, tmp))
    return ax, ay


def angles_to_pixels(ax, ay, size, h=H_SAT, half=HALF):
    """Scan angles -> (col, row) pixel centres in a size x size frame."""
    x = ax * h
    y = ay * h
    col = (x + half) / (2 * half) * size - 0.5
    row = (half - y) / (2 * half) * size - 0.5
    return col, row


def earth_to_pixels(lat_deg, lon_deg, size, h=H_SAT, sweep=SWEEP, ellipsoid=True):
    """Geodetic lat/lon on the Earth surface -> pixels (used by `validate`).
    Same maths as PROJ's +proj=geos. Returns NaN for points not visible."""
    lat = np.radians(lat_deg)
    lon = np.radians(np.asarray(lon_deg) - SAT_LON)
    a = A_EQ
    b = B_POL if ellipsoid else A_EQ
    latc = np.arctan((b / a) ** 2 * np.tan(lat))                  # geocentric
    r = (b / a) / np.hypot((b / a) * np.cos(latc), np.sin(latc))  # in units of a
    px = r * np.cos(latc) * np.cos(lon)
    py = r * np.cos(latc) * np.sin(lon)
    pz = r * np.sin(latc)
    sat = (a + h) / a
    vx, vy, vz = px - sat, py, pz          # vector satellite -> point
    # visibility: point faces the satellite
    vis = (sat - px) * px - py * py - pz * pz * (a / b) ** 2 > 0
    ax, ay = scan_angles(vx, vy, vz, sweep)
    col, row = angles_to_pixels(ax, ay, size, h)
    col = np.where(vis, col, np.nan)
    row = np.where(vis, row, np.nan)
    return col, row


# --- Sky model ----------------------------------------------------------------
class Sky:
    def __init__(self, mag_limit=9.0):
        self.ts = load.timescale()
        self.eph = load(os.path.join(DATA, "de421.bsp"))
        self.sat = wgs84.latlon(0.0, SAT_LON, elevation_m=H_SAT)
        self.observer = self.eph["earth"] + self.sat
        self._load_stars(mag_limit)

    def _load_stars(self, mag_limit):
        rows = []
        with open(os.path.join(DATA, "hip.tsv")) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue
                p = line.rstrip("\n").split("\t")
                if len(p) < 8 or not p[0].strip().isdigit():
                    continue
                try:
                    vmag, ra, de = float(p[1]), float(p[2]), float(p[3])
                except ValueError:
                    continue
                if vmag > mag_limit:
                    continue

                def num(s, default=0.0):
                    try:
                        return float(s)
                    except ValueError:
                        return default
                rows.append((vmag, ra, de, num(p[4]), num(p[5]), num(p[6]), num(p[7], 0.6)))
        arr = np.array(rows)
        self.vmag, self.bv = arr[:, 0], arr[:, 6]
        plx = np.clip(arr[:, 5], 0.0, None)  # negative parallaxes -> 0 (infinitely far)
        self.stars = Star(
            ra_hours=arr[:, 1] / 15.0,
            dec_degrees=arr[:, 2],
            ra_mas_per_year=arr[:, 3],
            dec_mas_per_year=arr[:, 4],
            parallax_mas=plx,
            epoch=self.ts.tt_jd(2448349.0625),  # Hipparcos epoch J1991.25
        )

    def directions(self, when):
        """Unit vectors (3, N) of all stars from the satellite, Earth-fixed axes."""
        t = self.ts.from_datetime(when)
        app = self.observer.at(t).observe(self.stars).apparent()
        v = app.frame_xyz(itrs).au
        return v / np.linalg.norm(v, axis=0), t

    def sun_dir(self, when):
        """Unit vector from Earth's centre to the Sun, Earth-fixed axes."""
        t = self.ts.from_datetime(when)
        v = self.eph["earth"].at(t).observe(self.eph["sun"]).apparent().frame_xyz(itrs).au
        return v / np.linalg.norm(v)

    def bodies(self, t):
        """Moon, Sun and planets: dicts with direction, angular radius, magnitude."""
        out = []
        obs = self.observer.at(t)
        sun_pos = obs.observe(self.eph["sun"]).apparent()
        sun_v = sun_pos.frame_xyz(itrs).au
        for name, key, radius_km, mag in [
            ("Sun", "sun", 696_000, -26.7),
            ("Moon", "moon", 1_737.4, None),
            ("Mercury", "mercury", 2_440, 0.0),
            ("Venus", "venus", 6_052, -4.2),
            ("Mars", "mars", 3_390, 0.5),
            ("Jupiter", "jupiter barycenter", 69_911, -2.3),
            ("Saturn", "saturn barycenter", 58_232, 0.6),
        ]:
            app = obs.observe(self.eph[key]).apparent()
            v = app.frame_xyz(itrs).au
            d_au = np.linalg.norm(v)
            ang_r = np.arcsin(min(1.0, radius_km / (d_au * 149_597_870.7)))
            out.append(dict(name=name, v=v / d_au, ang_r=ang_r, mag=mag,
                            sun_dir=sun_v / np.linalg.norm(sun_v)))
        return out


# --- Rendering ----------------------------------------------------------------
def bv_to_rgb(bv):
    """B-V colour index -> linear RGB (approx. blackbody), normalised to max 1."""
    bv = np.clip(bv, -0.4, 2.0)
    T = 4600.0 * (1 / (0.92 * bv + 1.7) + 1 / (0.92 * bv + 0.62))
    t = T / 100.0
    hot = np.maximum(t - 60, 1e-3)
    r = np.where(t <= 66, 255.0, 329.7 * hot ** -0.1332)
    g = np.where(t <= 66, 99.47 * np.log(t) - 161.1, 288.1 * hot ** -0.0755)
    b = np.where(t >= 66, 255.0, np.where(t <= 19, 0.0, 138.5 * np.log(np.maximum(t - 10, 1)) - 305.0))
    rgb = (np.clip(np.stack([r, g, b], -1), 0, 255) / 255.0) ** 2.2  # sRGB-ish -> linear
    rgb /= rgb.max(axis=-1, keepdims=True)
    return 0.55 + 0.45 * rgb                                # keep it subtle


def splat(acc, col, row, flux_rgb, sigma):
    """Add Gaussian PSFs (sub-pixel accurate) to acc (H, W, 3)."""
    H, W, _ = acc.shape
    r = int(np.ceil(3 * sigma))
    c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
    fc, fr = col - c0, row - r0
    norm = 1.0 / (2 * np.pi * sigma ** 2)
    for dy in range(-r, r + 2):
        for dx in range(-r, r + 2):
            w = norm * np.exp(-((dx - fc) ** 2 + (dy - fr) ** 2) / (2 * sigma ** 2))
            cc, rr = c0 + dx, r0 + dy
            ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
            np.add.at(acc, (rr[ok], cc[ok]), flux_rgb[ok] * w[ok, None])


def star_flux(mag, gain, contrast):
    """Integrated flux per star. contrast=1 is physical (2.512x per magnitude);
    lower values compress the range so faint stars stay visible."""
    return 1.5 * gain * 10 ** (-0.4 * contrast * (np.asarray(mag) - 6.0))


NATIVE_SIZE = 2048
NATIVE_PPR = H_SAT * NATIVE_SIZE / (2 * HALF)  # pixels per radian of the 2048 px WMS frame


def project(ax, ay, width, height, ppr):
    """Scan angles -> (col, row) on a width x height canvas centred on the
    sub-satellite direction. The GEOS projection is linear in scan angle, so a
    canvas wider than the WMS frame is an exact extension of it."""
    return width / 2 - 0.5 + ax * ppr, height / 2 - 0.5 - ay * ppr


def render(sky, when, size=None, gain=1.0, mag_limit=9.0, psf=0.7, bodies=True, contrast=0.75,
           width=None, height=None, ppr=None, shutter_min=0.0, body_shutter=1.0):
    """RGB float image (height, width, 3) in [0, 1] of the sky at `when` (UTC).
    Default: the WMS frame itself (size x size). For a wider view pass width,
    height and ppr (pixels per radian of scan angle).

    shutter_min > 0 adds motion blur: each star is drawn as the streak it
    traces while a virtual shutter is open from when - shutter/2 to
    when + shutter/2 (scene time). Total light per star is unchanged."""
    W = width or size
    H = height or size
    ppr = ppr or H_SAT * size / (2 * HALF)
    m = sky.vmag
    sigma = max(0.45, psf * ppr / NATIVE_PPR)
    acc = np.zeros((H, W, 3), np.float32)

    half = dt.timedelta(minutes=shutter_min / 2)
    ends = [when - half, when + half] if shutter_min > 0 else [when]
    pts = []
    for w in ends:
        v, t = sky.directions(w)
        ax, ay = scan_angles(*v)
        pts.append(project(ax, ay, W, H, ppr) + (v[0] < 0,))
    (c0, r0, front0), (c1, r1, front1) = pts[0], pts[-1]
    margin = 10 + (np.hypot(c1 - c0, r1 - r0).max() if len(pts) > 1 else 0)
    keep = (front0 & front1 & (m <= mag_limit)
            & (np.minimum(c0, c1) > -margin) & (np.maximum(c0, c1) < W + margin)
            & (np.minimum(r0, r1) > -margin) & (np.maximum(r0, r1) < H + margin))
    flux = star_flux(m[keep], gain, contrast)
    rgb = (bv_to_rgb(sky.bv[keep]) * flux[:, None]).astype(np.float32)
    bright = m[keep] < 3.0
    # one sub-sample per pixel of streak length keeps the streak continuous
    L = np.hypot(c1 - c0, r1 - r0)[keep]
    K = max(1, int(np.ceil(L.max() / 0.8))) if len(pts) > 1 and keep.any() else 1
    for k in range(K):
        f = (k + 0.5) / K if K > 1 else 0.0
        cc = c0[keep] + f * (c1[keep] - c0[keep])
        rr = r0[keep] + f * (r1[keep] - r0[keep])
        splat(acc, cc, rr, rgb / K, sigma)
        # soft halo on bright stars so they read as brighter, not just bigger blobs
        if bright.any():
            splat(acc, cc[bright], rr[bright], rgb[bright] * 0.25 / K, sigma * 4)

    t = sky.ts.from_datetime(when)
    if bodies:
        # Moon/planets: average a few renders across the shutter
        bs = shutter_min * body_shutter              # Moon/planets may get a shorter shutter
        Kb = 1 if bs <= 0 else max(1, min(int(np.ceil(K * body_shutter)), 48))
        for k in range(Kb):
            f = (k + 0.5) / Kb - 0.5 if Kb > 1 else 0.0
            tb = sky.ts.from_datetime(when + dt.timedelta(minutes=bs * f))
            for b in sky.bodies(tb):
                draw_body(acc, b, ppr, gain / Kb, sigma, contrast)

    np.negative(acc, out=acc)                      # soft saturation, in place (float32)
    np.exp(acc, out=acc)
    np.subtract(1.0, acc, out=acc)
    np.clip(acc, 0, 1, out=acc)
    np.power(acc, np.float32(1 / 2.2), out=acc)    # linear -> display gamma
    return acc, t


def draw_body(acc, b, ppr, gain, sigma, contrast):
    H, W, _ = acc.shape
    vx, vy, vz = b["v"]
    if vx >= 0:
        return
    ax, ay = scan_angles(vx, vy, vz)
    col, row = project(ax, ay, W, H, ppr)
    col, row = float(col), float(row)
    rad_px = b["ang_r"] * ppr
    if not (-5 * rad_px - 50 < col < W + 5 * rad_px + 50 and -5 * rad_px - 50 < row < H + 5 * rad_px + 50):
        return
    if b["name"] == "Moon":
        draw_moon(acc, col, row, rad_px, b["v"], b["sun_dir"], gain)
    elif b["name"] == "Sun":
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - col, yy - row)
        acc += (min(gain, 1.0) * (60.0 * (d < rad_px) + 3.0 * np.exp(-np.maximum(d - rad_px, 0) / (rad_px * 0.8))))[..., None]
    else:
        flux = star_flux(b["mag"], gain, contrast)
        splat(acc, np.array([col]), np.array([row]), np.array([[1.0, 0.97, 0.9]]) * flux, sigma)


def draw_moon(acc, col, row, rad_px, view, sun_dir, gain):
    """Lambert-lit sphere with the correct phase and orientation."""
    H, W, _ = acc.shape
    r = int(np.ceil(rad_px)) + 2
    c0, r0 = int(round(col)), int(round(row))
    ys, xs = np.mgrid[r0 - r:r0 + r + 1, c0 - r:c0 + r + 1]
    u = (xs - col) / rad_px
    w = -(ys - row) / rad_px
    # anti-aliased disc edge (1 px soft)
    inside = np.clip((1.0 - np.hypot(u, w)) * rad_px + 0.5, 0.0, 1.0)
    # local camera basis at the Moon: forward = view, right ~ +y (east), up ~ +z
    f = np.asarray(view)
    up0 = np.array([0.0, 0.0, 1.0])
    right = np.cross(f, up0); right /= np.linalg.norm(right)
    right = -right if right[1] < 0 else right
    up = np.cross(right, f); up /= np.linalg.norm(up)
    up = -up if up[2] < 0 else up
    depth = np.sqrt(np.clip(1 - u * u - w * w, 0, 1))
    normal = u[..., None] * right + w[..., None] * up - depth[..., None] * f
    lit = np.clip(normal @ np.asarray(sun_dir), 0, None) * inside
    # Earthshine: sunlight reflected by the Earth onto the Moon's night side.
    # The Earth is (almost) behind the camera, so it lights the face we see;
    # strongest near new Moon, when the Moon sees a nearly full Earth. Real
    # earthshine is ~1/5000 of the sunlit part; this is enhanced like the stars.
    # Near new Moon the Moon lies roughly toward the Sun (f . sun_dir ~ +1) and
    # sees a nearly full Earth; near full Moon it sees a new Earth.
    earth_phase = 0.5 * (1 + float(np.dot(f, np.asarray(sun_dir))))
    shine = 0.012 * earth_phase * depth * inside
    ok = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    acc[ys[ok], xs[ok]] += ((lit[ok] * 2.5 + shine[ok]) * min(gain, 1.0))[:, None] * np.array([1.0, 0.98, 0.94])


# --- Compositing ----------------------------------------------------------------
TIME_RE = re.compile(r"(\d{8})_(\d{4})")


def frame_time(path):
    m = TIME_RE.search(os.path.basename(path))
    if not m:
        raise ValueError(f"no YYYYmmdd_HHMM timestamp in {path}")
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)


def composite(frame, stars_rgb, feather=1.0, mask=None):
    """Stars behind the Earth disc, using the frame's alpha, or `mask` (an
    (H, W) alpha in [0, 1]) for frames without one, such as JPEGs. The disc
    never moves in these frames, so one PNG's alpha fits every frame."""
    f = np.asarray(frame.convert("RGBA"), dtype=np.float64) / 255.0
    if mask is not None:
        f[..., 3] = mask
    a = f[..., 3:4]
    if feather > 0:
        a = np.asarray(Image.fromarray((a[..., 0] * 255).astype(np.uint8))
                       .filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(feather)),
                       dtype=np.float64)[..., None] / 255.0
        a = np.maximum(a, f[..., 3:4])  # never let stars eat into the disc itself
    out = stars_rgb * (1 - a) + f[..., :3] * a
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def to_image(rgb):
    return Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), "RGB")


def parse_utc(s):
    t = dt.datetime.fromisoformat(s)
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t


# --- Validation ---------------------------------------------------------------
def validate(args):
    """Score candidate projection settings against the server's graticule layer,
    then draw our graticule over the server's coastlines for a visual check."""
    g = np.asarray(Image.open(os.path.join(DATA, "grat.png")).convert("RGBA"))[..., 3] > 0
    size = g.shape[0]
    # distance-to-nearest-line map (cheap: dilate a few times)
    from PIL import ImageFilter
    lines = Image.fromarray((g * 255).astype(np.uint8))
    near = {d: np.asarray(lines.filter(ImageFilter.MaxFilter(2 * d + 1))) > 0 for d in (0, 1, 2, 4)}

    pts = []
    for lat in range(-80, 81, 10):
        lon = np.linspace(-80, 80, 1601)
        pts.append((np.full_like(lon, lat), lon))
    for lon in range(-80, 81, 10):
        lat = np.linspace(-80, 80, 1601)
        pts.append((lat, np.full_like(lat, lon)))
    lat = np.concatenate([p[0] for p in pts]); lon = np.concatenate([p[1] for p in pts])

    def score(h, sweep, ell):
        c, r = earth_to_pixels(lat, lon, size, h=h, sweep=sweep, ellipsoid=ell)
        ok = np.isfinite(c) & (c >= 0) & (c < size - 0.5) & (r >= 0) & (r < size - 0.5)
        ci, ri = np.round(c[ok]).astype(int), np.round(r[ok]).astype(int)
        return {d: near[d][ri, ci].mean() for d in near}, ok.sum()

    print("fraction of our graticule points within d px of the server's lines")
    print(f"{'sweep':>5} {'ellipsoid':>9} {'h (m)':>12}   d=0    d=1    d=2    d=4")
    best = None
    for sweep in ("y", "x"):
        for ell in (True, False):
            for h in (35_785_831.0, 35_786_000.0, 35_786_023.0):
                s, n = score(h, sweep, ell)
                print(f"{sweep:>5} {str(ell):>9} {h:12.0f}  " + "  ".join(f"{s[d]:.3f}" for d in (0, 1, 2, 4)))
                if best is None or s[1] > best[0][1]:
                    best = (s, sweep, ell, h)
    print(f"\nbest: sweep={best[1]} ellipsoid={best[2]} h={best[3]:.0f}  (module uses sweep={SWEEP}, h={H_SAT:.0f})")

    # visual: server coastlines (white) + our graticule (red)
    coast = np.asarray(Image.open(os.path.join(DATA, "coast.png")).convert("RGBA"))[..., 3] > 0
    img = np.zeros((size, size, 3), np.uint8)
    img[coast] = 255
    img[g] = (90, 90, 90)
    c, r = earth_to_pixels(lat, lon, size)
    ok = np.isfinite(c) & (c >= 0) & (c < size - 0.5) & (r >= 0) & (r < size - 0.5)
    img[np.round(r[ok]).astype(int), np.round(c[ok]).astype(int)] = (255, 40, 40)
    out = args.out or os.path.join(DATA, "validate.png")
    Image.fromarray(img).save(out)
    print(f"overlay written to {out}")


# --- CLI ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--mag-limit", type=float, default=8.0, help="faintest star drawn (default 8)")
    common.add_argument("--gain", type=float, default=1.0, help="star brightness multiplier")
    common.add_argument("--contrast", type=float, default=0.75,
                        help="magnitude contrast, 1 = physical, lower = faint stars brighter")
    common.add_argument("--psf",type=float, default=0.7, help="star sigma in px at 2048 px")
    common.add_argument("--no-bodies", action="store_true", help="skip Sun, Moon, planets")
    common.add_argument("--scan-offset", type=float, default=SCAN_OFFSET_MIN,
                        help="minutes added to frame time (default 5 = mid-scan)")

    r = sub.add_parser("render", parents=[common])
    r.add_argument("--time", required=True, help="UTC, e.g. 2026-11-06T12:00 (used as-is)")
    r.add_argument("--size", type=int, default=2048)
    r.add_argument("-o", "--out", default="stars.png")

    c = sub.add_parser("composite", parents=[common])
    c.add_argument("frames", help="folder of YYYYmmdd_HHMM.png / .jpg frames")
    c.add_argument("--mask", help="PNG frame whose alpha marks the Earth disc; "
                   "required for frames without transparency (JPEG)")
    c.add_argument("-o", "--out", default="frames_sky")
    c.add_argument("--feather", type=float, default=1.0, help="disc-edge blur in px")

    v = sub.add_parser("validate")
    v.add_argument("-o", "--out")

    a = p.parse_args()
    if a.cmd == "validate":
        return validate(a)

    sky = Sky(mag_limit=a.mag_limit)
    kw = dict(gain=a.gain, mag_limit=a.mag_limit, psf=a.psf, bodies=not a.no_bodies, contrast=a.contrast)
    if a.cmd == "render":
        rgb, _ = render(sky, parse_utc(a.time), a.size, **kw)
        to_image(rgb).save(a.out)
        print(f"wrote {a.out}")
        return

    if not os.path.isdir(a.frames):
        sys.exit(f"folder '{a.frames}' not found. Download frames first, e.g.\n"
                 f"  python3 eumetview_frames.py --start 2026-11-06T00:00 --end 2026-11-07T00:00 --out {a.frames}")
    os.makedirs(a.out, exist_ok=True)
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted(f for f in os.listdir(a.frames) if f.lower().endswith(exts) and TIME_RE.search(f))
    if not frames:
        sys.exit(f"no YYYYmmdd_HHMM.png/.jpg frames in {a.frames}")
    mask = None
    if a.mask:
        m = Image.open(a.mask)
        if m.mode != "RGBA":
            sys.exit(f"--mask {a.mask} has no alpha channel")
        mask = np.asarray(m.getchannel("A"), dtype=np.float64) / 255.0
    for name in frames:
        dst = os.path.join(a.out, os.path.splitext(name)[0] + ".png")
        if os.path.exists(dst):
            print(f"skip {dst}")
            continue
        frame = Image.open(os.path.join(a.frames, name))
        if mask is None and "A" not in frame.getbands():
            sys.exit(f"{name} has no transparency: pass --mask <a PNG frame from the same server>")
        if mask is not None and mask.shape != (frame.size[1], frame.size[0]):
            sys.exit(f"--mask is {mask.shape[1]}x{mask.shape[0]} but {name} is {frame.size[0]}x{frame.size[1]}")
        when = frame_time(name) + dt.timedelta(minutes=a.scan_offset)
        rgb, _ = render(sky, when, frame.size[0], **kw)
        composite(frame, rgb, a.feather, mask).save(dst)
        print(f"ok   {dst}  (sky at {when:%Y-%m-%d %H:%M} UTC)")
    print(f"\nDone: {len(frames)} frames in ./{a.out}/")


if __name__ == "__main__":
    main()
