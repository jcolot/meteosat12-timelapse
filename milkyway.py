"""
Milky Way glow behind the Earth, at its true position for every frame.

Source: NASA Scientific Visualization Studio, "Deep Star Maps 2020"
(Ernie Wright), milkyway_2020_8k.exr: the Milky Way background in celestial
(ICRS/J2000) coordinates with the bright Hipparcos and Tycho stars removed,
so it adds the diffuse light and dust lanes without doubling the stars that
starfield.py draws. Equirectangular, RA 0h at the centre, RA increasing to
the left, Dec +90 at the top. Linear light.

Mapping: the camera is fixed to the rotating Earth, so each canvas pixel has
a fixed ray direction in Earth-fixed axes. Per frame that ray is rotated into
the true equator and equinox of date by Greenwich apparent sidereal time, then
back to J2000 by the precession-nutation matrix, and the map is sampled there.
Aberration (~20") is ignored: it is under a pixel and the glow is diffuse.
"""
import os

import cv2
import numpy as np

import starfield as sf

MAP = os.path.join(sf.DATA, "milkyway_8k.npy")      # float16 cache of the EXR
EXR = os.path.join(sf.DATA, "milkyway_2020_8k.exr")
GAIN = 0.35          # map units -> our linear light; tuned to sit with the boosted stars
FLOOR = 0.004        # map level treated as black (faint all-sky background)
SATURATION = 0.45   # NASA's map is warm/sepia; long-exposure photos read more silvery
# Physical calibration of the NASA map: the Scutum star cloud (RA 281.5,
# Dec -6) reads 0.089 map units after 20' smoothing; its surface brightness is
# ~20.5 V mag/arcsec^2 (~815 S10), ~600 S10 once the bright (Tycho) stars the
# map omits are taken out. So 1 map unit ~ 6700 S10.
MAP_TO_S10 = 6700.0
SOFTEN = 0.8         # px (map pixels) of blur: the map's faint-star grain would
                     # otherwise show as blobs when upsampled 4x


def load_map(soften=SOFTEN, rows=None):
    """The map as float32, optionally only rows[0]:rows[1] (cropped before
    conversion and blur, so the full 400 MB float32 map never exists)."""
    if not os.path.exists(MAP):
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        full = cv2.imread(EXR, cv2.IMREAD_UNCHANGED)[..., ::-1]
        np.save(MAP, full.astype(np.float16))
        del full
    m16 = np.load(MAP, mmap_mode="r")
    if rows is not None:
        m16 = m16[rows[0]:rows[1]]
    m = np.ascontiguousarray(m16, dtype=np.float32)
    if soften:
        m = cv2.GaussianBlur(m, (0, 0), soften)
    return np.ascontiguousarray(m)


def canvas_rays(width, height, ppr):
    """(3, H, W) unit ray directions from the satellite, Earth-fixed axes, for a
    canvas centred on the sub-satellite direction (same convention as
    starfield.project): scan angle = (pixel offset) / ppr, sweep-y GEOS."""
    x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2) / ppr
    y = (height / 2 - (np.arange(height, dtype=np.float32) + 0.5)) / ppr
    ax, ay = np.meshgrid(x, y)
    vy = np.tan(ax)
    vz = np.tan(ay) * np.hypot(vy, 1.0)
    d = np.stack([-np.ones_like(vy), vy, vz])
    if sf.SAT_LON:
        lon = np.radians(sf.SAT_LON)
        c, s = np.cos(lon), np.sin(lon)
        d = np.stack([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])
    return (d / np.linalg.norm(d, axis=0)).astype(np.float32)


def to_j2000(sky, when):
    """3x3 rotation: Earth-fixed axes -> GCRS/J2000 equatorial, at `when`."""
    t = sky.ts.from_datetime(when)
    g = np.radians(t.gast * 15.0)
    c, s = np.cos(g), np.sin(g)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)   # Earth-fixed -> true of date
    return (np.asarray(t.M).T @ rz).astype(np.float32)               # -> GCRS (J2000)


class MilkyWay:
    def __init__(self, width, height, ppr, gain=GAIN, floor=FLOOR, soften=SOFTEN, saturation=SATURATION):
        self.mh, self.mw = np.load(MAP, mmap_mode="r").shape[:2]
        self.rays = canvas_rays(width, height, ppr)
        self.gain, self.floor = gain, floor
        self.saturation = saturation
        # keep only the declination strip the canvas can ever see (+ margin):
        # the camera turns about the polar axis, so |Dec| is bounded per pixel
        dmax = np.degrees(np.arcsin(np.abs(self.rays[2]).max())) + 2.0
        r0 = max(0, int(np.floor((90 - dmax) / 180 * self.mh)) - 2)
        r1 = min(self.mh, int(np.ceil((90 + dmax) / 180 * self.mh)) + 2)
        self.r0 = r0
        self.map = load_map(soften, rows=(r0, r1))

    def sample(self, sky, when):
        """(H, W, 3) raw map values (NASA units) along every canvas ray."""
        R = to_j2000(sky, when)
        v = np.einsum("ij,jhw->ihw", R, self.rays)
        ra = np.arctan2(v[1], v[0])                                     # radians, -pi..pi
        dec = np.arcsin(np.clip(v[2], -1, 1))
        mx = (0.5 - ra / (2 * np.pi)) * self.mw - 0.5                   # RA 0 at centre, increasing left
        mx = np.mod(mx, self.mw).astype(np.float32)
        my = ((0.5 - dec / np.pi) * self.mh - 0.5 - self.r0).astype(np.float32)
        return cv2.remap(self.map, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def render_s10(self, sky, when):
        """(H, W, 3) Milky Way surface brightness in S10 units (calibrated)."""
        out = self.sample(sky, when) * MAP_TO_S10
        if self.saturation != 1:
            lum = out.mean(axis=2, keepdims=True)
            out = lum + self.saturation * (out - lum)
        return out

    def render(self, sky, when):
        """(H, W, 3) linear light of the Milky Way alone (legacy look)."""
        out = np.maximum(self.sample(sky, when) - self.floor, 0) * self.gain
        if self.saturation != 1:
            lum = out.mean(axis=2, keepdims=True)
            out = lum + self.saturation * (out - lum)
        return out


# --- Zodiacal light -------------------------------------------------------------
# Leinert et al. 1998, A&AS 127, 1, Table 17: zodiacal light at 500 nm in S10
# units, by helio-ecliptic longitude (lambda - lambda_sun) and ecliptic latitude,
# as transcribed in the open-source `skyscapes` package. -1 = not tabulated
# (too close to the Sun).
ZL_LON = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180], np.float32)
ZL_BETA = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60, 75, 90], np.float32)
ZL_TABLE = np.array([
    [-1, -1, -1, 3140, 1610, 985, 640, 275, 150, 100, 77],
    [-1, -1, -1, 2940, 1540, 945, 625, 271, 150, 100, 77],
    [-1, -1, 4740, 2470, 1370, 865, 590, 264, 148, 100, 77],
    [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100, 77],
    [6400, 4480, 2410, 1410, 910, 635, 454, 237, 141, 99, 77],
    [3840, 2830, 1730, 1100, 749, 545, 410, 223, 136, 97, 77],
    [2480, 1870, 1220, 845, 615, 467, 365, 207, 131, 95, 77],
    [1650, 1270, 910, 680, 510, 397, 320, 193, 125, 93, 77],
    [1180, 940, 700, 530, 416, 338, 282, 179, 120, 92, 77],
    [910, 730, 555, 442, 356, 292, 250, 166, 116, 90, 77],
    [505, 442, 352, 292, 243, 209, 183, 134, 104, 86, 77],
    [338, 317, 269, 227, 196, 172, 151, 116, 93, 82, 77],
    [259, 251, 225, 193, 166, 147, 132, 104, 86, 79, 77],
    [212, 210, 197, 170, 150, 133, 119, 96, 82, 77, 77],
    [188, 186, 177, 154, 138, 125, 113, 90, 77, 74, 77],
    [179, 178, 166, 147, 134, 122, 110, 90, 77, 73, 77],
    [179, 178, 165, 148, 137, 127, 116, 96, 79, 72, 77],
    [196, 192, 179, 165, 151, 141, 131, 104, 82, 72, 77],
    [230, 212, 195, 178, 163, 148, 134, 105, 83, 72, 77],
], np.float32)
EPS_J2000 = np.radians(23.4392911)
# Zodiacal light is reflected sunlight, slightly reddened (Leinert Table 19 vs
# the solar spectrum): warm white.
ZL_TINT = np.array([1.06, 1.0, 0.90], np.float32)
# Inside ~15 deg of the Sun the table has gaps; there the brightness follows a
# power law in elongation (inner zodiacal light / F-corona), capped near the Sun.
ZL_INNER_POW = 2.3
ZL_MIN_ELONG = 3.0


def _log_table():
    """log10 of the table, with the near-Sun gaps filled by the power law in
    elongation from the 15-degree, in-ecliptic value."""
    T = ZL_TABLE.copy()
    for i, lam in enumerate(ZL_LON):
        for j, beta in enumerate(ZL_BETA):
            if T[i, j] < 0:
                el = np.degrees(np.arccos(np.cos(np.radians(lam)) * np.cos(np.radians(beta))))
                T[i, j] = 11500 * (15.0 / max(el, ZL_MIN_ELONG)) ** ZL_INNER_POW
    return np.log10(T).astype(np.float32)


class Zodiacal:
    """Zodiacal light on the canvas, computed on a coarse grid (it is smooth)
    and scaled up."""
    def __init__(self, width, height, ppr, sky, down=8):
        self.W, self.H, self.down = width, height, down
        self.rays = canvas_rays(width // down, height // down, ppr / down)
        self.logT = _log_table()
        self.sun_vec = lambda when: self._sun(sky, when)

    @staticmethod
    def _sun(sky, when):
        t = sky.ts.from_datetime(when)
        v = sky.eph["earth"].at(t).observe(sky.eph["sun"]).apparent().position.au
        return v / np.linalg.norm(v)

    def render_s10(self, sky, when):
        logB = cv2.resize(self.log_lowres(sky, when), (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        return (10 ** logB)[..., None] * ZL_TINT

    def log_lowres(self, sky, when):
        """log10 brightness (S10) on the coarse grid (H/down, W/down)."""
        R = to_j2000(sky, when)
        v = np.einsum("ij,jhw->ihw", R, self.rays)
        ce, se = np.cos(EPS_J2000), np.sin(EPS_J2000)
        # equatorial -> ecliptic (rotate about x by +eps)
        ex, ey, ez = v[0], ce * v[1] + se * v[2], -se * v[1] + ce * v[2]
        sx, sy, sz = self.sun_vec(when)
        sun_lon = np.arctan2(ce * sy + se * sz, sx)
        lam = np.degrees(np.abs((np.arctan2(ey, ex) - sun_lon + np.pi) % (2 * np.pi) - np.pi))
        beta = np.degrees(np.abs(np.arcsin(np.clip(ez, -1, 1))))
        # bilinear in (lam, beta) on the log table, then F-corona power law for
        # anything closer to the Sun than the grid can resolve
        li = np.interp(lam, ZL_LON, np.arange(len(ZL_LON))).astype(np.float32)
        bj = np.interp(beta, ZL_BETA, np.arange(len(ZL_BETA))).astype(np.float32)
        logB = cv2.remap(self.logT, bj[..., None].reshape(bj.shape), li.reshape(li.shape),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # (near-Sun gaps are already filled with the inner power law in the
        # table itself; interpolating in log space keeps it seamless)
        return logB.astype(np.float32)


# --- Shared sky backdrop ----------------------------------------------------------
ZL_FLOOR = 1500.0   # S10: the zodiacal light's own black point (its smooth all-sky
                    # baseline is removed, as in astrophoto background extraction):
                    # only the glow near the Sun remains
ZL_KNEE = 300.0     # S10: soft toe for that black point
ZL_PEAK = 500.0     # S10: soft ceiling on what remains, so the glow nearest the Sun
                    # tops out near the brightest Milky Way star clouds
SKY_FLOOR = 350.0   # S10: soft black point applied to Milky Way + zodiacal light together
SKY_KNEE = 80.0     # S10: width of the soft toe (no hard contour at the black point)
SKY_LOG_K = 300.0   # S10: above the black point, brightness is shown on a log scale
                    # (equal steps per doubling, like the eye): A * ln(1 + excess / K)
SKY_LOG_A = 0.030   # linear light: puts the Scutum star cloud where the approved Milky Way look had it


class Backdrop:
    """Milky Way + zodiacal light on one physical scale (S10), one black point,
    one gain, so their brightness ratio stays true."""
    def __init__(self, width, height, ppr, sky, mw=1.0, zodi=1.0, floor=SKY_FLOOR, k=SKY_LOG_K, a=SKY_LOG_A,
                 zl_floor=ZL_FLOOR, zl_peak=ZL_PEAK):
        self.mw = MilkyWay(width, height, ppr) if mw > 0 else None
        self.zl = Zodiacal(width, height, ppr, sky) if zodi > 0 else None
        self.k_mw, self.k_zl = mw, zodi
        self.floor, self.k, self.a = floor, k, a
        self.zl_floor, self.zl_peak = zl_floor, zl_peak

    def render(self, sky, when, parts=False, mw_s10=None):
        tot = 0
        if mw_s10 is not None:                      # e.g. looked up from a sky strip
            mw = mw_s10 * self.k_mw
        else:
            mw = self.mw.render_s10(sky, when) * self.k_mw if self.mw else 0
        zl = self.zl.render_s10(sky, when) * self.k_zl if self.zl else 0
        if self.zl is not None and self.zl_floor:
            zl_l = zl.mean(axis=2, keepdims=True)
            kept = ZL_KNEE * np.logaddexp(0, (zl_l - self.zl_floor) / ZL_KNEE)
            if self.zl_peak:
                kept = self.zl_peak * (1 - np.exp(-kept / self.zl_peak))
            # the removed baseline is added back as a flat level just under the
            # shared black point, so it never lifts empty sky
            zl = zl * (kept / np.maximum(zl_l, 1e-6))
        tot = mw + zl
        lum = np.mean(tot, axis=2, keepdims=True)
        # soft toe: ~0 below the floor, linear above it
        x = (lum - self.floor) / SKY_KNEE
        lifted = SKY_KNEE * np.logaddexp(0, x)
        y = self.a * np.log1p(lifted / self.k)
        out = tot * (y / np.maximum(lum, 1e-6))
        return (out, mw, zl) if parts else out
