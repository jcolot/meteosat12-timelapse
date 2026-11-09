"""
Single-scattering atmosphere for the geostationary view (Nishita et al. 1993).

For every pixel we march the view ray from the satellite through a spherical
atmosphere (Rayleigh + Mie, exponential density) down to the surface, or past
the limb. At each sample:

    in-scatter += ds * [bR(l) rhoR PR(mu) + bM rhoM PM(mu)] * Tsun(l) * Tview(l)

Tsun comes from a precomputed table of optical depth along the sun ray as a
function of altitude and sun zenith (the Earth's shadow is in the table), so
there is no inner march. We also return the transmittance of sunlight to the
surface and to cloud tops, and of the surface back to the camera, which the
compositor uses to relight the GeoColour image.

Oblateness: the problem is solved in "spherised" coordinates (z scaled by a/b)
so the atmosphere shell hugs the WGS84 ellipsoid and the limb glow sits on the
real limb at the poles too.

All distances in km. Radiances are relative to the solar irradiance (E = 1).
"""
import numpy as np
import cv2

import starfield as sf

A_KM = sf.A_EQ / 1000.0
B_KM = sf.B_POL / 1000.0
SQ = A_KM / B_KM                         # z scale that turns the ellipsoid into a sphere
R_GROUND = A_KM
R_TOP = A_KM + 80.0
SAT_R = A_KM + sf.H_SAT / 1000.0

# Scattering coefficients at ~680 / 550 / 440 nm (R, G, B), per km
BETA_R = np.array([5.802e-3, 13.558e-3, 33.1e-3], np.float32)
H_R = 8.0
BETA_M_SCA = np.float32(3.996e-3)
BETA_M_EXT = np.float32(4.440e-3)
H_M = 1.2
G_MIE = 0.8
# Ozone: absorbs mostly in green/red; gives twilight its blue-violet band.
BETA_O3 = np.array([0.650e-3, 1.881e-3, 0.085e-3], np.float32)

# --- sun optical-depth table --------------------------------------------------
LUT_H = 96          # altitude samples (0..80 km, denser near the ground)
LUT_MU = 256        # cos(sun zenith) samples (-0.35..1)
MU_MIN = -0.35


def _h_of(i):
    return 80.0 * (i / (LUT_H - 1)) ** 2


def _i_of(h):
    return np.sqrt(np.clip(h, 0, 80.0) / 80.0) * (LUT_H - 1)


def _mu_of(j):
    return MU_MIN + (1 - MU_MIN) * j / (LUT_MU - 1)


def _j_of(mu):
    return (np.clip(mu, MU_MIN, 1.0) - MU_MIN) / (1 - MU_MIN) * (LUT_MU - 1)


def ozone_density(h):
    """Tent profile, peak at 25 km (Bruneton's model), relative to 1 at peak."""
    return np.clip(1 - np.abs(h - 25.0) / 15.0, 0, None)


def _build_lut(n_steps=400):
    """Column densities (Rayleigh, Mie, ozone) along the sun ray, in km of
    sea-level-equivalent air. Rays that hit the ground get a huge value."""
    h = _h_of(np.arange(LUT_H))[:, None]
    mu = _mu_of(np.arange(LUT_MU))[None, :]
    r = R_GROUND + h
    # distance to the top of the atmosphere along the ray
    b = r * mu
    t_top = -b + np.sqrt(np.maximum(b * b - r * r + R_TOP ** 2, 0))
    # does the ray hit the ground?
    disc_g = b * b - r * r + R_GROUND ** 2
    hits = (mu < 0) & (disc_g > 0)
    s = (np.arange(n_steps) + 0.5) / n_steps
    out = np.zeros((3, LUT_H, LUT_MU), np.float64)
    for k in range(n_steps):
        t = t_top * s[k]
        rr = np.sqrt(r * r + t * t + 2 * r * t * mu)
        hh = np.maximum(rr - R_GROUND, 0)   # rays into the ground are shadowed below
        ds = t_top / n_steps
        out[0] += np.exp(-hh / H_R) * ds
        out[1] += np.exp(-hh / H_M) * ds
        out[2] += ozone_density(hh) * ds
    # soft shadow edge: blend in over ~1 km of tangent height to avoid a hard step
    tangent = r * np.sqrt(np.maximum(1 - mu * mu, 0)) - R_GROUND
    shadow = np.where(mu < 0, np.clip(1 - tangent / 1.0, 0, 1), 0) * hits
    out = out + shadow[None] * 1e4
    return out.astype(np.float32)


_LUT = None


def lut():
    global _LUT
    if _LUT is None:
        _LUT = _build_lut()
    return _LUT


def sun_depth(h, mu):
    """Column densities (3, ...) along the sun ray from altitude h (km) with
    cos(zenith) mu. Uses cv2.remap for fast bilinear lookups."""
    L = lut()
    shape = np.shape(h)
    mx = _j_of(np.asarray(mu, np.float32)).reshape(-1, 1).astype(np.float32)
    my = _i_of(np.asarray(h, np.float32)).reshape(-1, 1).astype(np.float32)
    out = []
    # cv2.remap maps must be <= 32767 per side; split long vectors
    for k in range(3):
        parts = []
        for s0 in range(0, mx.shape[0], 30000):
            parts.append(cv2.remap(L[k], mx[s0:s0 + 30000], my[s0:s0 + 30000], cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE))
        out.append(np.concatenate(parts, 0).reshape(shape))
    return out


def transmittance(depths):
    """(..., 3) RGB transmittance from column densities (rayleigh, mie, ozone)."""
    dR, dM, dO = depths
    tau = (dR[..., None] * BETA_R + dM[..., None] * BETA_M_EXT
           + dO[..., None] * BETA_O3)
    return np.exp(-tau)


# --- phase functions ------------------------------------------------------------
def phase_r(mu):
    return 3.0 / (16 * np.pi) * (1 + mu * mu)


def phase_m(mu, g=G_MIE):
    # Cornette-Shanks
    g2 = g * g
    return 3.0 / (8 * np.pi) * (1 - g2) * (1 + mu * mu) / ((2 + g2) * (1 + g2 - 2 * g * mu) ** 1.5)


# --- view rays ---------------------------------------------------------------------
def view_rays(size):
    """Unit ray directions (3, size, size) from the satellite for the WMS frame
    pixels (Earth-fixed axes), same GEOS geometry as starfield."""
    h = sf.H_SAT
    c = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(c * sf.HALF, -c * sf.HALF)
    vy = np.tan(X / h)
    vz = np.tan(Y / h) * np.hypot(vy, 1.0)
    d = np.stack([-np.ones_like(vy), vy, vz])
    return (d / np.linalg.norm(d, axis=0)).astype(np.float32)


def reference():
    """Sun and view transmittance for a vertical ray from the ground (noon
    reference): the day side is normalised to this so midday looks like the
    source image."""
    T = transmittance([np.float32(H_R), np.float32(H_M), np.float32(15.0)])
    return T * T          # down and back up


# --- the integrator ------------------------------------------------------------------
class Atmosphere:
    def __init__(self, size=1024, steps=32, cloud_km=6.0):
        self.size = size
        self.steps = steps
        self.cloud_km = cloud_km
        d = view_rays(size)
        self.d = d
        # spherised ray
        o = np.array([SAT_R, 0.0, 0.0], np.float32)
        ds = d.copy()
        ds[2] *= SQ
        self.dlen = np.linalg.norm(ds, axis=0)           # |d'| (km per unit t)
        ds = ds / self.dlen                              # unit in spherised space
        self.ds_unit = ds
        b = (o[:, None, None] * ds).sum(0)
        c_top = SAT_R ** 2 - R_TOP ** 2
        c_gnd = SAT_R ** 2 - R_GROUND ** 2
        disc_t = b * b - c_top
        disc_g = b * b - c_gnd
        self.in_atm = disc_t > 0
        self.hit = disc_g > 0
        t0 = np.where(self.in_atm, -b - np.sqrt(np.maximum(disc_t, 0)), 0)
        t_exit = np.where(self.in_atm, -b + np.sqrt(np.maximum(disc_t, 0)), 0)
        t_g = np.where(self.hit, -b - np.sqrt(np.maximum(disc_g, 0)), 0)
        self.t0 = t0.astype(np.float32)
        self.t1 = np.where(self.hit, t_g, t_exit).astype(np.float32)
        self.o = o

    def view_transmittance(self):
        """(size, size, 3) transmittance surface -> camera. Independent of the
        Sun, so computed once."""
        N, S = self.size, self.steps
        dt = (self.t1 - self.t0) / S
        tau_v = np.zeros((N, N, 3), np.float32)
        for k in range(S):
            t = self.t0 + dt * (k + 0.5)
            p = self.o[:, None, None] + self.ds_unit * t
            hgt = np.maximum(np.linalg.norm(p, axis=0) - R_GROUND, 0)
            ext = (np.exp(-hgt / H_R)[..., None] * BETA_R + np.exp(-hgt / H_M)[..., None] * BETA_M_EXT
                   + ozone_density(hgt)[..., None] * BETA_O3)
            tau_v += ext * dt[..., None]
        return np.where(self.hit[..., None], np.exp(-tau_v), 1).astype(np.float32)

    def sky_transmittance(self):
        """(size, size, 3) transmittance of light from beyond the Earth that
        grazes the limb on its way to the camera (rays that pass through the
        atmosphere but miss the ground). 1 elsewhere; values on the disc are
        irrelevant (hidden). Independent of the Sun."""
        N, S = self.size, self.steps
        dt = (self.t1 - self.t0) / S
        tau = np.zeros((N, N, 3), np.float32)
        for k in range(S):
            t = self.t0 + dt * (k + 0.5)
            p = self.o[:, None, None] + self.ds_unit * t
            hgt = np.maximum(np.linalg.norm(p, axis=0) - R_GROUND, 0)
            ext = (np.exp(-hgt / H_R)[..., None] * BETA_R + np.exp(-hgt / H_M)[..., None] * BETA_M_EXT
                   + ozone_density(hgt)[..., None] * BETA_O3)
            tau += ext * dt[..., None]
        graze = (self.in_atm & ~self.hit)[..., None]
        return np.where(graze, np.exp(-tau), 1).astype(np.float32)

    def compute(self, sun_dir):
        """Returns dict of (size, size, 3) float32 fields:
        inscatter: radiance scattered toward the camera (E = 1)
        t_view:    transmittance surface -> camera
        t_sun:     transmittance sun -> surface (0.5 km, soft)
        t_cloud:   transmittance sun -> cloud top (cloud_km)
        mu_sun:    cos(sun zenith) at the surface point (spherical normal)
        """
        N, S = self.size, self.steps
        sun = np.asarray(sun_dir, np.float32)
        sun_s = sun * np.array([1, 1, SQ], np.float32)
        sun_s /= np.linalg.norm(sun_s)
        mu_view = np.einsum("kij,k->ij", self.d, sun)        # cos scattering angle
        pr, pm = phase_r(mu_view), phase_m(mu_view)

        L = self.t1 - self.t0
        dt = L / S
        insc = np.zeros((N, N, 3), np.float32)
        tau_v = np.zeros((N, N, 3), np.float32)
        for k in range(S):
            t = self.t0 + dt * (k + 0.5)
            p = self.o[:, None, None] + self.ds_unit * t
            r = np.linalg.norm(p, axis=0)
            hgt = np.maximum(r - R_GROUND, 0)
            mu_s = np.einsum("kij,k->ij", p, sun_s) / r
            rho_r = np.exp(-hgt / H_R)
            rho_m = np.exp(-hgt / H_M)
            rho_o = ozone_density(hgt)
            ext = (rho_r[..., None] * BETA_R + rho_m[..., None] * BETA_M_EXT
                   + rho_o[..., None] * BETA_O3)
            seg = dt[..., None]
            tv_mid = np.exp(-(tau_v + 0.5 * ext * seg))
            ts = transmittance(sun_depth(hgt, mu_s))
            sca = (rho_r[..., None] * BETA_R * pr[..., None]
                   + (rho_m * BETA_M_SCA * pm)[..., None])
            insc += sca * ts * tv_mid * seg
            tau_v += ext * seg
        valid = self.in_atm[..., None]
        insc *= valid

        # surface quantities (spherical approx of the normal is fine here)
        p_s = self.o[:, None, None] + self.ds_unit * self.t1
        r_s = np.linalg.norm(p_s, axis=0)
        mu_g = np.einsum("kij,k->ij", p_s, sun_s) / r_s
        t_sun = transmittance(sun_depth(np.full_like(mu_g, 0.5), mu_g))
        t_cloud = transmittance(sun_depth(np.full_like(mu_g, self.cloud_km), mu_g))
        t_view = np.exp(-tau_v)
        hitm = self.hit[..., None]
        return dict(inscatter=insc, t_view=np.where(hitm, t_view, 1),
                    t_sun=t_sun * hitm, t_cloud=t_cloud * hitm, mu_sun=mu_g * self.hit)


def upsample(field, size):
    if field.shape[0] == size:
        return field
    return cv2.resize(field, (size, size), interpolation=cv2.INTER_LINEAR)


def sun_transmittance(mu, h_km):
    """(..., 3) transmittance of sunlight to altitude h_km for cos(zenith) mu
    (any array shape), from the table: cheap enough for every pixel and frame."""
    return transmittance(sun_depth(np.full(np.shape(mu), h_km, np.float32), mu)).astype(np.float32)


class SunCurve:
    """Sunlight transmittance to a fixed altitude as a 1D curve of cos(zenith):
    tabulated once, then a cheap linear lookup for millions of pixels."""
    def __init__(self, h_km, n=131072):
        self.mu = np.linspace(MU_MIN, 1.0, n, dtype=np.float32)
        self.T = sun_transmittance(self.mu, h_km)                 # (n, 3)
        self.n = n

    def __call__(self, mu):
        x = (np.clip(mu, MU_MIN, 1.0) - MU_MIN) * np.float32((self.n - 1) / (1.0 - MU_MIN))
        i = np.minimum(x.astype(np.int32), self.n - 2)
        f = (x - i)[..., None]
        return self.T[i] * (1 - f) + self.T[i + 1] * f
