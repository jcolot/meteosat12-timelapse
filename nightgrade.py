"""
Night-side colour grade for EUMETView GeoColour frames.

GeoColour shows night clouds from infrared, rendered almost white, as bright
as daytime clouds. A night-side Earth seen by eye would show clouds dim and
blue-grey (moonlight/airglow), so this grade, on the night side only:

  * finds night per pixel from the true solar zenith angle at the frame time
    (surface normals of the WGS84 ellipsoid for every frame pixel, Sun from the
    JPL ephemeris), fading in between NIGHT_START and NIGHT_FULL degrees;
  * darkens clouds and shifts them to blue, in proportion to how bright they
    are, so the dark navy background is left as it is;
  * protects city lights (warm pixels: red well above blue).

GeoColour itself switches from the day to the night image at a zenith of about
79-85 deg (measured on the 2026-11-06 frames), hence the 80-88 deg ramp.
"""
import os

import cv2
import numpy as np

import starfield as sf

NIGHT_START = 80.0   # solar zenith (deg) where the grade begins
NIGHT_FULL = 88.0    # ... and where it is fully applied
BASE_RGB = np.array([29.0, 31.0, 41.0], np.float32)  # night background (dark navy, measured)
BASE_L = float(BASE_RGB.mean())
GAIN = 0.45          # fraction of the cloud signal kept at night
TINT = (0.62, 0.75, 1.0)  # colour of night clouds (normalised to blue = 1): muted steel blue
CAP = None           # soft cap on night cloud luminance above the background (None = linear)
BASE_SCALE = 1.0     # brightness kept for the night background (land/ocean, not lights)
LIGHTS_GAIN = None   # city light emission kept (None = leave the source's lights as they are)
LIGHTS_TINT = (1.0, 0.66, 0.36)  # sodium/LED mix: amber, as cities look from orbit
LIGHTS_SOFT = 1.2    # px: slight bloom so lights read as glow, not hard pixels

NORMALS = os.path.join(sf.DATA, "normals.npy")


def surface_normals(size=sf.NATIVE_SIZE):
    """(3, size, size) unit normals (Earth-fixed axes) of the WGS84 surface
    point seen in each pixel of the WMS frame; NaN off the disc."""
    a, b, h = sf.A_EQ, sf.B_POL, sf.H_SAT
    c = (np.arange(size) + 0.5) / size * 2 - 1
    X, Y = np.meshgrid(c * sf.HALF, -c * sf.HALF)
    ax, ay = X / h, Y / h
    # invert the sweep-y scan angles to a ray direction from the satellite
    vy = np.tan(ax)
    vz = np.tan(ay) * np.hypot(vy, 1.0)
    d = np.stack([-np.ones_like(vy), vy, vz])
    d /= np.linalg.norm(d, axis=0)
    S = np.array([a + h, 0.0, 0.0])[:, None, None]
    sc = np.array([1 / a, 1 / a, 1 / b])[:, None, None]
    D, P = d * sc, S * sc
    A = (D * D).sum(0)
    B = 2 * (D * P).sum(0)
    C = (P * P).sum(0) - 1
    disc = B * B - 4 * A * C
    hit = disc > 0
    t = np.where(hit, (-B - np.sqrt(np.where(hit, disc, 0))) / (2 * A), np.nan)
    pt = S + t * d
    n = pt * np.array([1 / a ** 2, 1 / a ** 2, 1 / b ** 2])[:, None, None]
    return (n / np.linalg.norm(n, axis=0)).astype(np.float32)


def load_normals():
    if os.path.exists(NORMALS):
        return np.load(NORMALS)
    n = surface_normals()
    np.save(NORMALS, n)
    return n


def night_weight_band(sun_dir, lo, hi):
    """0 by day .. 1 at night across solar zenith lo..hi (deg), per pixel of
    the frame; sun_dir: unit vector to the Sun, Earth-fixed axes."""
    n = np.nan_to_num(load_normals())
    cz = np.einsum("kij,k->ij", n, np.asarray(sun_dir, np.float32))
    return smoothstep(np.degrees(np.arccos(np.clip(cz, -1, 1))), lo, hi).astype(np.float32)


def smoothstep(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


class NightGrade:
    def __init__(self, gain=GAIN, tint=TINT, cap=CAP, base_scale=BASE_SCALE,
                 lights_gain=LIGHTS_GAIN, lights_tint=LIGHTS_TINT, lights_soft=LIGHTS_SOFT):
        self.normals = np.nan_to_num(load_normals())
        self.gain = gain
        self.tint = np.asarray(tint, np.float32)
        self.cap = cap
        self.base_scale = base_scale
        self.lights_gain = lights_gain
        self.lights_tint = np.asarray(lights_tint, np.float32)
        self.lights_soft = lights_soft

    def night_weight(self, sun_dir):
        """0 by day, 1 at night, smooth through twilight. sun_dir: unit
        vector to the Sun, Earth-fixed axes."""
        cz = np.einsum("kij,k->ij", self.normals, np.asarray(sun_dir, np.float32))
        zen = np.degrees(np.arccos(np.clip(cz, -1, 1)))
        return smoothstep(zen, NIGHT_START, NIGHT_FULL)

    def night(self, img, cloud=None):
        """The fully graded night look (no sun weighting). img: float32 0-255.
        cloud: optional cloud signal above the background (0-255 units, see
        night_cloud) replacing the one read from GeoColour's luminance."""
        L = img.mean(axis=2)
        # Cloud signal = light above the night background. The cloud colour is
        # set by the tint (GeoColour draws high clouds white and low clouds
        # light blue; both become the same muted blue), and its brightness and
        # texture come from the luminance. With a cap, brightness is
        # compressed by a soft knee, cap * (1 - exp(-x / cap)): faint structure
        # keeps its contrast, bright clouds can't stand out.
        c = np.maximum(L - BASE_L, 0) if cloud is None else cloud
        x = c * self.gain
        if self.cap:
            x = self.cap * (1 - np.exp(-x / self.cap))
        tint = self.tint * (3 / self.tint.sum())          # mean 1: tint keeps luminance
        base = img * self.base_scale
        target = BASE_RGB * self.base_scale + x[..., None] * tint
        w = smoothstep(c, 3.0, 30.0)[..., None]           # 0 on bare background
        graded = base + w * (target - base)
        # City lights: strongly warm, or mildly warm and point-like. Night
        # clouds are neutral to blue and extended, so they are not caught,
        # even where city glow tints them a little.
        rb = img[..., 0] - img[..., 2]
        point = np.clip((L - cv2.GaussianBlur(L, (0, 0), 6)) / 15, 0, 1)
        lights = np.maximum(np.clip((rb - 20) / 20, 0, 1),
                            np.clip((rb - 3) / 15, 0, 1) * point)
        if self.lights_gain is None:
            return graded + lights[..., None] * (img - graded)
        # Same exposure as the darkened night: the lights' emission (light
        # above the background) is scaled like everything else, given an
        # amber tint, and softened a little so they glow instead of spiking.
        lw = lights[..., None]
        bg = graded + lw * (BASE_RGB * self.base_scale - graded)   # ground under the city
        emis = np.maximum(L - BASE_L, 0) * lights
        if self.lights_soft:
            emis = 0.6 * emis + 0.4 * cv2.GaussianBlur(emis, (0, 0), self.lights_soft)
        lt = self.lights_tint * (3 / self.lights_tint.sum())
        return bg + (emis * self.lights_gain)[..., None] * lt

    def cloud_signal(self, img, cloud=None):
        """0-1 cloudiness of night pixels (for twilight cloud-top lighting)."""
        c = np.maximum(img.mean(axis=2) - BASE_L, 0) if cloud is None else cloud
        return smoothstep(c, 3.0, 30.0), c

    def __call__(self, img, sun_dir):
        """img: float32 (H, W, 3) in 0-255 at the native frame size."""
        n = self.night_weight(sun_dir)
        if n.max() == 0:
            return img
        return img + n[..., None] * (self.night(img) - img)


# --- Night clouds from the infrared layers ------------------------------------------
# GeoColour's night clouds come from 10.5 um brightness temperature, where low
# cloud is almost as warm as the sea under it and nearly vanishes. EUMETSAT's
# fog / low-cloud RGB (night microphysics: R = IR12.3-IR10.5, G = IR10.5-IR3.8,
# B = IR10.5) picks it out in green: ~0-60 over clear sea, 130-200 on low
# water cloud (measured against GeoColour on the 2026-11-07 frames). The 3.8 um
# channel also sees reflected sunlight, so the RGB is only trusted well past
# sunset (FOG_ZEN). High cloud comes from the raw 10.5 um image, above a local
# clear-sky baseline. The result never goes below GeoColour's own signal.
FOG_ZEN = (92.0, 97.0)   # solar zenith (deg) over which the fog RGB is phased in
LOW_G = (40.0, 220.0)    # fog RGB green: clear sea .. solid low cloud (wide: keeps texture)
LOW_B = (60.0, 150.0)    # fog RGB blue (warmth): excludes cold high cloud
LOW_GR = (-30.0, 10.0)   # green - red: desert shows pink (G-R ~ -50), low cloud cyan (~ +20)
LOW_LEVEL = 70.0         # cloud signal given to solid low cloud (GeoColour's low cloud: ~40-80)
IR_KNEE = 8.0            # grey levels above the clear-sky baseline before IR counts as cloud
IR_GAIN = 0.9            # IR grey levels -> cloud signal


def night_layers(fog, ir):
    """Per source frame, sun-independent: (H, W, 2) float32 0-255 with the
    low-cloud evidence from the fog RGB (H, W, 3/4) and the high-cloud signal
    from the 10.5 um grey (H, W). Warp these like the image for in-betweens,
    then combine with the instant's solar zenith in night_cloud."""
    g, b = fog[..., 1], fog[..., 2]
    low = smoothstep(g, *LOW_G) * smoothstep(b, *LOW_B) * smoothstep(g - fog[..., 0], *LOW_GR)
    if fog.shape[-1] == 4:
        low *= fog[..., 3] / 255.0
    ir = np.ascontiguousarray(ir, np.float32)
    base = cv2.erode(ir, np.ones((41, 41), np.uint8))
    base = cv2.GaussianBlur(base, (0, 0), 25)
    c_ir = np.maximum(ir - base - IR_KNEE, 0)
    return np.dstack([low * 255.0, c_ir]).astype(np.float32)


def night_cloud(geo, layers, zen, low_level=LOW_LEVEL, ir_gain=IR_GAIN):
    """Cloud signal (0-255 units above the night background): GeoColour's
    own (geo, (H, W, 3) 0-255), raised where the infrared layers (from
    night_layers) see low or high cloud. zen: solar zenith (deg) per pixel."""
    c_geo = np.maximum(geo.mean(axis=2) - BASE_L, 0)
    low = layers[..., 0] / 255.0 * smoothstep(zen, *FOG_ZEN)
    c_ir = layers[..., 1] * ir_gain
    return np.maximum(c_geo, np.maximum(low * low_level, c_ir)).astype(np.float32)
