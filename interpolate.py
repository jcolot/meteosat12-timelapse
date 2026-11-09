#!/usr/bin/env python3
"""
Smooth timelapse: motion-compensated in-between frames for the Earth, plus the
exact sky for every in-between time, streamed straight into ffmpeg.

Earth: dense optical flow (OpenCV DIS) between consecutive frames, measured on
a contrast-normalised image so the day/night terminator's large-scale
brightness ramp is not mistaken for motion. Each in-between frame advects both
neighbours along the flow (semi-Lagrangian step) and blends them. Where the
forward and backward flows disagree (clouds forming or dissolving), the flow is
damped towards a plain cross-fade instead of guessing.

Sky: rendered exactly at each in-between time (starfield.render), never
interpolated.

  .venv/bin/python interpolate.py frames_jpg --mask frames/20261106_0000.png \
      --steps 10 --fps 24 -o earth_smooth.mp4
"""
import argparse
import json
import time
import datetime as dt
import os
import subprocess
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image, ImageFilter

import atmosphere as at
import nightgrade as ng
import milkyway as mwy
import relight as rl
import skystrip as ss
import starfield as sf

FLOW_SCALE = 0.5  # flow is estimated at half resolution, then upsampled
SHARPEN = 0.25    # unsharp amount at the midpoint of an interval (tuned so
                  # in-betweens match the source frames' sharpness)


# --- Earth interpolation -----------------------------------------------------------
def nearest_inside(mask):
    """For every pixel, the (row, col) of the nearest pixel inside the disc.
    Used to extend the Earth image outward so warps near the limb never pull
    in black space."""
    outside = (mask < 0.5).astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(outside, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    inside_rc = np.argwhere(outside == 0)          # label k -> k-th zero pixel, row-major
    idx = labels - 1
    return inside_rc[idx, 0], inside_rc[idx, 1]


def flow_input(img, mask_small):
    """Grey, high-pass, locally contrast-normalised, uint8 (for DIS)."""
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.resize(g, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)
    hp = g - cv2.GaussianBlur(g, (0, 0), 12)
    sd = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 12) + 4.0)
    n = np.clip(128 + 40 * hp / sd, 0, 255)
    n[mask_small < 0.5] = 128
    return n.astype(np.uint8)


class Interp:
    def __init__(self, mask):
        self.mask = mask
        self.H, self.W = mask.shape
        self.fill_r, self.fill_c = nearest_inside(mask)
        self.mask_small = cv2.resize(mask.astype(np.float32), None, fx=FLOW_SCALE, fy=FLOW_SCALE,
                                     interpolation=cv2.INTER_AREA)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        self.xx, self.yy = xx, yy

    def extend(self, img):
        return img[self.fill_r, self.fill_c]

    def flows(self, a, b):
        fa, fb = flow_input(a, self.mask_small), flow_input(b, self.mask_small)
        f01 = self.dis.calc(fa, fb, None)
        f10 = self.dis.calc(fb, fa, None)
        v = 0.5 * (f01 - f10)                    # symmetric velocity estimate
        err = np.linalg.norm(f01 + f10, axis=2)  # forward/backward disagreement
        conf = np.exp(-(err / 3.0) ** 2)         # 1 = consistent, 0 = don't trust
        v *= conf[..., None]
        v = cv2.GaussianBlur(v, (0, 0), 3)
        v[self.mask_small < 0.5] = 0
        v = cv2.resize(v, (self.W, self.H), interpolation=cv2.INTER_LINEAR) / FLOW_SCALE
        return v

    def between(self, a, b, v, s, sharpen=SHARPEN):
        """Earth image at fraction s in (0, 1) between frames a and b.
        Cubic resampling, then a light unsharp mask that peaks at s=0.5:
        blending two frames softens fine texture that changed between them,
        and without compensation sharpness would pulse every source interval."""
        a0 = cv2.remap(a, self.xx - s * v[..., 0], self.yy - s * v[..., 1],
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        b1 = cv2.remap(b, self.xx + (1 - s) * v[..., 0], self.yy + (1 - s) * v[..., 1],
                       cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out = (1 - s) * a0.astype(np.float32) + s * b1.astype(np.float32)
        k = sharpen * 4 * s * (1 - s)
        if k > 0:
            out += k * (out - cv2.GaussianBlur(out, (0, 0), 1.0))
        return np.clip(out, 0, 255)


# --- Worker -------------------------------------------------------------------------
G = {}


def _exit_with_parent():
    """Workers quit by themselves if the main process goes away (killed, crashed),
    so a stopped render never leaves orphans rendering into nothing."""
    import threading
    parent = os.getppid()

    def watch():
        while True:
            time.sleep(2)
            if os.getppid() != parent:
                os._exit(1)
    threading.Thread(target=watch, daemon=True).start()


def init_worker(mask, feather_alpha, sky_kw, geom, night, look, credit):
    _exit_with_parent()
    G["interp"] = Interp(mask)
    G["credit"] = credit
    G["sky"] = sf.Sky(mag_limit=sky_kw["mag_limit"])
    G["strip"] = ss.SkyStrip(look["strip"]) if look.get("strip") else None
    G["backdrop"] = (mwy.Backdrop(geom["W"], geom["H"], geom["ppr"], G["sky"],
                                  mw=look["milkyway"], zodi=look["zodiacal"],
                                  a=mwy.SKY_LOG_A * look.get("sky_strength", 1.0))
                     if (look.get("milkyway") or look.get("zodiacal")) else None)
    # starlight and sky glow grazing the limb pass through the atmosphere
    G["sky_t"] = place(sky_transmittance(), geom, fill=1.0) if look.get("limb_fade", True) else None
    G["look"] = look
    G["gpu"] = None
    G["night"] = None
    G["relight"] = None
    if look["mode"] == "grade":
        G["night"] = ng.NightGrade(gain=night["gain"], tint=night["tint"])
    elif look["mode"] == "relight":
        G["relight"] = rl.Relight(
            k_air=look["k_air"], cos_pow=look["cos_pow"],
            night=ng.NightGrade(gain=night["gain"], tint=night["tint"], cap=look["night_cap"],
                                base_scale=look["night_base"], lights_gain=look["lights_gain"]))
    G["alpha"] = feather_alpha
    G["kw"] = sky_kw
    if look.get("device", "cpu") != "cpu" and G["relight"] is not None and G["strip"] is not None:
        import gpu   # imported only here: CPU runs don't need torch
        G["gpu"] = gpu.FrameGPU(G["relight"], G["backdrop"], geom, feather_alpha, G["sky_t"], credit,
                                dev=look["device"])
    G["geom"] = geom


def canvas_geometry(src, width, height, earth_px):
    """Where the src x src Earth frame goes on a width x height canvas.
    earth_px is the size the frame is scaled to; its centre stays on the
    canvas centre so the GEOS projection carries on exactly into the sky."""
    if not width:
        return dict(W=src, H=src, E=src, x0=0, y0=0, ppr=sf.H_SAT * src / (2 * sf.HALF))
    E = earth_px
    if (width - E) % 2 or (height - E) % 2:
        raise SystemExit("canvas minus earth size must be even on both axes (keeps the centre exact)")
    return dict(W=width, H=height, E=E, x0=(width - E) // 2, y0=(height - E) // 2,
                ppr=sf.H_SAT * E / (2 * sf.HALF))


def place(img, g, fill=0.0):
    """Scale a src-sized float image to E and put it on the canvas."""
    if g["E"] != img.shape[0]:
        img = cv2.resize(img, (g["E"], g["E"]), interpolation=cv2.INTER_AREA)
    shape = (g["H"], g["W"]) + img.shape[2:]
    out = np.full(shape, fill, np.float32)
    out[g["y0"]:g["y0"] + g["E"], g["x0"]:g["x0"] + g["E"]] = img
    return out


def cache_path(cache_dir, when):
    return os.path.join(cache_dir, when.strftime("%Y%m%d_%H%M%S") + ".npy")


def atm_worker_init(mag_limit):
    _exit_with_parent()
    G["atm"] = at.Atmosphere(size=1024, steps=32)
    G["sky"] = sf.Sky(mag_limit=1.0)


def atm_job(args):
    """Compute and cache the in-scatter field for one source-frame time."""
    cache_dir, when = args
    path = cache_path(cache_dir, when)
    if not os.path.exists(path):
        ins = G["atm"].compute(G["sky"].sun_dir(when))["inscatter"]
        np.save(path + ".tmp.npy", ins.astype(np.float16))
        os.replace(path + ".tmp.npy", path)
    return path


CREDIT_FONT = "/System/Library/Fonts/HelveticaNeue.ttc"


def credit_overlay(W, H, text, opacity=0.7):
    """(H, W, 1) alpha and (3,) colour of a small credit line, bottom right."""
    from PIL import ImageDraw, ImageFont
    size = max(12, round(H / 72))                     # 30 px at 2160
    try:
        font = ImageFont.truetype(CREDIT_FONT, size)
    except OSError:
        font = ImageFont.load_default()
    layer = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(layer)
    margin = round(H / 40)
    bbox = d.textbbox((0, 0), text, font=font)
    x = W - margin - (bbox[2] - bbox[0])
    y = H - margin - (bbox[3] - bbox[1])
    d.text((x - bbox[0], y - bbox[1]), text, font=font, fill=255)
    return np.asarray(layer, np.float32)[..., None] / 255.0 * opacity


SKY_T = os.path.join(sf.DATA, "sky_transmittance_2048.npy")


def sky_transmittance():
    """Limb transmittance for the 2048 px frame (Sun-independent; cached)."""
    if os.path.exists(SKY_T):
        return np.load(SKY_T).astype(np.float32)
    T = at.Atmosphere(size=2048, steps=48).sky_transmittance()
    np.save(SKY_T, T.astype(np.float16))
    return T


def load(path):
    return np.asarray(Image.open(path).convert("RGB"))


def night_layers(look, path):
    """Infrared night layers for the source frame at `path` (matched by name)."""
    stem = os.path.splitext(os.path.basename(path))[0]

    def find(d):
        for ext in (".jpg", ".png"):
            q = os.path.join(d, stem + ext)
            if os.path.exists(q):
                return q
        raise FileNotFoundError(f"no {stem} in {d}")
    fog = np.asarray(Image.open(find(look["night_fog"])).convert("RGB"), np.float32)
    ir = np.asarray(Image.open(find(look["night_ir"])).convert("L"), np.float32)
    return ng.night_layers(fog, ir)


def seg_path(parts, i):
    return os.path.join(parts, f"seg_{i:04d}.mp4")


def segment_job(args):
    """Render one interval and encode it to its own segment file."""
    i, job, enc = args
    t0 = time.time()
    final = seg_path(enc["parts"], i)
    tmp = final + ".part.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{enc['W']}x{enc['H']}", "-r", str(enc["fps"]), "-i", "-", "-vf", "format=yuv420p",
         "-c:v", "libx264", "-crf", str(enc["crf"]), "-preset", "medium", tmp],
        stdin=subprocess.PIPE)
    n = 0
    for k, f in enumerate(interval(job, i * enc["steps"], enc["stride"])):   # streamed to ffmpeg
        ff.stdin.write(f.tobytes())
        if enc["dump"]:
            Image.fromarray(f).save(os.path.join(enc["dump"], f"{i * enc['steps'] + k:05d}.jpg"), quality=92)
        n += 1
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on segment {i}")
    os.replace(tmp, final)
    return i, n, time.time() - t0


def interval(job, first=0, stride=1):
    """Render the frames for one interval [t_a, t_b): yields uint8 frames."""
    path_a, path_b, t_a, t_b, steps, offset_min, last = job
    ip, alpha, sky, kw, g = G["interp"], G["alpha"], G["sky"], G["kw"], G["geom"]
    look = G["look"]
    a = ip.extend(load(path_a))
    b = ip.extend(load(path_b))
    v = ip.flows(a, b)
    # optional infrared night layers, carried along the same motion as the image
    nl_a = nl_b = None
    if look.get("night_fog") and G["relight"] is not None:
        nl_a, nl_b = (ip.extend(night_layers(look, p)) for p in (path_a, path_b))
    off = dt.timedelta(minutes=offset_min)
    if G["relight"] is not None:
        ins_a = np.load(cache_path(look["cache"], t_a + off)).astype(np.float32)
        ins_b = np.load(cache_path(look["cache"], t_b + off)).astype(np.float32)
    shutter = look["shutter"] * (t_b - t_a).total_seconds() / 60.0 / steps
    ks = range(steps + 1) if last else range(steps)
    # --stride: keep only every n-th output frame (global index first + k),
    # for faster-than-real-time cuts rendered with their own, longer shutter
    ks = [k for k in ks if (first + k) % stride == 0]
    for k in ks:
        s = k / steps
        earth = a.astype(np.float32) if k == 0 else b.astype(np.float32) if k == steps else ip.between(a, b, v, s)
        when = t_a + (t_b - t_a) * s + off
        cloud = None
        if nl_a is not None:
            nl = nl_a if k == 0 else nl_b if k == steps else ip.between(nl_a, nl_b, v, s, sharpen=0)
            mu = G["relight"].mu_rows(sky.sun_dir(when))
            zen = np.degrees(np.arccos(np.clip(mu, -1, 1)))
            cloud = ng.night_cloud(np.asarray(earth, np.float32), nl, zen)
        if G["gpu"] is not None:
            # per-pixel work on the GPU; catalogue, Moon, strip lookups on the CPU
            Fg = G["gpu"]
            acc = ss.render(G["strip"], sky, when, g["ppr"], shutter_min=shutter, tone=False, **kw)
            B = G["backdrop"]
            back = None
            if B is not None:
                mw = G["strip"].mw_s10(sky, when) if B.mw is not None else None
                zl = B.zl.log_lowres(sky, when) if B.zl is not None else None
                back = Fg.backdrop_tone(mw, zl)
            ins = (1 - s) * ins_a + s * ins_b
            el, an = Fg.relight(np.asarray(earth, np.float32), sky.sun_dir(when), ins, cloud)
            yield Fg.compose(acc, back, el, an)
            continue
        if G["strip"] is not None:
            stars = ss.render(G["strip"], sky, when, g["ppr"], shutter_min=shutter, **kw)
        else:
            stars, _ = sf.render(sky, when, width=g["W"], height=g["H"], ppr=g["ppr"],
                                 shutter_min=shutter, **kw)
        sky_lin = rl.srgb_to_lin(stars)
        if G["backdrop"] is not None:
            # Milky Way + zodiacal light, one physical scale, in linear light
            mw = (G["strip"].mw_s10(sky, when)
                  if G["strip"] is not None and G["backdrop"].mw is not None else None)
            sky_lin = sky_lin + G["backdrop"].render(sky, when, mw_s10=mw)
        if G["sky_t"] is not None:
            sky_lin = sky_lin * G["sky_t"]
        stars = rl.lin_to_srgb(sky_lin)
        if G["relight"] is not None:
            # in-scatter blended between the two cached source frames; the
            # rest of the lighting is exact for this instant
            ins = (1 - s) * ins_a + s * ins_b
            earth_lin, air = G["relight"](np.asarray(earth, np.float32), sky.sun_dir(when), ins, cloud)
            lin = (sky_lin * (1 - alpha) + place(earth_lin, g) * alpha
                   + place(air, g) * (1 - alpha))
            frame = rl.lin_to_srgb(lin)
        else:
            if G["night"] is not None:
                earth = G["night"](np.asarray(earth, np.float32), sky.sun_dir(when))
            frame = stars * (1 - alpha) + place(earth / 255.0, g) * alpha
        if G["credit"] is not None:
            frame = frame * (1 - G["credit"]) + 0.92 * G["credit"]
        yield (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)


# --- Main ------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("frames", help="folder of YYYYmmdd_HHMM.jpg/.png Earth frames")
    p.add_argument("--mask", required=True, help="PNG frame whose alpha marks the Earth disc")
    p.add_argument("--steps", type=int, default=10, help="output frames per source interval")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--workers", type=int,
                   help="parallel worker processes (default: 1 with --device mps, which is fastest "
                        "since workers contend for the GPU; otherwise CPU cores - 2)")
    p.add_argument("--limit", type=int, help="only the first N intervals (for testing)")
    p.add_argument("--start-at", type=int, default=0, help="skip the first N intervals")
    p.add_argument("--canvas", help="output size WxH, e.g. 3840x2160 (default: the source frame)")
    p.add_argument("--earth-px", type=int, default=1740,
                   help="size the source frame is scaled to on the canvas (default 1740)")
    p.add_argument("--look", choices=["relight", "grade", "source"], default="relight",
                   help="relight: Nishita twilight + dark capped night (default); grade: blue night "
                        "clouds only; source: GeoColour as delivered")
    p.add_argument("--no-night-grade", action="store_true", help="same as --look source")
    p.add_argument("--night-gain", type=float, help="night cloud brightness kept "
                   "(default 0.35 with relight, %s with grade)" % ng.GAIN)
    p.add_argument("--night-cap", type=float, default=22.0, help="relight: soft cap on night clouds")
    p.add_argument("--night-base", type=float, default=0.55, help="relight: night background brightness")
    p.add_argument("--lights-gain", type=float, default=0.4, help="relight: city light brightness")
    p.add_argument("--k-air", type=float, default=0.35, help="relight: strength of the scattered sky light")
    p.add_argument("--cos-pow", type=float, default=0.6, help="relight: day-side falloff exponent")
    p.add_argument("--shutter", type=float, default=0.5,
                   help="star motion blur, as a fraction of the time between output frames (0 = off)")
    p.add_argument("--milkyway", type=float, default=1.0,
                   help="Milky Way glow, relative to its calibrated brightness (NASA SVS Deep Star "
                        "Maps 2020; 0 = off)")
    p.add_argument("--sky-strength", type=float, default=0.6,
                   help="overall brightness of Milky Way + zodiacal light (1 = calibrated look; default %(default)s)")
    p.add_argument("--zodiacal", type=float, default=1.0,
                   help="zodiacal light, relative to Leinert et al. 1998 (0 = off)")
    p.add_argument("--no-limb-fade", action="store_true",
                   help="don't dim/redden starlight grazing the Earth's atmosphere")
    p.add_argument("--credit", help="on-screen credit line (default: EUMETSAT attribution for "
                   "the frames' year, plus star/ephemeris sources)")
    p.add_argument("--no-credit", action="store_true", help="no on-screen credit (metadata is still written)")
    p.add_argument("--body-shutter", type=float, default=1.0,
                   help="Moon/planet motion blur relative to the stars' (e.g. 0.25 keeps the crescent crisp in fast cuts)")
    p.add_argument("--stride", type=int, default=1,
                   help="keep every n-th frame (n x faster video); pair with --shutter n*0.5 for natural motion blur")
    p.add_argument("--night-fog", help="folder of fog / low-cloud RGB frames (same names): "
                   "night clouds rebuilt from infrared, see nightgrade.night_layers")
    p.add_argument("--night-ir", help="folder of 10.5 um IR frames (same names), with --night-fog")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"],
                   help="mps: per-pixel work on the Apple Metal GPU (needs torch; relight look + strip sky)")
    p.add_argument("--sky-mode", choices=["strip", "direct"], default="strip",
                   help="strip: stars + Milky Way pre-rendered once per day and looked up per frame "
                        "(fast); direct: drawn per frame (the original method)")
    p.add_argument("--atm-cache", default=os.path.join(sf.DATA, "atm_cache"),
                   help="where per-source-frame in-scatter fields are cached")
    p.add_argument("--night-tint", default=",".join(str(x) for x in ng.TINT),
                   help="night cloud colour R,G,B relative to blue (default %(default)s)")
    p.add_argument("--dump", help="also write every output frame as JPEG into this folder")
    p.add_argument("--mag-limit", type=float, default=8.0)
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--contrast", type=float, default=0.75)
    p.add_argument("--psf", type=float, default=0.7)
    p.add_argument("--scan-offset", type=float, default=sf.SCAN_OFFSET_MIN)
    p.add_argument("-o", "--out", default="earth_smooth.mp4")
    p.add_argument("--parts", help="segment folder (default: <out>.parts); reruns resume from it")
    p.add_argument("--force", action="store_true", help="reuse a segment folder even if settings changed")
    a = p.parse_args()
    if a.workers is None:
        a.workers = 1 if a.device != "cpu" else max(1, (os.cpu_count() or 2) - 2)

    exts = (".png", ".jpg", ".jpeg")
    names = sorted(f for f in os.listdir(a.frames) if f.lower().endswith(exts) and sf.TIME_RE.search(f))
    if len(names) < 2:
        sys.exit("need at least two frames")
    times = [sf.frame_time(n) for n in names]
    gaps = [(t2 - t1) for t1, t2 in zip(times, times[1:])]
    if len(set(gaps)) != 1:
        print(f"warning: uneven frame spacing ({sorted(set(gaps))}); in-betweens follow real times", file=sys.stderr)

    m = Image.open(a.mask)
    if m.mode != "RGBA":
        sys.exit(f"--mask {a.mask} has no alpha channel")
    mask = np.asarray(m.getchannel("A"), dtype=np.float32) / 255.0
    feathered = np.asarray(m.getchannel("A").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32) / 255.0
    alpha_src = np.maximum(mask, feathered)
    cw, ch = (int(x) for x in a.canvas.lower().split("x")) if a.canvas else (None, None)
    geom = canvas_geometry(mask.shape[0], cw, ch, a.earth_px)
    alpha = place(alpha_src, geom)[..., None]

    jobs = []
    for i in range(len(names) - 1):
        jobs.append((os.path.join(a.frames, names[i]), os.path.join(a.frames, names[i + 1]),
                     times[i], times[i + 1], a.steps, a.scan_offset, i == len(names) - 2))
    idx = list(range(len(jobs)))[a.start_at:]
    jobs = jobs[a.start_at:]
    if a.limit:
        jobs, idx = jobs[:a.limit], idx[:a.limit]
        jobs[-1] = jobs[-1][:-1] + (True,)

    H, W = geom["H"], geom["W"]
    total = sum(a.steps + (1 if j[-1] else 0) for j in jobs)
    print(f"{len(jobs)} intervals x {a.steps} steps -> {total} frames, "
          f"{total / a.fps:.1f} s at {a.fps} fps, {W}x{H}, {a.workers} workers")

    # Attribution required by the EUMETSAT Data Policy (Art. 6.3) for Core
    # data under CC BY 4.0: "[Contains modified] EUMETSAT [Meteosat] [data]
    # [year]". Stars and ephemeris credited as good practice.
    year = times[0].year
    eumetsat = f"Contains modified EUMETSAT Meteosat data {year}"
    credit_text = a.credit or (f"{eumetsat} (CC BY 4.0)  \u00b7  Stars: ESA Hipparcos  \u00b7  "
                               f"Moon & planets: NASA/JPL DE421"
                               + ("  \u00b7  Milky Way: NASA SVS" if a.milkyway > 0 else "")
                               + ("  \u00b7  Zodiacal light: Leinert et al. 1998" if a.zodiacal > 0 else ""))
    credit = None if a.no_credit else credit_overlay(W, H, credit_text)
    meta = ["-metadata", f"copyright={eumetsat}, CC BY 4.0",
            "-metadata", f"comment={credit_text}. Imagery: EUMETView, MTG-I1 FCI GeoColour RGB. "
                         f"Sky, interpolation and night/twilight rendering added.",
            "-metadata", f"title=Earth from MTG-I1, {times[0]:%Y-%m-%d}"]
    if a.dump:
        os.makedirs(a.dump, exist_ok=True)
    mode = "source" if a.no_night_grade else a.look
    night_gain = a.night_gain if a.night_gain is not None else (0.35 if mode == "relight" else ng.GAIN)
    night = dict(gain=night_gain, tint=tuple(float(x) for x in a.night_tint.split(",")))
    look = dict(mode=mode, k_air=a.k_air, cos_pow=a.cos_pow, night_cap=a.night_cap,
                night_base=a.night_base, lights_gain=a.lights_gain, shutter=a.shutter, cache=a.atm_cache,
                milkyway=a.milkyway, zodiacal=a.zodiacal, limb_fade=not a.no_limb_fade,
                sky_strength=a.sky_strength, device=a.device,
                night_fog=a.night_fog, night_ir=a.night_ir)
    if mode == "relight":
        # the expensive in-scatter march: once per source frame, cached on disk
        os.makedirs(a.atm_cache, exist_ok=True)
        off = dt.timedelta(minutes=a.scan_offset)
        need = sorted({j[2] + off for j in jobs} | {j[3] + off for j in jobs})
        todo = [(a.atm_cache, w) for w in need if not os.path.exists(cache_path(a.atm_cache, w))]
        if todo:
            print(f"atmosphere: computing {len(todo)} of {len(need)} source-frame fields", flush=True)
            with Pool(a.workers, initializer=atm_worker_init, initargs=(1.0,)) as pool:
                for i, _ in enumerate(pool.imap_unordered(atm_job, todo), 1):
                    if i % 10 == 0 or i == len(todo):
                        print(f"  atmosphere {i}/{len(todo)}", flush=True)
    sky_kw = dict(gain=a.gain, mag_limit=a.mag_limit, psf=a.psf, contrast=a.contrast, bodies=True)

    # --- chunked, resumable rendering --------------------------------------
    # One segment per source interval, encoded by the worker that rendered
    # it, written under a temporary name and renamed when complete. A rerun
    # skips finished segments; interrupting loses only segments in progress.
    parts = a.parts or os.path.splitext(a.out)[0] + ".parts"
    os.makedirs(parts, exist_ok=True)
    settings = {k: (str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v)
                for k, v in sorted(vars(a).items())
                if k not in ("workers", "limit", "start_at", "dump", "parts", "force", "out")}
    sfile = os.path.join(parts, "settings.json")
    if os.path.exists(sfile):
        old = json.load(open(sfile))
        if old != settings and not a.force:
            diff = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
            sys.exit(f"{parts} holds segments rendered with different settings ({', '.join(diff)}). "
                     f"Use another --parts folder, delete it, or pass --force to mix them.")
    json.dump(settings, open(sfile, "w"), indent=1)

    enc = dict(W=W, H=H, fps=a.fps, crf=a.crf, parts=parts, dump=a.dump, steps=a.steps, stride=a.stride)
    seg_jobs = [(i, j, enc) for i, j in zip(idx, jobs)]
    todo = [sj for sj in seg_jobs if not os.path.exists(seg_path(parts, sj[0]))]
    print(f"segments: {len(seg_jobs) - len(todo)} already done, {len(todo)} to render  ({parts})", flush=True)
    sky_kw = dict(gain=a.gain, mag_limit=a.mag_limit, psf=a.psf, contrast=a.contrast, bodies=True,
                  body_shutter=a.body_shutter)
    if a.sky_mode == "strip" and todo:
        span = [j[2] for j in jobs] + [j[3] for j in jobs]
        day_mid = min(span) + (max(span) - min(span)) / 2 + dt.timedelta(minutes=a.scan_offset)
        shutter_min = a.shutter * (jobs[0][3] - jobs[0][2]).total_seconds() / 60.0 / a.steps
        look["strip"] = ss.build(day_mid, W, H, geom["ppr"], shutter_min, os.path.join(sf.DATA, "skystrip"),
                                 gain=a.gain, mag_limit=a.mag_limit, contrast=a.contrast, psf=a.psf,
                                 log=lambda m: print(m, flush=True))
    t0 = time.time()
    done_now = 0
    frames_left = sum(sj[1][4] + (1 if sj[1][-1] else 0) for sj in todo)
    progress = os.path.join(parts, "progress.json")
    if todo:
        with Pool(a.workers, initializer=init_worker,
                  initargs=(mask, alpha, sky_kw, geom, night, look, credit)) as pool:
            for i, nf, secs in pool.imap_unordered(segment_job, todo):
                done_now += 1
                frames_left -= nf
                el = time.time() - t0
                rate = el / done_now
                eta = rate * (len(todo) - done_now)
                done_total = len(seg_jobs) - len(todo) + done_now
                print(f"  seg {i:04d} done ({nf} frames, {secs:.0f}s)  {done_total}/{len(seg_jobs)}  "
                      f"elapsed {el / 60:.1f} min  ETA {eta / 60:.1f} min", flush=True)
                json.dump(dict(done=done_total, total=len(seg_jobs), elapsed_min=round(el / 60, 1),
                               eta_min=round(eta / 60, 1), last=i, updated=time.strftime("%H:%M:%S")),
                          open(progress, "w"))

    # --- join (no re-encode) -------------------------------------------------
    missing = [i for i in idx if not os.path.exists(seg_path(parts, i))]
    if missing:
        sys.exit(f"{len(missing)} segments missing (e.g. {missing[:5]}); rerun to finish them")
    lst = os.path.join(parts, "concat.txt")
    with open(lst, "w") as f:
        for i in idx:
            f.write(f"file '{os.path.abspath(seg_path(parts, i))}'\n")
    tmp = a.out + ".tmp.mp4"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", "-movflags", "+faststart"] + meta + [tmp])
    if r.returncode != 0:
        sys.exit("ffmpeg concat failed")
    os.replace(tmp, a.out)
    print(f"wrote {a.out}  ({len(idx)} segments)")


if __name__ == "__main__":
    main()
