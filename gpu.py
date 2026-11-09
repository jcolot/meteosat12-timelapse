"""
GPU (PyTorch / Apple Metal "mps") version of the per-pixel work of a frame:
relighting, sky-glow tone curve and final compositing. It mirrors the CPU code
in relight.py, nightgrade.py, milkyway.Backdrop and interpolate.interval
operation for operation (same formulas, same OpenCV blur kernels and border
handling), so frames match the CPU renderer to within rounding.

What stays on the CPU: motion flow and in-between warping, the star catalogue,
the Moon/planets, sky-strip lookups, the zodiacal-light table (a tiny grid),
and video encoding.

Tensors are channels-first (C, H, W), float32.
"""
import numpy as np
import torch
import torch.nn.functional as F

import atmosphere as at
import milkyway as mwy
import nightgrade as ng


def device(name="mps"):
    if name == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("Metal GPU (mps) not available")
    return torch.device(name)


def _t(a, dev):
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(dev)


def _chw(a, dev):
    """(H, W, C) numpy -> (C, H, W) tensor."""
    return _t(a, dev).permute(2, 0, 1).contiguous()


def srgb_to_lin(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = x.clamp(0, 1)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def smoothstep(x, lo, hi):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _gauss_kernel(sigma, dev):
    """Same kernel as cv2.GaussianBlur(ksize=(0, 0), sigma) for float images."""
    k = int(round(sigma * 4 * 2 + 1)) | 1
    x = torch.arange(k, dtype=torch.float32, device=dev) - (k - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    return g / g.sum()


def gaussian_blur(img, kernel):
    """Separable blur of (H, W) or (C, H, W) with reflect-101 borders (cv2 default)."""
    squeeze = img.dim() == 2
    x = img[None, None] if squeeze else img[:, None]
    k = kernel.numel()
    p = k // 2
    x = F.pad(x, (p, p, 0, 0), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, 1, k))
    x = F.pad(x, (0, 0, p, p), mode="reflect")
    x = F.conv2d(x, kernel.view(1, 1, k, 1))
    return x[0, 0] if squeeze else x[:, 0]


def resize(img, size, mode):
    """(C, H, W) -> (C, h, w). mode: 'bilinear' (cv2 INTER_LINEAR), 'bicubic'
    (cv2 INTER_CUBIC), 'area' (cv2 INTER_AREA, downscaling)."""
    h, w = size
    x = img[None]
    if mode == "area":
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    else:
        x = F.interpolate(x, size=(h, w), mode=mode, align_corners=False)
    return x[0]


class FrameGPU:
    def __init__(self, relight, backdrop, geom, alpha, sky_t, credit, dev="mps"):
        d = self.dev = device(dev) if isinstance(dev, str) else dev
        R, N = relight, relight.ng
        self.S = relight.size
        self.geom = geom
        # relighting constants
        self.normals = _t(N.normals, d)                                    # (3, S, S)
        self.t_view = _chw(R.t_view, d)
        self.ref = _t(R.ref, d).view(3, 1, 1)
        self.curve_mu0, self.curve_n = float(at.MU_MIN), R.sun_ground.n
        self.T_ground = _t(R.sun_ground.T, d)                              # (n, 3)
        self.T_cloud = _t(R.sun_cloud.T, d)
        self.R, self.N = R, N
        self.base_rgb = _t(ng.BASE_RGB, d).view(3, 1, 1)
        self.tint = _t(N.tint * (3 / N.tint.sum()), d).view(3, 1, 1)
        self.lights_tint = _t(N.lights_tint * (3 / N.lights_tint.sum()), d).view(3, 1, 1)
        self.k_point = _gauss_kernel(6.0, d)
        self.k_lights = _gauss_kernel(N.lights_soft, d) if N.lights_soft else None
        # canvas layers
        self.alpha = _chw(alpha, d)                                        # (1, H, W)
        self.sky_t = _chw(sky_t, d) if sky_t is not None else None
        self.credit = _chw(credit, d) if credit is not None else None
        self.backdrop = backdrop
        if backdrop is not None and backdrop.zl is not None:
            self.zl_tint = _t(mwy.ZL_TINT, d).view(3, 1, 1)

    # --- relighting (2048 x 2048 Earth frame) --------------------------------------
    def _curve(self, T, mu):
        n = self.curve_n
        x = (mu.clamp(self.curve_mu0, 1.0) - self.curve_mu0) * ((n - 1) / (1.0 - self.curve_mu0))
        i = x.long().clamp(max=n - 2)
        f = (x - i)[None]
        Ti = T[i].permute(2, 0, 1)
        Tj = T[i + 1].permute(2, 0, 1)
        return Ti * (1 - f) + Tj * f

    def _night(self, img, L, cloud=None):
        N = self.N
        c = (L - ng.BASE_L).clamp(min=0) if cloud is None else cloud
        x = c * N.gain
        if N.cap:
            x = N.cap * (1 - torch.exp(-x / N.cap))
        base = img * N.base_scale
        target = self.base_rgb * N.base_scale + x[None] * self.tint
        w = smoothstep(c, 3.0, 30.0)[None]
        graded = base + w * (target - base)
        rb = img[0] - img[2]
        point = ((L - gaussian_blur(L, self.k_point)) / 15).clamp(0, 1)
        lights = torch.maximum(((rb - 20) / 20).clamp(0, 1), ((rb - 3) / 15).clamp(0, 1) * point)
        if N.lights_gain is None:
            return graded + lights[None] * (img - graded)
        lw = lights[None]
        bg = graded + lw * (self.base_rgb * N.base_scale - graded)
        emis = c * lights
        if self.k_lights is not None:
            emis = 0.6 * emis + 0.4 * gaussian_blur(emis, self.k_lights)
        return bg + (emis * N.lights_gain)[None] * self.lights_tint

    def relight(self, img, sun_dir, inscatter, cloud=None):
        """img: (S, S, 3) float32 sRGB 0-255 (numpy). inscatter: (atm, atm, 3)
        numpy. Returns (earth_lin, air) as (3, S, S) tensors."""
        R, S, d = self.R, self.S, self.dev
        img = _chw(img, d)
        sx, sy, sz = (float(v) for v in sun_dir)
        n = self.normals
        mu = n[0] * sx + n[1] * sy + n[2] * sz                     # scan_min = 0: same Sun for all rows
        t_sun = self._curve(self.T_ground, mu)
        t_cloud = self._curve(self.T_cloud, mu)
        zen = torch.rad2deg(torch.arccos(mu.clamp(-1, 1)))
        air = resize(_chw(inscatter, d), (S, S), "bilinear")

        lin = srgb_to_lin(img / 255.0)
        nb = smoothstep(zen, *R.blend)[None]
        day = lin * (mu.clamp(0, 1) ** R.cos_pow)[None] * t_sun * self.t_view / self.ref
        L = img.mean(dim=0)
        if cloud is not None:
            cloud = _t(cloud, d)
        night = srgb_to_lin(self._night(img, L, cloud) / 255.0)

        c = (L - ng.BASE_L).clamp(min=0) if cloud is None else cloud
        w = smoothstep(c, 3.0, 30.0)
        refl = srgb_to_lin((c / 255.0 * 1.2).clamp(0, 1)) * w
        m = mu + R.cloud_dip
        illum = (m / 0.12).clamp(0, 1) * m.clamp(min=0)
        night_src = smoothstep(zen, *R.twi_from)[None]
        twi = (refl * illum)[None] * t_cloud * self.t_view / self.ref * night_src * R.twilight

        air_n = air * (np.pi * R.k_air) / self.ref
        earth = day * (1 - nb) + night * nb + twi + air_n
        return earth, air_n

    # --- sky glow tone curve (canvas) ------------------------------------------------
    def backdrop_tone(self, mw_s10, zl_log):
        """mw_s10: (H, W, 3) numpy or None; zl_log: low-res log10 zodiacal
        brightness (h, w) numpy or None. Returns (3, H, W) linear light."""
        B, d = self.backdrop, self.dev
        H, W = self.geom["H"], self.geom["W"]
        tot = torch.zeros((3, H, W), device=d)
        if mw_s10 is not None:
            tot = tot + _chw(mw_s10, d) * B.k_mw
        if zl_log is not None:
            lg = resize(_t(zl_log, d)[None], (H, W), "bicubic")[0]
            zl = (10.0 ** lg)[None] * self.zl_tint * B.k_zl
            if B.zl_floor:
                zl_l = zl.mean(dim=0, keepdim=True)
                kept = mwy.ZL_KNEE * F.softplus((zl_l - B.zl_floor) / mwy.ZL_KNEE)
                if B.zl_peak:
                    kept = B.zl_peak * (1 - torch.exp(-kept / B.zl_peak))
                zl = zl * (kept / zl_l.clamp(min=1e-6))
            tot = tot + zl
        lum = tot.mean(dim=0, keepdim=True)
        lifted = mwy.SKY_KNEE * F.softplus((lum - B.floor) / mwy.SKY_KNEE)
        y = B.a * torch.log1p(lifted / B.k)
        return tot * (y / lum.clamp(min=1e-6))

    # --- final composite (canvas) ------------------------------------------------------
    def place(self, img):
        g = self.geom
        if g["E"] != img.shape[1]:
            img = resize(img, (g["E"], g["E"]), "area")
        out = torch.zeros((img.shape[0], g["H"], g["W"]), device=self.dev)
        out[:, g["y0"]:g["y0"] + g["E"], g["x0"]:g["x0"] + g["E"]] = img
        return out

    def compose(self, star_acc, back, earth_lin, air):
        """star_acc: (H, W, 3) numpy, accumulated star flux before saturation.
        Returns the final (H, W, 3) uint8 frame (numpy)."""
        acc = _chw(star_acc, self.dev)
        stars = (1 - torch.exp(-acc)).clamp(0, 1) ** (1 / 2.2)
        sky_lin = srgb_to_lin(stars)
        if back is not None:
            sky_lin = sky_lin + back
        if self.sky_t is not None:
            sky_lin = sky_lin * self.sky_t
        a = self.alpha
        lin = sky_lin * (1 - a) + self.place(earth_lin) * a + self.place(air) * (1 - a)
        f = lin_to_srgb(lin)
        if self.credit is not None:
            f = f * (1 - self.credit) + 0.92 * self.credit
        out = (f.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
        return out.permute(1, 2, 0).contiguous().cpu().numpy()
