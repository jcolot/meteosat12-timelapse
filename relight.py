"""
Physically based relighting of a GeoColour frame around the terminator.

GeoColour's day image is normalised for sun angle (it stays bright right up to
the switch to the night image at ~80-85 deg zenith), which makes twilight a
hard band. Here, in linear light:

  day      = reflectance * cos(sun zenith)^k * T_sun * T_view / T_noon
             (Lambert falloff; sunlight reddened by the long, curved path
             through the atmosphere; Earth's shadow from the atmosphere model)
  night    = capped, blue-graded night image (nightgrade.NightGrade.night)
  twilight = night-side cloud tops still in sunlight (cloud_km up),
             lit by reddened grazing light
  air      = single-scattered sky light along the view ray (Nishita), incl.
             the glowing rim above the limb

  earth = day * (1 - n) + night * n + twilight + k_air * air

n blends across GeoColour's own day/night switch (true solar zenith 80-90 deg),
where the day term has already faded to almost nothing.
"""
import numpy as np

import atmosphere as at
import nightgrade as ng


def srgb_to_lin(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    out = (x + np.float32(0.055)) / np.float32(1.055)
    np.power(out, np.float32(2.4), out=out)
    low = x <= 0.04045
    out[low] = x[low] / np.float32(12.92)
    return out


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    out = np.power(x, np.float32(1 / 2.4))
    out *= np.float32(1.055)
    out -= np.float32(0.055)
    low = x <= 0.0031308
    out[low] = x[low] * np.float32(12.92)
    return out


class Relight:
    def __init__(self, size=2048, atm_size=1024, steps=32, k_air=1.0, cos_pow=1.0,
                 twilight=1.0, cloud_km=6.0, night=None, blend=(74.0, 86.0),
                 twi_from=(85.0, 89.0), scan_min=0.0):
        self.size = size
        self.atm = at.Atmosphere(size=atm_size, steps=steps, cloud_km=cloud_km)
        self.cloud_km = cloud_km
        self.t_view = at.upsample(self.atm.view_transmittance(), size)
        self.sun_ground = at.SunCurve(0.5)
        self.sun_cloud = at.SunCurve(cloud_km)
        self.ref = at.reference()
        self.k_air = k_air
        self.cos_pow = cos_pow
        self.twilight = twilight
        self.cloud_dip = np.sin(np.arccos(at.R_GROUND / (at.R_GROUND + cloud_km)))
        self.ng = night or ng.NightGrade()
        self.blend = blend
        self.twi_from = twi_from
        # FCI scans south -> north over the repeat cycle; row r of the frame
        # was imaged (scan_min * (1 - (r + .5) / size) - scan_min / 2) minutes
        # from the mid-scan time the caller passes.
        r = (np.arange(size) + 0.5) / size
        self.row_dt_s = (scan_min * (1 - r) - scan_min / 2) * 60.0

    def mu_rows(self, sun_dir):
        """cos(solar zenith) per pixel using each row's own scan time: in the
        Earth-fixed frame the Sun turns about the polar axis by -omega * dt."""
        ang = -2 * np.pi / 86400.0 * self.row_dt_s          # (size,)
        c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
        sx, sy, sz = (float(v) for v in sun_dir)
        rx = c * sx - s * sy
        ry = s * sx + c * sy
        n = self.ng.normals
        return (n[0] * rx.astype(np.float32) + n[1] * ry.astype(np.float32)
                + n[2] * np.float32(sz))

    def inscatter(self, sun_dir):
        """The expensive part (view-ray march), at atm_size. Cache it per
        source frame and blend between neighbours: it changes slowly."""
        return self.atm.compute(sun_dir)["inscatter"]

    def __call__(self, img, sun_dir, inscatter=None, cloud=None):
        """img: float32 (size, size, 3) sRGB 0-255. Returns (earth_lin, rim_lin):
        linear RGB of the Earth disc, and the additive air glow (whole frame;
        use it outside the disc only). Pass a cached/blended `inscatter`
        (atm_size) to skip the march; everything else is exact per frame."""
        S = self.size
        if inscatter is None:
            inscatter = self.inscatter(sun_dir)
        air = at.upsample(inscatter, S)
        t_view = self.t_view
        mu = self.mu_rows(sun_dir)
        t_sun = self.sun_ground(mu)
        t_cloud = self.sun_cloud(mu)
        zen = np.degrees(np.arccos(np.clip(mu, -1, 1)))

        lin = srgb_to_lin(img / 255.0)
        # Day and night are complements over one band, ending before
        # GeoColour's own switch (80-90 deg) so its cut never shows. The night
        # grade turns whatever the source shows there (day or IR clouds) into
        # the same dim blue, so the fall from day into night has no dip.
        n = ng.smoothstep(zen, *self.blend)[..., None]
        day = lin * (np.clip(mu, 0, 1) ** self.cos_pow)[..., None] * t_sun * t_view / self.ref
        night = srgb_to_lin(self.ng.night(img, cloud) / 255.0)

        # cloud tops past the terminator: sun above their (dipped) horizon
        w, c = self.ng.cloud_signal(img, cloud)
        refl = srgb_to_lin(np.clip(c / 255.0 * 1.2, 0, 1)) * w     # IR brightness as a cloud proxy
        illum = np.clip((mu + self.cloud_dip) / 0.12, 0, 1) * np.clip(mu + self.cloud_dip, 0, None)
        # only on pixels GeoColour already draws from its night (IR) image:
        # inside its switch band the "cloud" signal is daytime brightness
        night_src = ng.smoothstep(zen, *self.twi_from)[..., None]
        twi = (refl * illum)[..., None] * t_cloud * t_view / self.ref * night_src * self.twilight

        air_n = air * np.pi / self.ref * self.k_air
        earth = day * (1 - n) + night * n + twi + air_n
        return earth, air_n
