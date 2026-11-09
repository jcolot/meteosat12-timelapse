# meteosat12-timelapse

A full day of the Earth as seen by **Meteosat-12 (MTG-I1)**, the geostationary
satellite 35,786 km above 0°N 0°E, turned into a smooth 4K timelapse with the
**real sky behind it**: every star, the Moon and the planets where they actually
were, seen from the satellite's position, plus the Milky Way and zodiacal light.

Positions are exact. Brightness is enhanced: a camera exposed for the daylit Earth would
record no stars at all.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and `ffmpeg`.

```bash
uv sync                      # numpy, opencv, pillow, skyfield
uv run fetch_data.py         # Hipparcos stars, JPL ephemeris, NASA Milky Way map -> data/
```

Extras: `uv sync --extra gpu` (Apple Metal renderer), `uv sync --extra raw` (frames
from raw FCI data).

## A day, end to end

```bash
DAY=2026-11-08; D=20261108
# 1. GeoColour frames every 10 min from EUMETView (JPEG, 2048 px)
uv run eumetview_frames.py --start ${DAY}T00:00 --end 2026-11-09T00:00 --format jpeg --out frames_$D
# optional: infrared layers for better night clouds
uv run eumetview_frames.py --start ${DAY}T00:00 --end 2026-11-09T00:00 --format jpeg \
    --layer mtg_fd:rgb_fog --out fog_$D
uv run eumetview_frames.py --start ${DAY}T00:00 --end 2026-11-09T00:00 --format jpeg \
    --layer mtg_fd:ir105_hrfi --out ir_$D
# 2. fill slots the server does not have (flow-interpolated, listed in SYNTHETIC.txt)
uv run fill_gaps.py frames_$D MASK.png
# 3. render: 60 s at 24 fps, 3840x2160
uv run interpolate.py frames_$D --mask MASK.png --canvas 3840x2160 --sky-strength 0.4 \
    --night-fog fog_$D --night-ir ir_$D -o earth_$D.mp4
# 20 s cut: every 3rd frame with a longer shutter
uv run interpolate.py frames_$D --mask MASK.png --canvas 3840x2160 --sky-strength 0.4 \
    --stride 3 --shutter 1.5 --body-shutter 0.25 -o earth_${D}_20s.mp4
```

`MASK.png` is any transparent-background PNG frame (`--format png`); its alpha marks the
Earth disc. Rendering runs in chunks (`<out>.parts/`): stop it any time and rerun the
same command to resume. Add `--device mps` on Apple silicon (about 3× faster).

### From raw satellite data (experimental)

The EUMETSAT Data Store has the raw FCI Level 1c images (about 1 GB per 10 min).
With `KEY` and `SECRET` from [api.eumetsat.int](https://api.eumetsat.int/api-key/) in
`.env`:

```bash
uv run datastore.py EO:EUM:DAT:0662 2026-11-08T12:00:00Z 2026-11-08T12:09:00Z data/raw_fci
uv run --extra raw raw/fci_frame.py data/raw_fci/<product>.zip true_color frame.png
```

## How it works

- **Geometry**: GEOS projection, sweep `y`, WGS84, satellite at 35,785,831 m;
  `starfield.py validate` checks it against EUMETView's graticule (100 % within 1 px).
- **Sky**: Hipparcos stars and JPL DE421 bodies through Skyfield, observed from the
  satellite and expressed in Earth-fixed axes, so the sky turns behind a still Earth.
  Stars, Milky Way (NASA SVS) and zodiacal light (Leinert et al. 1998) are pre-rendered
  once per day in a right-ascension strip; each frame is a lookup with motion blur.
- **In-betweens**: optical flow (OpenCV DIS) between 10-min frames, cubic warps,
  forward/backward consistency where clouds form or dissolve.
- **Light**: GeoColour normalises the Sun away; it is put back physically: Lambert
  falloff, sunlight reddened through the atmosphere, single-scattering sky
  (Nishita: Rayleigh, Mie, ozone) for the twilight band and the glowing limb.
- **Night**: GeoColour's night clouds are infrared (temperature), not light. They are
  graded dim and blue, city lights amber; optionally rebuilt from the fog / low-cloud RGB
  and 10.5 µm layers, where low cloud, nearly invisible in GeoColour, shows up.

## Credits and data licences

On-screen and in the video metadata:

- Contains modified EUMETSAT Meteosat data 2026 ([CC BY 4.0](https://www.eumetsat.int/data-policy)).
- Stars: ESA Hipparcos catalogue (via VizieR, CDS).
- Moon and planets: NASA/JPL DE421.
- Milky Way: NASA Scientific Visualization Studio, *Deep Star Maps 2020*.
- Zodiacal light: Leinert et al. 1998, A&AS 127, 1.

Code: MIT, see [LICENSE](LICENSE).
