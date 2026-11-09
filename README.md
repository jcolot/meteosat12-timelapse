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
# 15 s portrait 9:16 (Shorts / Reels), whole disc: every 4th frame, longer shutter
uv run interpolate.py frames_$D --mask MASK.png --canvas 2160x3840 --earth-px 2020 \
    --sky-strength 0.4 --night-fog fog_$D --night-ir ir_$D \
    --stride 4 --shutter 2.0 --body-shutter 0.2 -o earth_${D}_15s_vertical.mp4
```

`MASK.png` is any transparent-background PNG frame (`--format png`); its alpha marks the
Earth disc. Rendering runs in chunks (`<out>.parts/`): stop it any time and rerun the
same command to resume. Add `--device mps` on Apple silicon (about 3× faster).

### From raw satellite data (the nightly default)

The frames can instead be built from the raw FCI Level 1c images in the EUMETSAT
Data Store (about 1 GB per 10 min), with no dependence on EUMETView:
`raw/fci_frame.py` makes the three layers the renderer uses: true colour by day
(satpy), a GeoColour-like night side (infrared clouds over a static night
background and city lights, `assets/`), the fog / low-cloud RGB and 10.5 µm infrared,
on the same grid and grey scales as EUMETView's (fitted against them).

Credentials: `KEY` and `SECRET` from [api.eumetsat.int](https://api.eumetsat.int/api-key/)
in `.env` (or `EUMETSAT_KEY` / `EUMETSAT_SECRET` in the environment).

```bash
# one shard of a day's slots (here all of them), then merge, fill gaps, status
uv run --extra raw daily.py raw --date 2026-11-08 --shard 0 --shards 1
uv run daily.py gather --date 2026-11-08
uv run daily.py render --cut full --date 2026-11-08 --device mps
```

## Nightly on GitHub Actions

`.github/workflows/daily.yml` renders yesterday at 02:30 UTC and publishes both videos
as a release tagged with the date. The raw slots are split over 12 parallel jobs
(download, build frames, delete, about 12 slots each), then merged, rendered (60 s
and 15 s portrait in parallel) and released. It needs two repository secrets:

```bash
grep '^KEY=' .env | cut -d= -f2- | gh secret set EUMETSAT_KEY
grep '^SECRET=' .env | cut -d= -f2- | gh secret set EUMETSAT_SECRET
```

Run it by hand for any day from the Actions tab (source `raw` or `eumetview`). To redo
only the render and release, pass the run ID of an earlier run as `frames_from_run`
(frames are kept 3 days). Once the videos are in the release, the run's artifacts
(frames, videos) are deleted.

### YouTube

If the secrets below exist, the release job also uploads both videos to YouTube
(the 15 s portrait one as a Short); privacy comes from the repository variable
`YOUTUBE_PRIVACY` (default `public`).

1. In a Google Cloud project, enable the *YouTube Data API v3* and create an OAuth
   client of type *Desktop app*. Put its ID and secret in `.env` as
   `YOUTUBE_CLIENT_ID=` and `YOUTUBE_CLIENT_SECRET=`, and store them as secrets:

   ```bash
   grep '^YOUTUBE_CLIENT_ID=' .env | cut -d= -f2- | gh secret set YOUTUBE_CLIENT_ID
   grep '^YOUTUBE_CLIENT_SECRET=' .env | cut -d= -f2- | gh secret set YOUTUBE_CLIENT_SECRET
   ```

2. Sign in once with the channel's Google account; the refresh token goes straight into
   the `YOUTUBE_REFRESH_TOKEN` secret:

   ```bash
   uv run youtube.py auth
   ```

Videos uploaded through an API project that Google has not audited are locked to
*private*; request the audit (YouTube API Services compliance form) to publish them
publicly. While the OAuth consent screen is in *Testing* mode, refresh tokens expire
after 7 days: set it to *In production* (no verification needed for your own use).

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
- `assets/citylights.png`, `assets/night_background.jpg`: derived from EUMETView
  GeoColour night frames (EUMETSAT, CC BY 4.0).

Code: MIT, see [LICENSE](LICENSE).
