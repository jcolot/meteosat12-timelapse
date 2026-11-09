#!/usr/bin/env python3
"""One day, end to end, for scheduled runs (GitHub Actions) or by hand.

  daily.py download [--date YYYY-MM-DD] [--budget-min 330]
      GeoColour + infrared night layers from EUMETView into work/<day>/,
      repeated passes until complete or out of time budget; then fills
      missing slots and writes work/<day>/status.json.
  daily.py render --cut full|short [--date ...] [--device cpu|mps]
      60 s (full) or 20 s (short) 4K video into out/.
  daily.py notes [--date ...]
      Release notes (markdown) on stdout.

--date defaults to yesterday (UTC).
"""
import argparse, datetime as dt, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
LAYERS = {                       # folder -> (EUMETView layer, required)
    "geocolour": ("mtg_fd:rgb_geocolour", True),
    "ir": ("mtg_fd:ir105_hrfi", False),
    "fog": ("mtg_fd:rgb_fog", False),
}
SLOTS = 145                      # 00:00 .. 24:00 every 10 min
MIN_GEOCOLOUR = 130              # below this the day is not rendered
MASK = os.path.join(HERE, "data", "mask.png")
CUTS = {
    "full": [],
    "short": ["--stride", "3", "--shutter", "1.5", "--body-shutter", "0.25"],
}


def day_of(a):
    return dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)


def work_dir(day):
    return os.path.join(HERE, "work", day.strftime("%Y%m%d"))


def count(d, real=False):
    """Frames in d; real=True leaves out slots filled by fill_gaps.py."""
    if not os.path.isdir(d):
        return 0
    n = len([f for f in os.listdir(d) if f[:8].isdigit()])
    syn = os.path.join(d, "SYNTHETIC.txt")
    if real and os.path.exists(syn):
        n -= len(open(syn).read().splitlines())
    return n


def fetch(layer, start, end, out, fmt="jpeg"):
    """One downloader pass (skips frames already there). Returns a Popen."""
    cmd = [PY, os.path.join(HERE, "eumetview_frames.py"), "--start", start, "--end", end,
           "--format", fmt, "--layer", layer, "--out", out]
    log = open(out + ".log", "a")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=HERE)


def ensure_mask(day):
    """Any transparent-background PNG frame: its alpha marks the Earth disc
    (identical every day, so it is fetched once and cached)."""
    if os.path.exists(MASK):
        return
    d = os.path.join(HERE, "work", "mask")
    os.makedirs(d, exist_ok=True)
    for hhmm in ("12:00", "12:10", "11:50", "12:20"):
        t = f"{day.isoformat()}T{hhmm}"
        fetch(LAYERS["geocolour"][0], t, t, d, fmt="png").wait()
        pngs = [f for f in os.listdir(d) if f.endswith(".png")]
        if pngs:
            os.replace(os.path.join(d, pngs[0]), MASK)
            return
    sys.exit("could not fetch a PNG frame for the Earth mask")


def download(a):
    day = day_of(a)
    wd = work_dir(day)
    os.makedirs(wd, exist_ok=True)
    start, end = f"{day.isoformat()}T00:00", f"{(day + dt.timedelta(days=1)).isoformat()}T00:00"
    deadline = time.time() + a.budget_min * 60
    ensure_mask(day)
    # all layers in parallel, repeated passes (the server times out and
    # returns 500s), until each is complete or the time budget is spent
    pending = {name: None for name in LAYERS}
    passes = {name: 0 for name in LAYERS}
    while pending and time.time() < deadline:
        for name in list(pending):
            p = pending[name]
            if p is not None and p.poll() is None:
                continue
            out = os.path.join(wd, name)
            if count(out) >= SLOTS or passes[name] >= a.passes:
                pending.pop(name)
                continue
            passes[name] += 1
            print(f"{name}: pass {passes[name]} ({count(out)}/{SLOTS})", flush=True)
            pending[name] = fetch(LAYERS[name][0], start, end, out)
        time.sleep(10)
    for p in pending.values():
        if p is not None and p.poll() is None:
            p.terminate()
    have = {name: count(os.path.join(wd, name), real=True) for name in LAYERS}
    print("downloaded:", have, flush=True)
    if have["geocolour"] < MIN_GEOCOLOUR:
        sys.exit(f"only {have['geocolour']} GeoColour frames: not rendering this day")
    # fill missing slots (flow midpoints, listed in SYNTHETIC.txt)
    for name in LAYERS:
        out = os.path.join(wd, name)
        if 2 <= count(out) < SLOTS:
            subprocess.run([PY, os.path.join(HERE, "fill_gaps.py"), out, MASK], check=True, cwd=HERE)
    names = lambda n: sorted(f for f in os.listdir(os.path.join(wd, n)) if f[:8].isdigit()) \
        if os.path.isdir(os.path.join(wd, n)) else []
    # infrared night layers only if they cover every GeoColour frame
    night = all(set(names("geocolour")) <= set(names(n)) for n in ("ir", "fog"))
    synth = {}
    for name in LAYERS:
        f = os.path.join(wd, name, "SYNTHETIC.txt")
        synth[name] = len(open(f).read().splitlines()) if os.path.exists(f) else 0
    status = dict(day=day.isoformat(), downloaded=have, synthetic=synth, night_layers=night)
    json.dump(status, open(os.path.join(wd, "status.json"), "w"), indent=1)
    print(json.dumps(status), flush=True)


def render(a):
    day = day_of(a)
    wd = work_dir(day)
    st = json.load(open(os.path.join(wd, "status.json")))
    os.makedirs(os.path.join(HERE, "out"), exist_ok=True)
    name = f"earth_{day.isoformat()}_4k" + ("" if a.cut == "full" else "_20s") + ".mp4"
    cmd = [PY, os.path.join(HERE, "interpolate.py"), os.path.join(wd, "geocolour"),
           "--mask", MASK, "--canvas", "3840x2160", "--sky-strength", "0.4",
           "--device", a.device, "-o", os.path.join(HERE, "out", name)] + CUTS[a.cut]
    if st["night_layers"]:
        cmd += ["--night-fog", os.path.join(wd, "fog"), "--night-ir", os.path.join(wd, "ir")]
    if a.workers:
        cmd += ["--workers", str(a.workers)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=HERE)


def notes(a):
    day = day_of(a)
    st = json.load(open(os.path.join(work_dir(day), "status.json")))
    syn = st["synthetic"]["geocolour"]
    print(f"""The Earth from **Meteosat-12 (MTG-I1)**, 36,000 km above 0°N 0°E, on {day:%-d %B %Y} (UTC),
every 10 minutes, with the real sky behind it: stars, Moon and planets where they actually were.

- `earth_{day}_4k.mp4`: 60 s, 3840×2160, 24 fps
- `earth_{day}_4k_20s.mp4`: 20 s cut

Frames: {st['downloaded']['geocolour']}/145 from EUMETView{f', {syn} missing slot(s) filled by motion interpolation' if syn else ''}.
Night clouds: {'GeoColour + infrared fog / low-cloud and 10.5 µm layers' if st['night_layers'] else 'GeoColour'}.

Contains modified EUMETSAT Meteosat data {day.year} (CC BY 4.0). Stars: ESA Hipparcos. Moon and planets: NASA/JPL DE421.
Milky Way: NASA SVS. Zodiacal light: Leinert et al. 1998.""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["download", "render", "notes"])
    p.add_argument("--date", help="YYYY-MM-DD (default: yesterday, UTC)")
    p.add_argument("--budget-min", type=float, default=330, help="download: time budget (min)")
    p.add_argument("--passes", type=int, default=8, help="download: max passes per layer")
    p.add_argument("--cut", choices=list(CUTS), default="full")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    {"download": download, "render": render, "notes": notes}[a.step](a)


if __name__ == "__main__":
    main()
