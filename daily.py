#!/usr/bin/env python3
"""One day, end to end, for scheduled runs (GitHub Actions) or by hand.

  daily.py raw --shard K --shards N [--date ...]
      Frames from raw FCI data (EUMETSAT Data Store, EUMETSAT_KEY/SECRET):
      every N-th 10-min slot starting at K, downloaded, turned into
      geocolour/fog/ir frames in work/<day>/ and deleted (needs the "raw" extra).
  daily.py gather [--date ...]
      After the shards are merged into work/<day>/: fills missing slots,
      writes work/<day>/status.json.
  daily.py download [--date YYYY-MM-DD] [--budget-min 330]
      Alternative source: GeoColour + infrared night layers from EUMETView
      (repeated passes until complete or out of time budget), then gather.
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


WORK = os.environ.get("DAILY_WORK", os.path.join(HERE, "work"))


def work_dir(day):
    return os.path.join(WORK, day.strftime("%Y%m%d"))


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


def ensure_mask(day=None):
    """RGBA PNG whose alpha marks the Earth disc, from the projection geometry
    (anti-aliased by 2x supersampling); matches EUMETView's PNG alpha."""
    if os.path.exists(MASK):
        return
    import cv2, numpy as np
    from PIL import Image
    import nightgrade as ng
    import starfield as sf
    n = ng.surface_normals(2 * sf.NATIVE_SIZE)
    a = cv2.resize((~np.isnan(n[0])).astype(np.float32), (sf.NATIVE_SIZE,) * 2,
                   interpolation=cv2.INTER_AREA)
    rgba = np.zeros((sf.NATIVE_SIZE, sf.NATIVE_SIZE, 4), np.uint8)
    rgba[..., 3] = np.round(a * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(MASK), exist_ok=True)
    Image.fromarray(rgba).save(MASK)


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
    gather_dir(wd, day, "EUMETView")


def raw(a):
    sys.path.insert(0, os.path.join(HERE, "raw"))
    import datastore
    import fci_frame
    day = day_of(a)
    wd = work_dir(day)
    tmp = os.path.join(WORK, "raw_tmp")
    os.makedirs(tmp, exist_ok=True)
    t0 = dt.datetime.combine(day, dt.time())
    slots = [t0 + dt.timedelta(minutes=10 * i) for i in range(SLOTS) if i % a.shards == a.shard]
    tok, tok_t = datastore.token(), time.time()
    for k, t in enumerate(slots):
        name = t.strftime("%Y%m%d_%H%M") + ".jpg"
        if all(os.path.exists(os.path.join(wd, sub, name)) for sub in LAYERS):
            continue
        for attempt in range(3):
            try:
                if time.time() - tok_t > 1800:               # tokens last ~1 h
                    tok, tok_t = datastore.token(), time.time()
                ids = [p for p in datastore.search(datastore.FDHSI, t.strftime("%Y-%m-%dT%H:%M:00Z"),
                                                   (t + dt.timedelta(minutes=9, seconds=59)).strftime("%Y-%m-%dT%H:%M:%SZ"))
                       if fci_frame.slot_time(p + ".zip") == t]
                if not ids:
                    print(f"{name}: no product", flush=True)
                    break
                t1 = time.time()
                z = datastore.download(datastore.FDHSI, ids[0], tmp, tok)
                t2 = time.time()
                fci_frame.build(z, wd)
                print(f"{name}: ok ({k + 1}/{len(slots)}, download {t2 - t1:.0f} s, build {time.time() - t2:.0f} s)",
                      flush=True)
                break
            except Exception as e:                            # noqa: BLE001  (one slot must not stop the shard)
                print(f"{name}: attempt {attempt + 1} failed: {e}", flush=True)
                for f in os.listdir(tmp):
                    p = os.path.join(tmp, f)
                    (os.remove if os.path.isfile(p) else __import__("shutil").rmtree)(p)
                time.sleep(20 * (attempt + 1))


def gather(a):
    day = day_of(a)
    gather_dir(work_dir(day), day, "raw FCI Level 1c (EUMETSAT Data Store)")


def gather_dir(wd, day, source):
    ensure_mask()
    have = {name: count(os.path.join(wd, name), real=True) for name in LAYERS}
    print("frames:", have, flush=True)
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
    status = dict(day=day.isoformat(), source=source, downloaded=have, synthetic=synth, night_layers=night)
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

Frames: {st['downloaded']['geocolour']}/145 from {st.get('source', 'EUMETView')}{f', {syn} missing slot(s) filled by motion interpolation' if syn else ''}.
Night clouds: {'GeoColour + infrared fog / low-cloud and 10.5 µm layers' if st['night_layers'] else 'GeoColour'}.

Contains modified EUMETSAT Meteosat data {day.year} (CC BY 4.0). Stars: ESA Hipparcos. Moon and planets: NASA/JPL DE421.
Milky Way: NASA SVS. Zodiacal light: Leinert et al. 1998.""")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["raw", "gather", "download", "render", "notes"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--date", help="YYYY-MM-DD (default: yesterday, UTC)")
    p.add_argument("--budget-min", type=float, default=330, help="download: time budget (min)")
    p.add_argument("--passes", type=int, default=8, help="download: max passes per layer")
    p.add_argument("--cut", choices=list(CUTS), default="full")
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    {"raw": raw, "gather": gather, "download": download, "render": render, "notes": notes}[a.step](a)


if __name__ == "__main__":
    main()
