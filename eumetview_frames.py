#!/usr/bin/env python3
"""
Download clean full-disc frames (no borders, no coastlines) from EUMETView's
WMS server, ready to assemble into a timelapse.

Usage examples:
  python3 eumetview_frames.py --start 2026-11-06T00:00 --end 2026-11-07T00:00
  python3 eumetview_frames.py --start 2026-11-06T06:00 --end 2026-11-06T18:00 --step 10 --size 2048

Then build the video (12 fps, H.264):
  ffmpeg -framerate 12 -pattern_type glob -i 'frames/*.png' \
         -vf "format=yuv420p" -c:v libx264 -crf 18 earth.mp4

Only the Python standard library is needed.
"""
import argparse
import datetime as dt
import os
import sys
import time
import urllib.parse
import urllib.request

WMS = "https://view.eumetsat.int/geoserver/wms"
# The EUMETView viewer's own geostationary (0 deg) projection.
CRS = "EPSG:900003"
# Half-width of the full disc in metres. If the disc edge looks clipped,
# raise this a little (e.g. 5600000); if there's too much margin, lower it.
HALF = 5570000

LAYERS = {
    "geocolour": "mtg_fd:rgb_geocolour",   # true colour by day, city lights by night
    "truecolour": "mtg_fd:rgb_truecolour", # daylight only (name may differ, check the viewer)
}


def frame_url(layer, when, size, fmt="png"):
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "FORMAT": "image/jpeg" if fmt == "jpeg" else "image/png",
        "TRANSPARENT": "false" if fmt == "jpeg" else "true",
        "LAYERS": layer,
        "STYLES": "",
        "TIME": when.strftime("%Y-%m-%dT%H:%M:00.000Z"),
        "WIDTH": size,
        "HEIGHT": size,
        "CRS": CRS,
        "BBOX": f"{-HALF},{-HALF},{HALF},{HALF}",
    }
    return WMS + "?" + urllib.parse.urlencode(params)


def fetch(url, path, retries=3):
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "eumetview-frames/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                ctype = r.headers.get("Content-Type", "")
                data = r.read()
            if not ctype.startswith("image/"):
                # WMS errors come back as XML; show the first bit
                raise RuntimeError(data[:300].decode(errors="replace"))
            with open(path, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as e:
            if attempt == retries:
                raise
            print(f"   retry {attempt} ({e})", file=sys.stderr)
            time.sleep(3 * attempt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="UTC, e.g. 2026-11-06T00:00")
    p.add_argument("--end", required=True, help="UTC, inclusive")
    p.add_argument("--step", type=int, default=10, help="minutes (10 = every available image)")
    p.add_argument("--size", type=int, default=2048, help="pixels (square)")
    p.add_argument("--layer", default="geocolour", help="geocolour, truecolour, or a raw layer name")
    p.add_argument("--out", default="frames")
    p.add_argument("--format", choices=["png", "jpeg"], default="png",
                   help="jpeg is ~10x smaller/faster but has no transparency (space is black)")
    a = p.parse_args()

    layer = LAYERS.get(a.layer, a.layer)
    t = dt.datetime.fromisoformat(a.start)
    end = dt.datetime.fromisoformat(a.end)
    os.makedirs(a.out, exist_ok=True)

    n, failed = 0, []
    while t <= end:
        path = os.path.join(a.out, t.strftime("%Y%m%d_%H%M") + (".jpg" if a.format == "jpeg" else ".png"))
        if os.path.exists(path):
            print(f"skip {path}")
        else:
            try:
                size = fetch(frame_url(layer, t, a.size, a.format), path)
                print(f"ok   {path}  ({size // 1024} KB)", flush=True)
            except Exception as e:
                # keep going; a re-run fetches only the missing frames
                print(f"FAIL {path}  ({str(e)[:120]})", flush=True)
                failed.append(path)
            time.sleep(0.5)  # be gentle with the server
        n += 1
        t += dt.timedelta(minutes=a.step)
    print(f"\nDone: {n - len(failed)}/{n} frames in ./{a.out}/")
    if failed:
        print(f"{len(failed)} failed; run the same command again to fetch them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
