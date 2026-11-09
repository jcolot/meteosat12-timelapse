#!/usr/bin/env python3
"""Download the static inputs into data/ (run once; skips what is there):
  hip.tsv               Hipparcos main catalogue (ESA 1997, I/239), V < 9.5, via VizieR (CfA mirror)
  de421.bsp             JPL planetary ephemeris (Sun, Moon, planets)
  milkyway_2020_8k.exr  NASA SVS 4851 "Deep Star Maps 2020" Milky Way, celestial coordinates
"""
import os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
FILES = {
    "hip.tsv": "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=I/239/hip_main"
               "&-out=HIP,Vmag,RAICRS,DEICRS,pmRA,pmDE,Plx,B-V&-out.max=200000&Vmag=%3C9.5",
    "de421.bsp": "https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/de421.bsp",
    "milkyway_2020_8k.exr": "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/milkyway_2020_8k.exr",
}


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            print("have", name)
            continue
        print("get ", name, "<-", url, flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "meteosat12-timelapse/1.0"})
        with urllib.request.urlopen(req, timeout=600) as r, open(path + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(path + ".part", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
