"""Fill missing 10-min slots with an optical-flow midpoint of the neighbours,
so the timelapse keeps an even pace. Usage: fill_gaps.py FRAMES_DIR MASK_PNG"""
import os, sys
from datetime import timedelta
import numpy as np
from PIL import Image
import starfield as sf
from interpolate import Interp, load

d, mask_png = sys.argv[1], sys.argv[2]
names = sorted(f for f in os.listdir(d) if sf.TIME_RE.search(f))
times = {sf.frame_time(n): n for n in names}
ext = os.path.splitext(names[0])[1]
mask = np.asarray(Image.open(mask_png).getchannel("A"), np.float32) / 255.0
ip = Interp(mask)
ts = sorted(times)
for ta, tb in zip(ts, ts[1:]):
    gap = int((tb - ta).total_seconds() // 600)
    if gap < 2:
        continue
    a, b = load(os.path.join(d, times[ta])), load(os.path.join(d, times[tb]))
    a, b = ip.extend(a), ip.extend(b)
    v = ip.flows(a, b)
    for k in range(1, gap):
        t = ta + timedelta(minutes=10 * k)
        out = ip.between(a, b, v, k / gap) * mask[..., None]
        p = os.path.join(d, t.strftime("%Y%m%d_%H%M") + ext)
        Image.fromarray(out.astype(np.uint8)).save(p, quality=95)
        with open(os.path.join(d, "SYNTHETIC.txt"), "a") as f:
            f.write(f"{os.path.basename(p)}  flow midpoint of {times[ta]} and {times[tb]}\n")
        print("filled", p)
