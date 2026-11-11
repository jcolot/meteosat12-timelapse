#!/usr/bin/env python3
"""YouTube thumbnail of a day's video: the noon UTC frame (fully sunlit disc,
the same moment every day so videos compare at a glance) with the date.

  thumbnail.py VIDEO OUT.jpg --date YYYY-MM-DD

16:9 videos give 1280x720 (disc enlarged, date on the left); portrait videos
720x1280 as a polaroid (date handwritten on the white strip). Needs Pillow and ffmpeg (or the imageio-ffmpeg package).
"""
import argparse, datetime as dt, io, re, shutil, subprocess
from PIL import Image, ImageDraw, ImageFont

import os

# Space Grotesk and Caveat (SIL OFL, fonts/OFL-*.txt), shipped in the repo so every machine renders the same
FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONT = os.path.join(FONTS, "SpaceGrotesk.ttf")
HAND = os.path.join(FONTS, "Caveat.ttf")          # handwritten caption of the Shorts' polaroid


def font(size, weight=700, path=FONT):
    f = ImageFont.truetype(path, size)
    f.set_variation_by_axes([weight])                                    # single axis: wght
    return f


def ffmpeg():
    """ffmpeg on the PATH, else the static one of the imageio-ffmpeg package."""
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def noon_frame(video):
    """The frame at the middle of the video: 12:00 UTC (videos run 00:00 to 24:00)."""
    ff = ffmpeg()
    info = subprocess.run([ff, "-i", video], capture_output=True, text=True).stderr
    h, m, sec = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
    dur = int(h) * 3600 + int(m) * 60 + float(sec)
    png = subprocess.run([ff, "-v", "error", "-ss", f"{dur / 2:.3f}", "-i", video, "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True, check=True).stdout
    return Image.open(io.BytesIO(png)).convert("RGB")


def text(d, xy, s, size, anchor):
    f = font(size)
    d.text((xy[0] + size // 25, xy[1] + size // 25), s, font=f, fill=(0, 0, 0), anchor=anchor)   # soft shadow
    d.text(xy, s, font=f, fill=(255, 255, 255), anchor=anchor)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--date", required=True)
    a = p.parse_args()
    day = dt.date.fromisoformat(a.date)
    im = noon_frame(a.video)
    W, H = im.size
    if W > H:
        # crop to 5/6 so the disc nearly fills the height (the credit line falls
        # outside), window at the left edge so the disc sits right of the date
        cw, ch = round(W * 5 / 6), round(H * 5 / 6)
        im = im.crop((0, (H - ch) // 2, cw, (H + ch) // 2)).resize((1280, 720), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        text(d, (36, 340), f"{day.day} {day:%b}".upper(), 80, "ls")
        text(d, (36, 430), f"{day.year}", 80, "ls")
    else:
        # a polaroid: the thumbnail's edge is the white frame, the date handwritten on the
        # strip. The Shorts shelf crops 9:16 thumbnails to about 2:3 (~100 px off the top and
        # bottom at 1280): photo and date stay inside that band
        m, safe, strip = 40, 100, 200
        top, bottom = safe + m, 1280 - safe
        pw, ph = 720 - 2 * m, bottom - strip - top
        ch = round(W * ph / pw)
        photo = im.crop((0, (H - ch) // 2, W, (H + ch) // 2)).resize((pw, ph), Image.LANCZOS)
        im = Image.new("RGB", (720, 1280), (246, 244, 238))
        im.paste(photo, (m, top))
        ImageDraw.Draw(im).text((360, bottom - strip // 2), f"{day.day} {day:%b} {day.year}",
                                font=font(96, 700, HAND), fill=(40, 40, 46), anchor="mm")
    im.save(a.out, quality=90)


if __name__ == "__main__":
    main()
