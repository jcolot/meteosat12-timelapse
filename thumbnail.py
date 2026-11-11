#!/usr/bin/env python3
"""YouTube thumbnail of a day's video: the noon UTC frame (fully sunlit disc,
the same moment every day so videos compare at a glance) with the date.

  thumbnail.py VIDEO OUT.jpg --date YYYY-MM-DD

16:9 videos give 1280x720 (disc enlarged, date on the left); portrait videos
720x1280 (date at the top). Needs ffmpeg and Pillow.
"""
import argparse, datetime as dt, io, subprocess
from PIL import Image, ImageDraw, ImageFont

FONTS = [("/System/Library/Fonts/HelveticaNeue.ttc", 1),                # macOS, bold
         ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0),  # Debian/Ubuntu
         ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 0)]


def font(size):
    for path, index in FONTS:
        try:
            return ImageFont.truetype(path, size, index=index)
        except OSError:
            pass
    return ImageFont.load_default(size)


def noon_frame(video):
    """The frame at the middle of the video: 12:00 UTC (videos run 00:00 to 24:00)."""
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                video], capture_output=True, text=True, check=True).stdout)
    png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{dur / 2:.3f}", "-i", video, "-frames:v", "1",
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
        im = im.resize((720, 1280), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        text(d, (360, 230), f"{day.day} {day:%b} {day.year}".upper(), 86, "ms")
    im.save(a.out, quality=90)


if __name__ == "__main__":
    main()
