"""Record the dashboard driving itself through several queries, for a demo clip.

Frames come out of Chrome via CDP Page.captureScreenshot (the browse daemon is
headless, so there is no window to screen-record). Each frame is stamped with the
time it was taken and ffmpeg is fed real per-frame durations, so the playback speed
is the actual speed -- the whole point is how fast matches appear.
"""
import base64, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

BROWSE = Path.home() / ".claude/skills/gstack/browse/dist/browse"
URL = "http://127.0.0.1:8420"
FRAMES = Path("/tmp/jvs-frames")
DATA = re.compile(rb'"data":\s*"([^"]+)"')

QUERIES = [
    ("a car driving through snow",        5.0),
    ("a person cooking food outdoors",    5.0),
    ("a dog running on a beach",          5.5),
]


def browse(*args, quiet=True):
    return subprocess.run([str(BROWSE), *args], capture_output=True, text=True).stdout


def shot(n):
    r = subprocess.run([str(BROWSE), "cdp", "Page.captureScreenshot",
                        '{"format":"jpeg","quality":80}'], capture_output=True)
    m = DATA.search(r.stdout)
    if not m:
        return None
    (FRAMES / f"{n:05d}.jpg").write_bytes(base64.b64decode(m.group(1)))
    return time.time()


def capture(seconds, stamps, start_n):
    """Capture as fast as the round trip allows for `seconds`, recording real times."""
    n, t_end = start_n, time.time() + seconds
    while time.time() < t_end:
        t = shot(n)
        if t:
            stamps.append((n, t)); n += 1
    return n


def main():
    if FRAMES.exists():
        shutil.rmtree(FRAMES)
    FRAMES.mkdir(parents=True)

    browse("viewport", "1280x720")
    browse("goto", URL)
    time.sleep(2.0)

    stamps, n = [], 0
    n = capture(2.0, stamps, n)                       # intro: dataset size on screen

    for qi, (q, hold) in enumerate(QUERIES):
        # Drive the form, not the chips: the chips live in the intro, which the first
        # search replaces -- clicking them silently did nothing from query 2 onward.
        browse("js", f'(()=>{{const i=document.getElementById("q");'
                     f'i.value={json.dumps(q)};'
                     f'document.getElementById("f").requestSubmit();return 1}})()')
        n = capture(hold, stamps, n)                  # stream the matches in
        browse("js", "window.scrollTo({top:300,behavior:'smooth'}); 1")
        n = capture(2.0, stamps, n)                   # pan down the two columns
        if qi == 1:                                   # prove the results are real video
            browse("js", "(()=>{const t=document.querySelector('#jevCol .cthumb');"
                         "t&&t.click();return 1})()")
            n = capture(3.0, stamps, n)
        browse("js", "window.scrollTo({top:0,behavior:'smooth'}); 1")
        n = capture(1.0, stamps, n)

    print(f"{len(stamps)} frames")
    if len(stamps) < 2:
        sys.exit("no frames captured")

    # Real per-frame durations -> playback matches wall-clock.
    lines = []
    for i, (fn, t) in enumerate(stamps):
        dur = (stamps[i + 1][1] - t) if i + 1 < len(stamps) else 0.10
        lines.append(f"file '{FRAMES / f'{fn:05d}.jpg'}'\nduration {max(dur, 0.02):.4f}")
    lines.append(f"file '{FRAMES / f'{stamps[-1][0]:05d}.jpg'}'")
    concat = FRAMES / "list.txt"
    concat.write_text("\n".join(lines))

    span = stamps[-1][1] - stamps[0][1]
    print(f"{span:.1f}s of wall clock · {len(stamps)/span:.1f} fps captured")

    out = Path("demo.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                    # -r alone resamples the variable concat timing to constant 30fps;
                    # pairing it with -vsync vfr is contradictory and ffmpeg refuses.
                    "-i", str(concat), "-r", "30",
                    # JPEG frames are full-range; without the explicit range convert
                    # this lands as yuvj420p and shifts colour on some players.
                    "-vf", "scale=in_range=full:out_range=limited,format=yuv420p",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-profile:v", "high", "-level", "4.0",
                    "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-movflags", "+faststart", str(out)], check=True)
    print(f"{out} · {out.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
