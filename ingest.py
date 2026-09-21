"""Videos -> scenes -> captions -> embeddings -> index/{scenes.jsonl,vecs.npy}.

Resumable: already-captioned (video, start) pairs are skipped, so a long run can
be interrupted. Captions print as they land -- if they read vague, stop here,
because nothing downstream can recover from a bad caption.
"""
import argparse, json, shutil, tempfile, time
from pathlib import Path

import numpy as np

import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of video files")
    ap.add_argument("--out", default="index")
    ap.add_argument("--frames", type=int, default=4, help="frames sampled per scene")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    ap.add_argument("--split-scenes", action="store_true",
                    help="detect cuts within each file; off = one scene per file "
                         "(right for short single-shot clips like MSR-VTT)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "scenes.jsonl"

    done = set()
    if jsonl.exists():
        for line in jsonl.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["video"], round(r["start"], 2)))

    vids = sorted(p for p in Path(args.videos).iterdir()
                  if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"})
    if args.limit:
        vids = vids[:args.limit]

    t0 = time.time()
    with jsonl.open("a") as fh:
        for vi, v in enumerate(vids, 1):
            segs = core.scenes(v) if args.split_scenes else [(0.0, core.duration(v))]
            for start, end in segs:
                if (v.name, round(start, 2)) in done:
                    continue
                tmp = Path(tempfile.mkdtemp())
                try:
                    fr = core.frames(v, start, end, args.frames, tmp)
                    if not fr:
                        continue
                    cap = core.caption(fr)
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)
                fh.write(json.dumps({"video": v.name, "path": str(v), "start": round(start, 2),
                                     "end": round(end, 2), "caption": cap}) + "\n")
                fh.flush()
            rate = (time.time() - t0) / vi
            print(f"[{vi}/{len(vids)}] {v.name} ({rate:.1f}s/clip, "
                  f"~{rate * (len(vids) - vi) / 60:.0f}m left)\n  {cap[:160]}", flush=True)

    # Embed in one pass at the end: cheap, and it keeps vecs.npy consistent with
    # the jsonl even when ingest was resumed across several runs.
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    vecs = np.vstack([core.embed([r["caption"] for r in rows[i:i + 64]])
                      for i in range(0, len(rows), 64)])
    np.save(out / "vecs.npy", vecs)
    print(f"indexed {len(rows)} scenes -> {out}")


if __name__ == "__main__":
    main()
