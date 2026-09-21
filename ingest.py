"""Build index/{scenes.jsonl,vecs.npy}, from video or from existing captions.

Two sources:
  --videos DIR        scenes -> frames -> VLM caption -> embed  (the real pipeline)
  --from-captions ..  use MSR-VTT's human captions as the scene text

The second exists to isolate the question this repo is actually asking. A weak
captioner confounds every retrieval number: you cannot tell a bad judge from a bad
description. Human captions remove that variable -- they are an UPPER BOUND on what
any captioner would produce, so results on them are Jev's best case, not its
average case.

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
    ap.add_argument("--videos", help="directory of video files")
    ap.add_argument("--from-captions", metavar="ANNOTATIONS",
                    help="MSRVTT_data.json -- index human captions instead of captioning video")
    ap.add_argument("--max-captions", type=int, default=0, metavar="N",
                    help="keep only the first N captions per clip (0 = all). Merging all "
                         "20 makes a long, self-contradictory block; a real captioner "
                         "emits 2-3 sentences, and some judges are sensitive to this.")
    ap.add_argument("--holdout", metavar="GT_JSON",
                    help="msrvtt_test_1k.json: restrict to these videos AND exclude each "
                         "video's eval caption from its indexed text (965/1000 eval queries "
                         "appear verbatim in the annotation pool -- indexing them is leakage)")
    ap.add_argument("--out", default="index")
    ap.add_argument("--frames", type=int, default=4, help="frames sampled per scene")
    ap.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    ap.add_argument("--split-scenes", action="store_true",
                    help="detect cuts within each file; off = one scene per file "
                         "(right for short single-shot clips like MSR-VTT)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "scenes.jsonl"

    if args.from_captions:
        build_from_captions(args, jsonl)
        embed_index(out, jsonl)
        return

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

    embed_index(out, jsonl)


def build_from_captions(args, jsonl):
    data = json.loads(Path(args.from_captions).read_text())
    by_vid = {}
    for s in data["sentences"]:
        by_vid.setdefault(s["video_id"], []).append(s["caption"])
    meta = {v["video_id"]: v for v in data["videos"]}

    held = {}
    if args.holdout:
        for g in json.loads(Path(args.holdout).read_text()):
            held[g["video_id"]] = g["caption"]
        vids = [v for v in held if v in by_vid]
    else:
        vids = sorted(by_vid)
    if args.limit:
        vids = vids[:args.limit]

    with jsonl.open("w") as fh:
        for vid in vids:
            # Drop the eval caption, then dedupe: several annotators often write the
            # same sentence, and a near-duplicate of the query would leak just as much.
            caps = [c for c in by_vid[vid] if c.strip().lower() != held.get(vid, "").strip().lower()]
            seen, uniq = set(), []
            for c in caps:
                k = c.strip().lower()
                if k not in seen:
                    seen.add(k); uniq.append(c.strip())
            if args.max_captions:
                uniq = uniq[:args.max_captions]
            m = meta.get(vid, {})
            fh.write(json.dumps({
                "video": f"{vid}.mp4", "path": "", "start": 0.0,
                "end": round(m.get("end time", 0) - m.get("start time", 0), 2),
                "caption": ". ".join(uniq) + ".",
            }) + "\n")
    print(f"built {len(vids)} scenes from human captions"
          + (f" (held out {len(held)} eval captions)" if held else ""))


def embed_index(out, jsonl):
    """One pass at the end: cheap, and it keeps vecs.npy consistent with the jsonl
    even when ingest was resumed across several runs."""
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    vecs = np.vstack([core.embed([r["caption"] for r in rows[i:i + 64]])
                      for i in range(0, len(rows), 64)])
    np.save(out / "vecs.npy", vecs)
    print(f"indexed {len(rows)} scenes -> {out}")


if __name__ == "__main__":
    main()
