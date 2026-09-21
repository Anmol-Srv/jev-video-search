"""Shared pieces: Ollama captioning/embedding, and the Jev rerank.

Jev is text-only (docs.typesafe.ai/concepts/state.md: state is string/JSON, images
and video "not supported (yet)"), so the vision model owns perception and Jev only
judges the captions it produces.
"""
import base64, json, os, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

OLLAMA = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
VLM = os.environ.get("JVS_VLM", "minicpm-v4.5")
EMBED_MODEL = os.environ.get("JVS_EMBED", "mxbai-embed-large")
JEV_MODEL = "jev-1.13.0"          # pinned: thresholds below are calibrated to it
FLOOR = float(os.environ.get("JVS_FLOOR", "0.55"))
CONCURRENCY = 6                   # Jev allows 1200 req/min; 6 in flight is plenty

# The caption is the ceiling on everything downstream, so ask for the things
# compositional queries are made of: who, what object, where, and how they relate.
# A caption of "a man in a car" can never answer "laptop being used inside a car".
CAPTION_PROMPT = (
    "Describe this video scene in 2-3 sentences for a search index. State the people, "
    "the objects they are using, the setting or location, the action taking place, and "
    "how the objects are positioned relative to the people and the setting. Be concrete "
    "and literal. Do not speculate about emotions, story, or anything not visible."
)


# --- Ollama ------------------------------------------------------------------

def caption(frame_paths: list[Path], model: str = VLM, timeout: int = 300) -> str:
    images = [base64.b64encode(p.read_bytes()).decode() for p in frame_paths]
    r = requests.post(f"{OLLAMA}/api/generate", timeout=timeout, json={
        "model": model, "prompt": CAPTION_PROMPT, "images": images,
        "stream": False, "options": {"temperature": 0},
    })
    r.raise_for_status()
    return r.json()["response"].strip()


def embed(texts: list[str], model: str = EMBED_MODEL) -> np.ndarray:
    """L2-normalised, so cosine similarity is a plain dot product."""
    r = requests.post(f"{OLLAMA}/api/embed", timeout=120,
                      json={"model": model, "input": texts})
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True).clip(min=1e-9)


# --- ffmpeg ------------------------------------------------------------------

def duration(video: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def scenes(video: Path, threshold: float = 0.35, min_len: float = 1.5) -> list[tuple[float, float]]:
    """Scene boundaries via ffmpeg's scene score. Returns [(start, end), ...]."""
    dur = duration(video)
    if dur <= 0:
        return []
    out = subprocess.run(
        ["ffmpeg", "-nostats", "-i", str(video), "-filter:v",
         f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"],
        capture_output=True, text=True).stderr
    cuts = sorted({float(line.split("pts_time:")[1].split()[0])
                   for line in out.splitlines() if "pts_time:" in line})
    bounds = [0.0] + [c for c in cuts if min_len < c < dur - min_len] + [dur]
    segs = [(a, b) for a, b in zip(bounds, bounds[1:]) if b - a >= min_len]
    return segs or [(0.0, dur)]


def frames(video: Path, start: float, end: float, n: int, outdir: Path) -> list[Path]:
    """n frames evenly spaced inside [start, end), skipping the exact edges."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        t = start + (end - start) * (i + 0.5) / n
        p = outdir / f"{t:.3f}.jpg"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{t:.3f}",
                        "-i", str(video), "-frames:v", "1", "-vf", "scale=448:-2",
                        "-y", str(p)], check=False, capture_output=True)
        if p.exists() and p.stat().st_size > 0:
            paths.append(p)
    return paths


# --- Jev ---------------------------------------------------------------------

def jev_key() -> str | None:
    """Reuse the key Sigil already has configured rather than asking for a new one."""
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    cfg = Path.home() / ".sigil" / "config.json"
    if cfg.exists():
        return json.loads(cfg.read_text()).get("jev", {}).get("apiKey")
    return None


# The relationship clause is the whole point. An embedding treats "laptop inside a
# car" as roughly {laptop, car} and happily returns a laptop on a desk next to a
# parked car. Jev reads the caption and can tell the two apart -- but only if the
# criteria say so explicitly.
DEPICTS = {
    "instructions": (
        "`query` is what a viewer is searching for in a video. `scene.caption` describes "
        "one scene from a video. Does that scene show what the query describes, including "
        "the relationship, setting, or activity the query specifies between the things in it?"
    ),
    "criteria": {
        "true": ("The scene shows the queried subjects and objects in the queried "
                 "relationship, setting, or activity."),
        "false": ("The scene contains some of the queried things but not in the queried "
                  "relationship or setting, or is merely on a similar topic."),
    },
}


def jev_scores(query: str, captions: list[str], api_key: str | None = None,
               model: str = JEV_MODEL) -> list[float | None]:
    """One request per (query, caption) pair. None where Jev could not answer.

    Per-candidate rather than one packed request: irrelevant candidates act as
    distractors for jev-1.13, and "does candidate 3..." index indirection reads
    less reliably than a question about a named field.
    """
    from typesafe_sdk import Noul, TypeSafeClient

    key = api_key or jev_key()
    if not key:
        return [None] * len(captions)

    with TypeSafeClient(api_key=key, model=model, timeout=20.0) as client:
        def one(cap: str):
            try:
                resp = client.system_one(
                    state={"query": query[:1000], "scene": {"caption": cap[:1200]}},
                    questions={"depicts_query": Noul(**DEPICTS)},
                )
                return float(resp.nouls["depicts_query"].noul)
            except Exception:
                return None   # a candidate Jev can't score keeps its embedding rank

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            return list(pool.map(one, captions))


# --- index -------------------------------------------------------------------

def load_index(d: Path):
    rows = [json.loads(l) for l in (d / "scenes.jsonl").read_text().splitlines() if l.strip()]
    return rows, np.load(d / "vecs.npy")


def search(query: str, rows, vecs, top_k: int = 50, use_jev: bool = True):
    """Embedding shortlist, then Jev reorders it. Falls back to embedding order
    on any Jev failure -- the index stays authoritative, Jev is an improvement."""
    qv = embed([query])[0]
    order = np.argsort(-(vecs @ qv))[:top_k]
    shortlist = [dict(rows[i], sim=float(vecs[i] @ qv)) for i in order]
    if not use_jev:
        return shortlist, {"applied": False, "reason": "disabled"}

    scores = jev_scores(query, [s["caption"] for s in shortlist])
    if all(s is None for s in scores):
        return shortlist, {"applied": False, "reason": "unavailable"}

    for s, sc in zip(shortlist, scores):
        s["jev"] = sc
    # Unscored candidates sort by their embedding rank, below everything Jev saw.
    ranked = sorted(enumerate(shortlist),
                    key=lambda p: (-(p[1].get("jev") if p[1].get("jev") is not None else -1), p[0]))
    return [s for _, s in ranked], {
        "applied": True, "scored": sum(s is not None for s in scores),
        "above_floor": sum(1 for s in scores if s is not None and s >= FLOOR),
    }
