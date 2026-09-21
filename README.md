# jev-video-search

Search video by what happens on screen — *"a man playing guitar while sitting in a
vehicle"* — and get back the clips that actually show it.

An embedding retrieves 50 candidates in ~30 ms. [Jev](https://typesafe.ai) reads each one
and decides whether it really matches. The dashboard runs both side by side so you can
watch them disagree.

https://github.com/USER/REPO/assets/demo.mp4

## The point

**Embeddings always return k.** There is no similarity score at which they abstain,
because an embedding score has no absolute meaning — only a relative one. Ask for
something the library does not contain and you still get 50 confident-looking results.

Jev can say *nothing here matches*, and on this corpus it is right when it does.

It also handles the thing embeddings structurally drop: **relationships**. "laptop
**inside** a car" is roughly `{laptop, car}` to an embedding, so a laptop on a desk beside
a parked car scores well. Measured on MSR-VTT, Jev's rerank is worth **+9 points at R@1**
against a recall@50 ceiling of 91%.

Full numbers, method, and the caveats that bound them: **[FINDINGS.md](FINDINGS.md)**.

## Quickstart

```bash
python3 -m venv .venv && .venv/bin/pip install typesafe-sdk numpy requests
export TYPESAFE_API_KEY=...        # or let it read ~/.sigil/config.json
ollama pull mxbai-embed-large      # query embeddings, runs locally
```

The index is committed, so these work immediately — no download:

```bash
python compare.py                  # speed / count / kept-vs-kept, 8 queries
python eval.py --split msrvtt -n 100 --top-k 50    # R@1/5/10 ± Jev, recall@50
python eval.py --sweep index/eval_scores.json      # replay thresholds, no API calls
python test_core.py                # ranking logic, no network
```

The dashboard plays the clips, so it needs them:

```bash
./fetch_dataset.sh                 # MSR-VTT, 2.1 GB
python server.py                   # http://127.0.0.1:8420
```

## What's here

| file | |
|---|---|
| `core.py` | Ollama embed/caption, ffmpeg scene split, the Jev question + streaming scorer |
| `server.py` | Dashboard, `/api/stream` (SSE), range-serving for the clips. Stdlib only |
| `dashboard.html` | Single page, no build step |
| `ingest.py` | Build the index — from video, or from MSR-VTT's human captions |
| `eval.py` | Retrieval benchmark, cutoff sweep, compositional set |
| `compare.py` | Terminal side-by-side: speed, count, kept-vs-kept |
| `record.py` | Drives the dashboard and records `demo.mp4` via CDP |
| `queries.txt` | 20 compositional queries |

## The Jev question

One request per (query, caption) pair. The relationship clause in the criteria is doing
the real work — it is the exact thing an embedding drops.

```python
state = {"query": ..., "scene": {"caption": ...}}
Noul(
  instructions="`query` is what a viewer is searching for in a video. `scene.caption` "
    "describes one scene. Does that scene show what the query describes, including the "
    "relationship, setting, or activity the query specifies between the things in it?",
  criteria={
    "true":  "The scene shows the queried subjects and objects in the queried "
             "relationship, setting, or activity.",
    "false": "The scene contains some of the queried things but not in the queried "
             "relationship or setting, or is merely on a similar topic.",
  })
```

Per-candidate rather than one packed request: irrelevant candidates act as distractors for
`jev-1.13`, and "does candidate 3…" index indirection reads less reliably than a question
about a named field. The local ranking stays authoritative — on any Jev failure the
embedding order is returned unchanged.

## Dataset

[MSR-VTT](https://huggingface.co/datasets/friedrichor/MSR-VTT) — 10,000 clips on disk, the
standard 1k-A test split indexed. It ships ground-truth query→clip pairs, which is what
makes the comparison measurable instead of anecdotal.

The index is built from MSR-VTT's **human** captions, not generated ones. That isolates
"is Jev a good judge" from "is the captioner good" — at the cost of being an upper bound.
See [FINDINGS.md §6](FINDINGS.md).

## Not built

Thumbnails · ASR track for speech queries · a second Jev `Score` on subject prominence
(central vs incidental) · ingest-time attribute panel for hard filters.
