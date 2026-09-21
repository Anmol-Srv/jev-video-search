# jev-video-search

Netflix-style semantic scene search — "laptop being used inside a car" → the clips
and timestamps where that actually happens.

## Where Jev fits

**Jev is text-only.** [`docs.typesafe.ai/concepts/state.md`](https://docs.typesafe.ai/concepts/state.md)
states that `state` must be a string, JSON object, or array of text values, and that
images/audio/video are "not supported (yet)". There is no way to hand Jev a frame.

So Jev is not the perception layer — it's the **judge**:

```
video ──ffmpeg scene-detect──> scenes ──frames──> MiniCPM-V 4.5 ──> caption
                                                                      │
                                              mxbai-embed-large ──> vector
                                                                      │
query ──embed──> cosine top-K ──> Jev Noul per candidate ──> floor ──> ranked timestamps
```

Same shape as `sigil/src/lib/jev.js`: cheap local shortlist, then one narrow Noul per
candidate, with the local order as the fallback on any Jev failure.

## The hypothesis

Embedding search on captions handles literal queries fine. It fails on **compositional**
ones — "laptop **inside** a car" is roughly `{laptop, car}` to an embedding, so a laptop
on a desk beside a parked car scores well. Jev reads the caption and can tell them apart.

Measured on four handwritten captions for that exact query:

| caption | embedding | Jev |
|---|---|---|
| woman typing on a laptop in the back seat of a moving car | 0.737 | **0.98** |
| a laptop open on an office desk | 0.597 | 0.02 |
| a man driving a car on a highway | 0.529 | 0.03 |
| two people playing guitar on a stage | 0.235 | 0.01 |

The embedding gets the order right but the margin is mushy (0.74 vs 0.60), and an
embedding score has no absolute meaning — so embedding-only search can never answer
"**none of these match**". A Jev probability can. That cutoff is arguably the bigger win.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install typesafe-sdk numpy requests
ollama pull minicpm-v4.5          # captioner; mxbai-embed-large for vectors

python ingest.py --videos data/clips --out index --limit 200
python search.py "laptop being used inside a car"
python eval.py --split msrvtt -n 100        # R@1/5/10 ± Jev, recall@K ceiling
python eval.py --queries queries.txt        # compositional, hand-judged
python test_core.py                         # ranking logic, no network
```

The Jev API key is read from Sigil's config (`~/.sigil/config.json`) or `TYPESAFE_API_KEY`.

## Reading the eval

`recall@K` is the ceiling: Jev can only reorder what retrieval surfaced. If it's low,
the problem is captions or embeddings, not the judge.

Expect Jev to move the compositional numbers and barely touch the MSR-VTT ones —
MSR-VTT queries are literal descriptions, which is exactly the case embeddings already
handle. A flat result there is the expected outcome, not a failure.

## Not built

Web player/thumbnails · ASR track for speech queries · a second Jev `Score` on subject
prominence (central vs incidental) · ingest-time Jev attribute panel for hard filters.
