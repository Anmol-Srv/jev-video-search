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

## Results

**MSR-VTT 1k-A, 100 queries, 1000 clips, top-50 shortlist** (`eval.py --split msrvtt`):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| embeddings | 49.0 | 76.0 | 81.0 |
| **+ Jev** | **58.0** | **81.0** | **85.0** |

`recall@50 = 91%` is the ceiling — Jev cannot rank what retrieval never surfaced, so
it recovered 9 of the 42 points available at R@1. Run-to-run variance is a few points
at n=100 (two runs gave R@1 57 and 58); treat small deltas as noise.

**Compositional queries** (`eval.py --queries queries.txt`), the case embeddings are
expected to fail: 17/20 returned a confident match, 3 returned nothing. Examples:

- *"a person cooking food outdoors"* — embeddings returned 4/5 **indoor** cooking
  (stirring a pot, frying, mixing batter). Jev returned 5/5 genuinely outdoors.
- *"a car driving through snow"* — embeddings returned five cars on dry roads. Jev
  found the one snow clip in the corpus at rank 1, from embedding position >5.
- *"playing an instrument while sitting down"* — Jev promoted a man sitting at a piano
  from `sim 0.537`, below the embedding top-5, and dropped a generic "man plays guitar".
- *"a crowd watching a performance on a stage"* — embeddings matched "on stage"; Jev
  promoted the clips that actually mention an audience watching.

### The cutoff: Jev can say "none of these match"

An embedding score has no absolute meaning, so embedding-only search always returns
its top-k no matter how wrong. The three queries above that returned nothing were all
**correct rejections**, verified against the corpus:

| query | top Jev score | corpus reality |
|---|---|---|
| a dog running on a beach | 0.11 | one dog+beach clip, a couple *walking* a dog |
| someone riding a motorcycle at night | 0.31 | zero night motorcycle clips |
| a plane flying above the clouds | 0.22 | six plane/sky clips: helicopters, aerial city, mechanical fault |

The `--split msrvtt` cutoff sweep looks much worse (~6.7 "false positives" per query at
floor 0.55) — but that metric is **confounded by single-label ground truth**. MSR-VTT
marks exactly one of 1000 clips correct per query, so the dozens of genuinely relevant
near-duplicates ("a man is talking") are all scored as errors. The compositional set,
where the corpus can be checked directly, is the more honest read of cutoff behaviour.

## Caveat that bounds all of the above

These use MSR-VTT's **human** captions, not generated ones (see `--from-captions`).
That isolates "is Jev a good judge" from "is the captioner good", at the cost of being
an **upper bound**: a real VLM caption is shorter, vaguer, and sometimes wrong. Expect
lower absolute numbers end-to-end. The Jev-vs-embeddings *delta* is the transferable
result; the absolute R@k values are not.

## Dashboard

```bash
python server.py       # http://127.0.0.1:8420
```

Stdlib only, no build step. One page, built as a demo of the idea: text search and Jev
racing on the same query, side by side.

- **Both columns fill in lockstep.** Text search ranks all 1,000 clips in ~30ms and is
  ready immediately. Each time Jev clears a clip, that match appears on the left and the
  clip wording alone would have ranked in the same slot appears on the right. You watch
  the two disagree in real time.
- **Results stream as they qualify.** `core.jev_stream` yields answers via `as_completed`
  over SSE, so a match is on screen the moment it is known: three by 1.0s, all five by
  1.4s, instead of a blank wait for the slowest of fifty.
- **The bars animate while the search runs.** Both are scaled to the projected finish
  time, extrapolated from how many clips have been read, so the Jev bar grows toward 100%
  while the text-search bar collapses to a sliver. Scaling against elapsed-so-far pins the
  running bar at 100% and shows nothing.
- **Every clip plays.** A caption is not evidence. Search "a dog running on a beach" and
  the left column stays empty while the right shows a couple *walking* a dog and an empty
  beach — the rejection is checkable in a glance.

Verified in-browser: WCAG AA on every text role (measured composited), keyboard reachable
with a visible focus ring, no horizontal overflow at 390/768/1280, HTTP range requests so
video seeks and Safari plays at all.

## Run

```bash
python3 -m venv .venv && .venv/bin/pip install typesafe-sdk numpy requests
# captions: --from-captions uses MSR-VTT's human captions (no VLM needed)

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
