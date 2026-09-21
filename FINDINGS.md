# Findings

Measured on MSR-VTT, 1,000 clips, `jev-1.13.0`. Everything here is reproducible from
this repo — `eval.py` and `compare.py` need no downloads.

## 1. Jev cannot see video

[`docs.typesafe.ai/concepts/state.md`](https://docs.typesafe.ai/concepts/state.md) states
that `state` must be a string, JSON object, or array of text values, and that images,
audio and video are "not supported (yet)". There is no way to hand Jev a frame.

So Jev is not the perception layer here. A vision model (or, below, a human) turns clips
into text; an embedding retrieves cheap candidates; **Jev is the judge** that decides which
of those actually show what was asked. Same shape as a reranker over documents.

## 2. Ranking: +9 points at R@1

MSR-VTT 1k-A, 100 queries, top-50 shortlist:

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| embeddings | 49.0 | 76.0 | 81.0 |
| **+ Jev** | **58.0** | **81.0** | **85.0** |

`recall@50 = 91%` is the ceiling — Jev can only reorder what retrieval surfaced, so it
recovered 9 of the 42 points available at R@1. Run-to-run variance is a few points at
n=100 (two runs gave R@1 57 and 58); treat small deltas as noise.

This was the surprise. MSR-VTT queries are literal descriptions, the case embeddings
already handle, so the prediction was that Jev would barely move these numbers.

## 3. The bigger difference is refusal, not ranking

Across 8 queries × 50 candidates:

| | returned | kept | cut |
|---|---|---|---|
| embeddings | 400 | 400 | 0 |
| + Jev | 400 | 48 | 352 (88%) |

**Embeddings always return k.** There is no similarity at which they abstain, because an
embedding score has no absolute meaning — only a relative one. Jev returned nothing on 2
of 8 queries, and both are correct: the corpus has no night-motorcycle clip and no
plane-above-clouds clip.

On a 20-query compositional set: 17 confident matches, 3 empty. All three refusals were
verified by hand against the corpus:

| query | top Jev score | what the corpus actually holds |
|---|---|---|
| a dog running on a beach | 0.11 | one dog+beach clip: a couple *walking* a dog |
| someone riding a motorcycle at night | 0.31 | zero night motorcycle clips |
| a plane flying above the clouds | 0.22 | six plane/sky clips: helicopters, aerial city, mechanical fault |

### A metric that lies

`eval.py --split msrvtt` also reports a cutoff sweep, and it looks bad: ~6.7 "false
positives" per query at floor 0.55. **That number is confounded by single-label ground
truth.** MSR-VTT marks exactly one of 1,000 clips correct per query, so the many genuinely
relevant near-duplicates ("a man is talking") are all counted as errors. The compositional
set, where the corpus can be checked directly, is the more honest read.

## 4. Where it wins: relationships

Embeddings treat "laptop **inside** a car" as roughly `{laptop, car}`. Four handwritten
captions, that exact query:

| caption | embedding | Jev |
|---|---|---|
| woman typing on a laptop in the back seat of a moving car | 0.737 | **0.98** |
| a laptop open on an office desk | 0.597 | 0.02 |
| a man driving a car on a highway | 0.529 | 0.03 |
| two people playing guitar on a stage | 0.235 | 0.01 |

The embedding orders it right but with a mushy margin (0.74 vs 0.60). Jev's spread is
0.98 vs 0.02 — a usable threshold, not just an order.

Real examples from the corpus:

- **"a man playing guitar while sitting in a vehicle"** — Jev returns a man playing guitar
  in a van and one on a truck tailgate. Text search returns two plain guitar close-ups.
  Zero overlap between the two lists.
- **"a person cooking food outdoors"** — embeddings return 4/5 *indoor* cooking (stirring a
  pot, frying, mixing batter). Jev returns 5/5 outdoors, promoting one clip from rank #43.
- **"a car driving through snow"** — embeddings return five cars on dry roads. Jev finds
  the corpus's single snow clip at rank 1, from embedding position #7.

## 5. Speed, and a mistake worth repeating

Per search, 1,000-clip corpus, k=50:

| | embed query | rank corpus | Jev | total |
|---|---|---|---|---|
| embeddings | 31.2 ms | 0.18 ms | — | **31.4 ms** |
| + Jev | 31.2 ms | 0.18 ms | 1465 ms | **1497 ms** |

Jev costs **45×** the embedding path, ~29 ms amortised per candidate.

The split matters more than the total. "Embedding search" is two unrelated costs: one
~31 ms call to Ollama for the query vector (fixed), and a 0.18 ms dot product over the
corpus (scales). **The corpus would need ~8 million clips before ranking costs what one
Jev search costs.** Jev scales with `k`, not corpus size.

### Concurrency was a guess, and it cost 2.6×

True single-call Jev latency is ~960 ms, so wall time is dominated by wave count, not by
Jev. Measured at k=50:

| in flight | 6 | 16 | 25 | 36 | 50 |
|---|---|---|---|---|---|
| wall time | 3785 ms | 1927 ms | 1422 ms | 1386 ms | 1105 ms |

The original 6 was a guess. Now 25 — past that returns diminish, and a 50-wide burst at
~1 s latency is 50 req/s against a 20/s budget that batch eval would blow. Median search
went 3747 ms → 1465 ms.

Streaming matters for the same reason: the first answers land in ~1 s, so the UI shows a
match the moment it is known rather than after the slowest of fifty (3 matches on screen
at 1.0 s, all 5 by 1.4 s).

## 6. The caveat that bounds all of it

The index is built from MSR-VTT's **human** captions, not generated ones.

That was deliberate: a weak captioner confounds every number, and you cannot tell a bad
judge from a bad description. It isolates "is Jev a good judge" — at the cost of being an
**upper bound**. A real VLM caption is shorter, vaguer, and sometimes wrong, so expect
lower absolute numbers end to end.

**The Jev-vs-embeddings delta is the transferable result. The absolute R@k values are not.**

The end-to-end video path (`ingest.py --videos`) is written and ffmpeg-tested but has
never run a real captioning pass; a local 8B VLM was too slow on the target machine
(~5 min/clip) and was abandoned in favour of isolating the judge.
