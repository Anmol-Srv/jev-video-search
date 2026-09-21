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

## 2. Ranking: the gain is real but format-dependent

MSR-VTT 1k-A, 100 queries, top-50 shortlist, on two index formats.

**Merged captions** (all ~20 human captions per clip, median 771 chars):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| embeddings | 49.0 | 76.0 | 81.0 |
| **+ Jev** | **58.0** | **81.0** | **85.0** |

**Short captions** (3 captions per clip, median 136 chars — closer to what a real
captioner emits):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| **embeddings** | **45.0** | **63.0** | **71.0** |
| + Jev | 42.0 | 59.0 | 67.0 |

**An earlier version of this file reported only the +9 and called it the headline. That
was wrong to state unqualified.** Jev's rerank helps when it has a lot of caption text to
read and slightly hurts when it does not. The merged format is unusually rich — nobody
ships an index of twenty concatenated human captions — so the realistic number for
*ranking* is roughly break-even, not +9.

Run-to-run variance is a few points at n=100; treat small deltas as noise. `recall@50` is
the ceiling: 91% merged, 85% short.

What survives the format change is refusal, not ranking. See §3 and §6.

## 3. Refusal is the durable win

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

## 6. Three-way: embeddings vs Jev vs Laya

[Laya](https://github.com/NandhaKishorM/laya) is an open-weights System-1 decision model
with the same typed-question API (`choice`/`score`/`noul`), so the identical question and
identical shortlist go to both judges. 421M params, ModernBERT-large, run locally on MPS.

`python three_way.py --index index-short -n 100`

| | R@1 merged | R@1 short | AUC merged | AUC short | speed / 50 clips |
|---|---|---|---|---|---|
| embeddings | 49.0 | 45.0 | — | — | 25 ms |
| + Jev | **58.0** | 42.0 | **0.908** | **0.847** | 1.4 s (network, 25 in flight) |
| + Laya | 30.0 | 22.0 | 0.847 | 0.818 | 4.4 s / 2.9 s (local, sequential) |

**Laya loses ranking on both formats.** Its AUC says it separates right from wrong
reasonably on average, yet it wrecks the top of the list: on short captions it demotes the
correct clip 49 times and promotes it 13, where Jev is near break-even at 26 down / 20 up.
Average separation and top-1 precision are different things, and search only cares about
the second.

### The decisive test: can it say "nothing here matches"?

Three queries the corpus can answer, three it cannot. A usable refusal threshold needs the
worst real match to score above the best false alarm:

| | worst real match | best false alarm | margin |
|---|---|---|---|
| Jev, merged | 0.86 | 0.31 | **+0.55** |
| Jev, short | 0.89 | 0.32 | **+0.57** |
| Laya, merged | 0.46 | 0.57 | **−0.11 — overlaps, no threshold works** |
| Laya, short | 0.61 | 0.55 | +0.06 |

Jev holds a wide margin on both formats. Laya's bands overlap on merged captions (a query
with no answer scores *higher* than one with an answer) and clear by 0.06 on short ones,
which will not survive a different corpus.

This is also the one claim that is format-independent, which is why §2's ranking number
should not be the headline and this should.

### Caption form is not a neutral choice

Laya is far more sensitive to it than Jev. Same two clips, same query:

| clip | form | Jev | Laya |
|---|---|---|---|
| correct (outdoor cooking) | merged | 0.97 | 0.50 |
| wrong (indoor pot) | merged | 0.14 | **0.49** |
| correct | one sentence | 0.96 | 0.64 |
| wrong | one sentence | 0.03 | 0.18 |

On merged captions Laya separates correct from wrong by 0.01. It is not truncation —
captions are median 172 tokens, max 373, inside Laya's 512 context. Benchmarking Laya only
on the merged index would have been a rigged test, which is why both formats are reported.

## 7. The caveat that bounds all of it

The index is built from MSR-VTT's **human** captions, not generated ones.

That was deliberate: a weak captioner confounds every number, and you cannot tell a bad
judge from a bad description. It isolates "is Jev a good judge" — at the cost of being an
**upper bound**. A real VLM caption is shorter, vaguer, and sometimes wrong, so expect
lower absolute numbers end to end.

**The Jev-vs-embeddings delta is the transferable result. The absolute R@k values are not.**

The end-to-end video path (`ingest.py --videos`) is written and ffmpeg-tested but has
never run a real captioning pass; a local 8B VLM was too slow on the target machine
(~5 min/clip) and was abandoned in favour of isolating the judge.
