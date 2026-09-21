# PRODUCT.md — jev-video-search dashboard

## Register
product — a measurement instrument. Design serves the task; the tool should disappear.

## Platform
web (local only; binds 127.0.0.1)

## Purpose
Decide whether Jev's rerank is worth paying for in video scene search. The dashboard
exists to make that judgement *verifiable*: run a query, see exactly what Jev changed
about the embedding ranking, and watch the actual clip to confirm the call was right.

## Users
One: the developer evaluating the prototype. Fluent, local, terminal open beside this
window. No onboarding, no auth, no multi-user anything.

## Core design principle
**Show the delta, not the two lists.** The naive build is embeddings-left/Jev-right,
which makes the user diff 50 rows by eye. The information that matters is movement:
what Jev promoted, what it demoted, what it rejected outright. Movement is encoded per
row; the cutoff is a physical line in the layout.

## Second principle
**Captions are not evidence.** Every row plays the real clip. Judging a reranker from
caption text alone just moves the trust problem; the video is the ground truth.

## Anti-references
- LLM-eval dashboards (violet accent, gradient headers, hero-metric tiles) — the
  first category reflex.
- Terminal-green "hacker" tooling — the second-order reflex once violet is avoided.
- Consumer video search (Netflix/YouTube grids). This is an instrument, not a browser.

## Visual direction
Grading-suite neutral: near-black, essentially untinted surround, because video is on
screen and a tinted surround biases how the footage reads. One signal color (sage-green,
seed hue 140) carries confidence and promotion. Rejected rows recede by opacity and
desaturation rather than turning red — rejection here is a correct answer, not an error.

## Accessibility
Body text ≥4.5:1. Rank movement never encoded by color alone (always arrow + number).
Full keyboard operation. Honours prefers-reduced-motion.
