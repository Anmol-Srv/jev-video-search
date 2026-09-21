"""Measure what Jev actually adds.

Two modes, because they answer different questions:

  --split msrvtt   Ground-truth retrieval (1 correct clip per query). Gives R@1/5/10
                   for embeddings alone vs embeddings+Jev, plus recall@K -- the hard
                   ceiling, since Jev can only reorder what retrieval surfaced.
                   MSR-VTT queries are literal descriptions, which is the case
                   embeddings already handle, so a flat result here is expected.

  --queries FILE   Compositional/relational queries with no ground truth. Prints
                   top-10 with and without Jev for hand judging. This is where the
                   two are expected to diverge.

Both report cutoff behaviour: an embedding score has no absolute meaning, so
embedding-only search can never answer "none of these match". A Jev probability can.
"""
import argparse, json, random
from pathlib import Path

import numpy as np

import core


def score_query(query, truth, rows, vecs, top_k):
    """Score one query. Returns the shortlist with both signals attached, so the
    threshold sweep can be replayed from disk instead of re-billing Jev."""
    qv = core.embed([query])[0]
    order = np.argsort(-(vecs @ qv))[:top_k]
    shortlist = [rows[i] for i in order]

    def rank_of(seq):
        for i, r in enumerate(seq):
            if r["video"] == truth:
                return i + 1
        return None

    sims = vecs @ qv
    scores = core.jev_scores(query, [s["caption"] for s in shortlist])
    return {"query": query, "truth": truth,
            "hits": [{"video": r["video"], "sim": float(sims[i]), "jev": j}
                     for i, r, j in zip(order, shortlist, scores)]}


def measure(records, floor):
    """Replayable metrics: everything here is derived from cached scores."""
    E, J, above, fps = [], [], 0, []

    def rank_of(seq, truth):
        for i, h in enumerate(seq):
            if h["video"] == truth:
                return i + 1
        return None

    for rec in records:
        hits, truth = rec["hits"], rec["truth"]
        E.append(rank_of(hits, truth))
        if all(h["jev"] is None for h in hits):
            J.append(E[-1]); fps.append(0); continue
        rr = sorted(enumerate(hits),
                    key=lambda p: (-(p[1]["jev"] if p[1]["jev"] is not None else -1), p[0]))
        J.append(rank_of([h for _, h in rr], truth))
        ts = next((h["jev"] for h in hits if h["video"] == truth), None)
        above += ts is not None and ts >= floor
        fps.append(sum(1 for h in hits
                       if h["jev"] is not None and h["jev"] >= floor and h["video"] != truth))
    return E, J, above, fps


def eval_msrvtt(args, rows, vecs):
    gt = json.load(open(args.gt))
    have = {r["video"] for r in rows}
    gt = [g for g in gt if g["video"] in have]
    random.Random(0).shuffle(gt)
    gt = gt[:args.n]
    print(f"{len(gt)} queries over {len(rows)} indexed scenes, top_k={args.top_k}\n")

    records = []
    for i, g in enumerate(gt, 1):
        records.append(score_query(g["caption"], g["video"], rows, vecs, args.top_k))
        if i % 10 == 0:
            print(f"  ...{i}/{len(gt)}", flush=True)
    Path(args.dump).write_text(json.dumps(records))
    print(f"scores cached -> {args.dump} (replay sweeps with --sweep, no new API calls)")
    report(records, args.top_k)


def report(records, top_k):
    E, J, above, fps = measure(records, core.FLOOR)
    n = len(records)

    def rk(rs, k):
        return 100 * sum(1 for r in rs if r is not None and r <= k) / len(rs)

    print(f"\n{'':<14}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    print(f"{'embeddings':<14}{rk(E,1):>8.1f}{rk(E,5):>8.1f}{rk(E,10):>8.1f}")
    print(f"{'+ jev':<14}{rk(J,1):>8.1f}{rk(J,5):>8.1f}{rk(J,10):>8.1f}")
    print(f"\nrecall@{top_k}: {100*sum(r is not None for r in E)/n:.1f}%  <- ceiling; jev "
          f"cannot rank what retrieval never surfaced")

    # An embedding score has no absolute meaning, so "none of these match" is not a
    # question embedding-only search can answer. Whether Jev can depends entirely on
    # where the floor sits -- so show the tradeoff instead of asserting one number.
    print(f"\ncutoff sweep (can it say 'no'?)")
    print(f"{'floor':>7}{'true clip kept':>17}{'false pos/query':>18}")
    for f in (0.3, 0.5, 0.55, 0.7, 0.8, 0.9, 0.95):
        _, _, a, fp = measure(records, f)
        print(f"{f:>7}{100*a/n:>16.0f}%{np.mean(fp):>18.1f}")


def eval_queries(args, rows, vecs):
    for q in [l.strip() for l in Path(args.queries).read_text().splitlines()
              if l.strip() and not l.startswith("#")]:
        print(f"\n{'='*78}\nQ: {q}")
        for label, use_jev in (("embeddings", False), ("+ jev", True)):
            hits, _ = core.search(q, rows, vecs, args.top_k, use_jev=use_jev)
            print(f"\n  -- {label}")
            for h in hits[:5]:
                j = h.get("jev")
                mark = "" if j is None else ("  PASS" if j >= core.FLOOR else "  drop")
                print(f"    {(f'{j:.2f}' if j is not None else '    ')} sim {h['sim']:.3f} "
                      f"{h['video']}{mark}\n        {h['caption'][:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--split", choices=["msrvtt"])
    ap.add_argument("--gt", default="data/msrvtt_test_1k.json")
    ap.add_argument("--queries")
    ap.add_argument("-n", type=int, default=100, help="queries to evaluate")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--dump", default="index/eval_scores.json")
    ap.add_argument("--sweep", help="replay metrics from a cached dump; no API calls")
    args = ap.parse_args()

    if args.sweep:
        report(json.loads(Path(args.sweep).read_text()), args.top_k)
        return
    rows, vecs = core.load_index(Path(args.index))
    if args.queries:
        eval_queries(args, rows, vecs)
    else:
        eval_msrvtt(args, rows, vecs)


if __name__ == "__main__":
    main()
