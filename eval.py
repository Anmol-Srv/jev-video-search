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


def ranks(query, truth, rows, vecs, top_k):
    """(embedding rank, jev rank, jev score of the true clip). rank None = missed."""
    qv = core.embed([query])[0]
    order = np.argsort(-(vecs @ qv))[:top_k]
    shortlist = [rows[i] for i in order]

    def rank_of(seq):
        for i, r in enumerate(seq):
            if r["video"] == truth:
                return i + 1
        return None

    emb_rank = rank_of(shortlist)
    scores = core.jev_scores(query, [s["caption"] for s in shortlist])
    if all(s is None for s in scores):
        return emb_rank, emb_rank, None, 0
    pairs = list(zip(shortlist, scores))
    reranked = sorted(enumerate(pairs),
                      key=lambda p: (-(p[1][1] if p[1][1] is not None else -1), p[0]))
    jev_rank = rank_of([p[0] for _, p in reranked])
    truth_score = next((s for r, s in pairs if r["video"] == truth), None)
    false_pos = sum(1 for r, s in pairs
                    if s is not None and s >= core.FLOOR and r["video"] != truth)
    return emb_rank, jev_rank, truth_score, false_pos


def eval_msrvtt(args, rows, vecs):
    gt = json.load(open(args.gt))
    have = {r["video"] for r in rows}
    gt = [g for g in gt if g["video"] in have]
    random.Random(0).shuffle(gt)
    gt = gt[:args.n]
    print(f"{len(gt)} queries over {len(rows)} indexed scenes, top_k={args.top_k}\n")

    E, J, hits_above, fps, ceiling = [], [], 0, [], 0
    for i, g in enumerate(gt, 1):
        e, j, ts, fp = ranks(g["caption"], g["video"], rows, vecs, args.top_k)
        E.append(e); J.append(j); fps.append(fp)
        ceiling += e is not None
        hits_above += ts is not None and ts >= core.FLOOR
        if i % 10 == 0:
            print(f"  ...{i}/{len(gt)}", flush=True)

    def rk(rs, k):
        return 100 * sum(1 for r in rs if r is not None and r <= k) / len(rs)

    print(f"\n{'':<14}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    print(f"{'embeddings':<14}{rk(E,1):>8.1f}{rk(E,5):>8.1f}{rk(E,10):>8.1f}")
    print(f"{'+ jev':<14}{rk(J,1):>8.1f}{rk(J,5):>8.1f}{rk(J,10):>8.1f}")
    print(f"\nrecall@{args.top_k}: {100*ceiling/len(gt):.1f}%  <- ceiling; jev cannot "
          f"rank what retrieval never surfaced")
    print(f"cutoff: true clip scored >= {core.FLOOR} on {100*hits_above/len(gt):.1f}% of "
          f"queries, with {np.mean(fps):.1f} false positives above the floor per query")


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
    args = ap.parse_args()

    rows, vecs = core.load_index(Path(args.index))
    if args.queries:
        eval_queries(args, rows, vecs)
    else:
        eval_msrvtt(args, rows, vecs)


if __name__ == "__main__":
    main()
