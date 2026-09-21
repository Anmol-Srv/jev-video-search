"""Embeddings vs +Jev vs +Laya on the same shortlist, same question, same corpus.

Two things make a naive comparison wrong, and both are handled here:

1. The judges are calibrated differently. Jev's probabilities are bimodal (a match
   ~0.97, a miss ~0.05); Laya's sit in a narrow band (~0.61 vs ~0.45 on the same
   query). Any single shared threshold flatters one and buries the other, so ranking
   is measured threshold-free and each judge gets its own threshold, fitted here.

2. Caption form matters to one of them. Run this against both index/ (all ~20 human
   captions merged) and index-short/ (3 captions, closer to what a captioner emits).
"""
import argparse, json, statistics as st, time
from pathlib import Path

import numpy as np

import core


def roc_auc(pos, neg):
    """P(a random positive scores above a random negative). Ties count as half."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    rsum = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    return (rsum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(pos, neg):
    """Youden's J: the cut that maximises (true-positive rate - false-positive rate)."""
    best = (0.5, -1)
    for t in [i / 200 for i in range(201)]:
        tpr = sum(1 for v in pos if v >= t) / len(pos)
        fpr = sum(1 for v in neg if v >= t) / len(neg)
        if tpr - fpr > best[1]:
            best = (t, tpr - fpr)
    return best[0]


def collect(args, rows, vecs, gt):
    recs, t_emb, t_jev, t_laya = [], [], [], []
    for n, g in enumerate(gt, 1):
        q, truth = g["caption"], g["video"]
        t0 = time.perf_counter(); qv = core.embed([q])[0]; sims = vecs @ qv
        order = np.argsort(-sims)[:args.top_k]; t_emb.append(time.perf_counter() - t0)
        caps = [rows[i]["caption"] for i in order]

        t0 = time.perf_counter(); J = core.jev_scores(q, caps);  t_jev.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); L = core.laya_scores(q, caps); t_laya.append(time.perf_counter() - t0)

        recs.append({"q": q, "truth": truth,
                     "hits": [{"video": rows[i]["video"], "sim": float(sims[i]),
                               "jev": j, "laya": l}
                              for i, j, l in zip(order, J, L)]})
        if n % 10 == 0:
            print(f"  …{n}/{len(gt)}", flush=True)
    return recs, {"emb": st.median(t_emb) * 1000, "jev": st.median(t_jev) * 1000,
                  "laya": st.median(t_laya) * 1000}


def report(recs, times, top_k, label):
    n = len(recs)
    def rank_of(seq, truth):
        for i, h in enumerate(seq):
            if h["video"] == truth:
                return i + 1
        return None

    def ranks_for(key):
        out = []
        for r in recs:
            hits = r["hits"]
            if key:
                hits = sorted(enumerate(hits),
                              key=lambda p: (-(p[1][key] if p[1][key] is not None else -1), p[0]))
                hits = [h for _, h in hits]
            out.append(rank_of(hits, r["truth"]))
        return out

    E, J, L = ranks_for(None), ranks_for("jev"), ranks_for("laya")
    rk = lambda rs, k: 100 * sum(1 for x in rs if x is not None and x <= k) / len(rs)

    print(f"\n\n=== {label} · {n} queries · top-{top_k} shortlist ===")
    print(f"\nRANKING (threshold-free)\n{'':<14}{'R@1':>8}{'R@5':>8}{'R@10':>8}")
    for name, rs in (("embeddings", E), ("+ jev", J), ("+ laya", L)):
        print(f"{name:<14}{rk(rs,1):>8.1f}{rk(rs,5):>8.1f}{rk(rs,10):>8.1f}")
    print(f"recall@{top_k}: {100*sum(x is not None for x in E)/n:.1f}%  <- ceiling for both judges")

    print(f"\nSEPARATION (can it tell the right clip from the other {top_k-1}?)")
    print(f"{'':<14}{'AUC':>8}{'best cut':>10}{'TPR':>8}{'FPR':>8}{'true clip':>11}{'others':>9}")
    for name, key in (("jev", "jev"), ("laya", "laya")):
        pos = [h[key] for r in recs for h in r["hits"] if h["video"] == r["truth"] and h[key] is not None]
        neg = [h[key] for r in recs for h in r["hits"] if h["video"] != r["truth"] and h[key] is not None]
        if not pos: continue
        t = best_threshold(pos, neg)
        tpr = 100 * sum(1 for v in pos if v >= t) / len(pos)
        fpr = 100 * sum(1 for v in neg if v >= t) / len(neg)
        print(f"{name:<14}{roc_auc(pos,neg):>8.3f}{t:>10.2f}{tpr:>7.0f}%{fpr:>7.0f}%"
              f"{st.mean(pos):>11.2f}{st.mean(neg):>9.2f}")
    print("AUC is threshold-free. 'best cut' is Youden's J fitted on this data, which is\n"
          "the most generous reading each judge can get; TPR/FPR are measured at it.")

    print(f"\nSPEED (median ms per search, {top_k} candidates)")
    for name, v, note in (("embeddings", times["emb"], "one Ollama call + a dot product"),
                          ("+ jev", times["jev"], f"{core.CONCURRENCY} in flight, network"),
                          ("+ laya", times["laya"], "sequential, local on MPS")):
        print(f"{name:<14}{v:>9.0f}   {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("--gt", default="data/msrvtt_test_1k.json")
    ap.add_argument("-n", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--replay", action="store_true", help="report from cache, no scoring")
    args = ap.parse_args()

    cache = Path(args.cache or f"{args.index}/three_way.json")
    rows, vecs = core.load_index(Path(args.index))
    if args.replay and cache.exists():
        d = json.loads(cache.read_text())
        return report(d["recs"], d["times"], d["top_k"], f"{args.index} (cached)")

    import random
    gt = [g for g in json.load(open(args.gt)) if g["video"] in {r["video"] for r in rows}]
    random.Random(0).shuffle(gt)
    gt = gt[:args.n]
    print(f"{args.index}: {len(rows)} clips · scoring {len(gt)} queries with jev and laya")
    recs, times = collect(args, rows, vecs, gt)
    cache.write_text(json.dumps({"recs": recs, "times": times, "top_k": args.top_k}))
    report(recs, times, args.top_k, args.index)


if __name__ == "__main__":
    main()
