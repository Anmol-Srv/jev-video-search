"""Side-by-side: embeddings vs embeddings+Jev on speed, count, and ranked list.

Speed is split three ways because "embedding search" is two very different costs:
one network call to Ollama for the query vector, then a dot product over the whole
corpus. Only the second scales with corpus size; only the first is comparable to Jev.
"""
import argparse, statistics as st, time
from pathlib import Path

import numpy as np

import core

QUERIES = [
    "a car driving through snow",
    "a person cooking food outdoors",
    "someone playing a musical instrument while sitting down",
    "a person talking directly to the camera in a kitchen",
    "a crowd watching a performance on a stage",
    "someone typing on a computer in an office",
    "a dog running on a beach",
    "a plane flying above the clouds",
]


def measure(query, rows, vecs, k):
    t0 = time.perf_counter(); qv = core.embed([query])[0]; t_q = time.perf_counter() - t0
    t0 = time.perf_counter(); sims = vecs @ qv; order = np.argsort(-sims)[:k]
    t_rank = time.perf_counter() - t0

    shortlist = [rows[i] for i in order]
    t0 = time.perf_counter()
    scores = core.jev_scores(query, [s["caption"] for s in shortlist])
    t_jev = time.perf_counter() - t0

    pairs = [(rows[i], float(sims[i]), s) for i, s in zip(order, scores)]
    ranked = sorted(enumerate(pairs), key=lambda p: (-(p[1][2] if p[1][2] is not None else -1), p[0]))
    kept = [p for _, p in ranked if p[2] is not None and p[2] >= core.FLOOR]
    return {"query": query, "emb": pairs, "jev": [p for _, p in ranked], "kept": kept,
            "tq": t_q, "trank": t_rank, "tjev": t_jev, "k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="index")
    ap.add_argument("-k", type=int, default=50)
    ap.add_argument("--lists", type=int, default=2, help="queries to print full lists for")
    ap.add_argument("--rows", type=int, default=10, help="depth of the printed lists")
    args = ap.parse_args()

    rows, vecs = core.load_index(Path(args.index))
    print(f"corpus {len(rows)} scenes · shortlist k={args.k} · floor {core.FLOOR} · "
          f"jev {core.JEV_MODEL} @ {core.CONCURRENCY} concurrent\n")

    results = [measure(q, rows, vecs, args.k) for q in QUERIES]

    # ---- SPEED -------------------------------------------------------------
    print("SPEED  (ms per search)")
    print(f"{'query':<52}{'embed':>9}{'rank':>9}{'jev':>9}{'jev/call':>10}{'slower':>9}")
    print("-" * 98)
    for r in results:
        emb_total = (r["tq"] + r["trank"]) * 1000
        print(f"{r['query'][:50]:<52}{r['tq']*1000:>9.1f}{r['trank']*1000:>9.2f}"
              f"{r['tjev']*1000:>9.0f}{r['tjev']*1000/r['k']:>10.1f}"
              f"{r['tjev']*1000/emb_total:>8.0f}x")
    embs = [(r["tq"] + r["trank"]) * 1000 for r in results]
    jevs = [r["tjev"] * 1000 for r in results]
    print("-" * 98)
    print(f"{'median':<52}{st.median([r['tq']*1000 for r in results]):>9.1f}"
          f"{st.median([r['trank']*1000 for r in results]):>9.2f}"
          f"{st.median(jevs):>9.0f}{st.median(jevs)/args.k:>10.1f}"
          f"{st.median(jevs)/st.median(embs):>8.0f}x")

    # ---- COUNT -------------------------------------------------------------
    print("\n\nCOUNT  (what each method is willing to return)")
    print(f"{'query':<52}{'embeddings':>12}{'jev kept':>10}{'jev cut':>9}{'reduction':>11}")
    print("-" * 98)
    for r in results:
        cut = r["k"] - len(r["kept"])
        print(f"{r['query'][:50]:<52}{r['k']:>12}{len(r['kept']):>10}{cut:>9}"
              f"{100*cut/r['k']:>10.0f}%")
    tot_kept = sum(len(r["kept"]) for r in results)
    tot = sum(r["k"] for r in results)
    print("-" * 98)
    print(f"{'total':<52}{tot:>12}{tot_kept:>10}{tot-tot_kept:>9}{100*(tot-tot_kept)/tot:>10.0f}%")
    print("\nEmbeddings always return k. There is no score at which they abstain: an\n"
          "embedding similarity has no absolute meaning, only a relative one.")

    # ---- LIST --------------------------------------------------------------
    for r in results[:args.lists]:
        print(f"\n\nLIST  \"{r['query']}\"   top {args.rows}")
        print(f"{'#':>3}  {'EMBEDDINGS':<44}{'':>4}{'+ JEV':<44}")
        print("-" * 98)
        for i in range(args.rows):
            e = r["emb"][i] if i < len(r["emb"]) else None
            j = r["jev"][i] if i < len(r["jev"]) else None
            el = f"{e[1]:.3f} {e[0]['video'][:-4]:<12}" if e else ""
            if j:
                mark = "keep" if (j[2] is not None and j[2] >= core.FLOOR) else "cut "
                jl = f"{j[2]:.2f} {j[0]['video'][:-4]:<12} {mark}"
            else:
                jl = ""
            same = "  =" if e and j and e[0]["video"] == j[0]["video"] else "   "
            print(f"{i+1:>3}  {el:<44}{same:>4}{jl:<44}")
        overlap = len({p[0]["video"] for p in r["emb"][:args.rows]}
                      & {p[0]["video"] for p in r["jev"][:args.rows]})
        print(f"     top-{args.rows} overlap: {overlap}/{args.rows} "
              f"({100*overlap/args.rows:.0f}% of the list is the same clips, reordered)")


if __name__ == "__main__":
    main()
