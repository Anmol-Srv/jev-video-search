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
    # Only the kept sets, not the whole shortlist. Embeddings have no cutoff, so
    # "kept by embedding" is defined as its top-N at the SAME N Jev kept: the
    # apples-to-apples question is "what would each method actually have shown?"
    for r in results[:args.lists]:
        n = len(r["kept"])
        jev_set = r["kept"]
        shown = n if n else args.rows          # nothing kept -> show what emb would have anyway
        emb_set = r["emb"][:shown]

        jev_rank = {p[0]["video"]: i + 1 for i, p in enumerate(r["jev"])}
        emb_rank = {p[0]["video"]: i + 1 for i, p in enumerate(r["emb"])}
        jev_score = {p[0]["video"]: p[2] for p in r["jev"]}

        head = (f'kept by jev {n}  ·  kept by embedding = its top {shown}'
                if n else f'jev kept NOTHING  ·  embedding would still show its top {shown}')
        print(f'\n\nLIST  "{r["query"]}"\n      {head}')
        print(f"{'':<2}{'KEPT BY JEV':<44}{'':2}{'KEPT BY EMBEDDING':<44}")
        print(f"{'#':>2}  {'score':<6}{'clip':<13}{'was emb#':<10}{'':2}"
              f"{'sim':<7}{'clip':<13}{'jev':<6}{'verdict':<9}")
        print("-" * 98)
        for i in range(max(len(jev_set), len(emb_set))):
            if i < len(jev_set):
                v = jev_set[i][0]["video"][:-4]
                left = f"{jev_set[i][2]:<6.2f}{v:<13}#{emb_rank[v + '.mp4']:<9}"
            else:
                left = " " * 29
            if i < len(emb_set):
                e = emb_set[i]; v = e[0]["video"][:-4]
                js = jev_score[e[0]["video"]]
                kept = js is not None and js >= core.FLOOR
                right = (f"{e[1]:<7.3f}{v:<13}{(f'{js:.2f}' if js is not None else '  - '):<6}"
                         f"{'kept' if kept else 'JEV CUT':<9}")
            else:
                right = ""
            print(f"{i+1:>2}  {left}{'':2}{right}")

        a = {p[0]["video"] for p in jev_set}
        b = {p[0]["video"] for p in emb_set}
        same = len(a & b)
        print("-" * 98)
        if n:
            print(f"    same clips in both sets: {same}/{n}"
                  f"   ·   only jev would show: {len(a - b)}"
                  f"   ·   only embedding would show: {len(b - a)}")
            cut = [p for p in emb_set if not (jev_score[p[0]['video']] is not None
                                              and jev_score[p[0]['video']] >= core.FLOOR)]
            if cut:
                print(f"    embedding would have shown {len(cut)} clip(s) jev rejects, e.g. "
                      f"{cut[0][0]['video'][:-4]} (sim {cut[0][1]:.3f}, jev {jev_score[cut[0][0]['video']]:.2f})")
                print(f"      \"{cut[0][0]['caption'][:120]}\"")
        else:
            print(f"    embedding would show {shown} clips here; jev shows none.")
            print(f"      top pick {emb_set[0][0]['video'][:-4]} sim {emb_set[0][1]:.3f} / "
                  f"jev {jev_score[emb_set[0][0]['video']]:.2f}")
            print(f"      \"{emb_set[0][0]['caption'][:120]}\"")


if __name__ == "__main__":
    main()
