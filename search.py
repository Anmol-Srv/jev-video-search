"""Query -> embedding shortlist -> Jev rerank -> ranked timestamps."""
import argparse
from pathlib import Path

import core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="index")
    ap.add_argument("--top-k", type=int, default=50, help="embedding shortlist size")
    ap.add_argument("-n", type=int, default=10, help="results to show")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--all", action="store_true", help="show results below the floor too")
    args = ap.parse_args()

    rows, vecs = core.load_index(Path(args.index))
    hits, meta = core.search(args.query, rows, vecs, args.top_k, use_jev=not args.no_jev)
    print(f"{len(rows)} scenes | {meta}\n")

    shown = 0
    for h in hits:
        j = h.get("jev")
        if not args.all and meta["applied"] and (j is None or j < core.FLOOR):
            continue
        print(f"{(f'{j:.2f}' if j is not None else '  - ')}  sim {h['sim']:.3f}  "
              f"{h['video']} {h['start']:.1f}-{h['end']:.1f}s\n    {h['caption'][:180]}")
        shown += 1
        if shown >= args.n:
            break
    if not shown:
        print(f"nothing above the floor ({core.FLOOR}). --all to see the shortlist anyway.")


if __name__ == "__main__":
    main()
