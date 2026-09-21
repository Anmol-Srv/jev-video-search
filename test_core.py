"""Smallest checks that fail if the ranking logic breaks. Run: python test_core.py"""
import numpy as np
import core


def rows_fixture():
    caps = ["laptop in a car", "man driving", "laptop on a desk", "guitar on stage"]
    return [{"video": f"v{i}.mp4", "start": 0.0, "end": 5.0, "caption": c}
            for i, c in enumerate(caps)]


def test_rerank_promotes_and_keeps_everything():
    rows = rows_fixture()
    vecs = np.eye(4, dtype=np.float32)
    core.embed = lambda t: np.array([[0, 0, 1, 0]], dtype=np.float32)   # embeds prefer row 2
    # Key by caption, not position: jev_scores receives the SHORTLIST order, so a
    # positional stub silently tests a different pairing than production uses.
    want = {"laptop in a car": 0.9, "man driving": 0.1, "laptop on a desk": 0.2,
            "guitar on stage": 0.0}
    core.jev_scores = lambda q, caps: [want[c] for c in caps]           # jev prefers row 0

    hits, meta = core.search("laptop being used inside a car", rows, vecs, top_k=4)
    assert meta["applied"] and meta["scored"] == 4, meta
    assert hits[0]["caption"] == "laptop in a car", hits[0]
    # Rerank is a reorder, never a drop: the floor filters at display time only.
    assert len(hits) == 4
    assert meta["above_floor"] == 1, meta


def test_falls_back_to_embedding_order_when_jev_is_down():
    rows = rows_fixture()
    vecs = np.eye(4, dtype=np.float32)
    core.embed = lambda t: np.array([[0, 0, 1, 0]], dtype=np.float32)
    core.jev_scores = lambda q, caps: [None] * len(caps)

    hits, meta = core.search("anything", rows, vecs, top_k=4)
    assert meta == {"applied": False, "reason": "unavailable"}, meta
    assert hits[0]["caption"] == "laptop on a desk", hits[0]   # embedding's pick, unchanged


def test_unscored_candidates_sink_below_scored_ones():
    rows = rows_fixture()
    vecs = np.eye(4, dtype=np.float32)
    core.embed = lambda t: np.array([[1, 0, 0, 0]], dtype=np.float32)
    # Row 0 is the embedding's top hit but Jev failed on it; a scored row must win.
    core.jev_scores = lambda q, caps: [0.8 if c == "man driving" else None for c in caps]

    hits, _ = core.search("q", rows, vecs, top_k=4)
    assert hits[0]["caption"] == "man driving", hits[0]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"ok  {name}")
