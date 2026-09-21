"""Local dashboard server. Stdlib only -- no Flask, no build step.

Serves three things: the dashboard, a search endpoint, and the raw clips.
Binds loopback only; there is no auth because there is nothing to authenticate to.
"""
import json, mimetypes, re, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np

import core

ROOT = Path(__file__).parent
VIDEO_DIR = ROOT / "data" / "clips_raw" / "video"
ROWS, VECS = core.load_index(ROOT / "index")
BY_VIDEO = {r["video"]: i for i, r in enumerate(ROWS)}


def run_search(query: str, top_k: int):
    """Both rankings from one pass, with the timing split kept separate -- the
    point of the readout is to show what the rerank specifically costs."""
    t0 = time.perf_counter()
    qv = core.embed([query])[0]
    sims = VECS @ qv
    order = np.argsort(-sims)[:top_k]
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    scores = core.jev_scores(query, [ROWS[i]["caption"] for i in order])
    t_jev = time.perf_counter() - t1

    hits = [{"video": ROWS[i]["video"], "caption": ROWS[i]["caption"],
             "end": ROWS[i]["end"], "sim": round(float(sims[i]), 4),
             "jev": (round(s, 4) if s is not None else None), "embRank": n + 1}
            for n, (i, s) in enumerate(zip(order, scores))]

    # Rank by Jev, with embedding position as the tiebreak. Candidates Jev could not
    # score keep their embedding rank rather than sinking to the bottom as if rejected.
    ranked = sorted(hits, key=lambda h: (-(h["jev"] if h["jev"] is not None else -1),
                                         h["embRank"]))
    for n, h in enumerate(ranked):
        h["jevRank"] = n + 1

    scored = sum(h["jev"] is not None for h in hits)
    return {
        "query": query, "hits": ranked, "floor": core.FLOOR,
        "applied": scored > 0,
        "stats": {"candidates": len(hits), "scored": scored,
                  "jevCalls": len(hits), "concurrency": core.CONCURRENCY,
                  "embedMs": round(t_embed * 1000), "jevMs": round(t_jev * 1000),
                  "totalMs": round((time.perf_counter() - t0) * 1000),
                  "corpus": len(ROWS)},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        if "/api/" in (a[0] if a else ""):
            print(f"  {a[0]}")

    def _send(self, code, body: bytes, ctype, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._send(200, (ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/info":
            have = sum(1 for r in ROWS if (VIDEO_DIR / r["video"]).is_file())
            return self._send(200, json.dumps({
                "corpus": len(ROWS), "clips": have, "floor": core.FLOOR,
                "jev": bool(core.jev_key()), "model": core.JEV_MODEL,
            }).encode(), "application/json")
        if u.path == "/api/search":
            q = (parse_qs(u.query).get("q") or [""])[0].strip()
            k = int((parse_qs(u.query).get("k") or ["50"])[0])
            if not q:
                return self._send(400, b'{"error":"empty query"}', "application/json")
            try:
                payload = run_search(q, max(1, min(k, 200)))
            except Exception as e:                      # surface it in the UI, don't 500 silently
                return self._send(502, json.dumps({"error": str(e)[:300]}).encode(),
                                  "application/json")
            return self._send(200, json.dumps(payload).encode(), "application/json")
        if u.path == "/api/stream":
            q = (parse_qs(u.query).get("q") or [""])[0].strip()
            k = int((parse_qs(u.query).get("k") or ["50"])[0])
            return self.stream_search(q, max(1, min(k, 200)))
        if u.path.startswith("/video/"):
            return self.serve_video(unquote(u.path[len("/video/"):]))
        self._send(404, b"not found", "text/plain")

    def stream_search(self, query, k):
        """Server-sent events: the clip list goes out immediately, then one event per
        score as it lands. The UI can show a match the moment it is known instead of
        waiting on the slowest of 50."""
        if not query:
            return self._send(400, b'{"error":"empty query"}', "application/json")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")   # no Content-Length; we close when done
        self.end_headers()

        def emit(event, data):
            self.wfile.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode())
            self.wfile.flush()

        try:
            t0 = time.perf_counter()
            qv = core.embed([query])[0]
            sims = VECS @ qv
            order = np.argsort(-sims)[:k]
            embed_ms = round((time.perf_counter() - t0) * 1000)
            clips = [{"i": n, "video": ROWS[i]["video"], "caption": ROWS[i]["caption"],
                      "sim": round(float(sims[i]), 4), "searchRank": n + 1}
                     for n, i in enumerate(order)]
            emit("clips", {"query": query, "clips": clips, "floor": core.FLOOR,
                           "embedMs": embed_ms, "corpus": len(ROWS)})

            t1 = time.perf_counter()
            done = 0
            for n, score in core.jev_stream(query, [c["caption"] for c in clips]):
                done += 1
                emit("score", {"i": n, "score": score, "done": done})
            emit("done", {"checkMs": round((time.perf_counter() - t1) * 1000),
                          "totalMs": round((time.perf_counter() - t0) * 1000),
                          "embedMs": embed_ms, "checked": done, "corpus": len(ROWS)})
        except BrokenPipeError:
            pass                                   # user navigated away mid-search
        except Exception as e:
            try: emit("fail", {"error": str(e)[:300]})
            except Exception: pass

    def serve_video(self, name):
        # Reject anything that isn't a bare filename: this joins onto a real directory.
        if not re.fullmatch(r"[\w.-]+\.mp4", name):
            return self._send(400, b"bad name", "text/plain")
        path = VIDEO_DIR / name
        if not path.is_file():
            return self._send(404, b"no such clip", "text/plain")

        size = path.stat().st_size
        ctype = mimetypes.guess_type(name)[0] or "video/mp4"
        rng = self.headers.get("Range")
        # Safari refuses to play a video the server didn't range-serve, and seeking
        # is broken everywhere without it. 206 is not optional here.
        if not rng:
            return self._send(200, path.read_bytes(), ctype, {"Accept-Ranges": "bytes"})

        m = re.match(r"bytes=(\d*)-(\d*)", rng)
        start = int(m.group(1)) if m and m.group(1) else 0
        end = int(m.group(2)) if m and m.group(2) else size - 1
        start, end = max(0, start), min(end, size - 1)
        if start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return
        with path.open("rb") as fh:
            fh.seek(start)
            chunk = fh.read(end - start + 1)
        self._send(206, chunk, ctype, {"Accept-Ranges": "bytes",
                                       "Content-Range": f"bytes {start}-{end}/{size}"})


if __name__ == "__main__":
    have = sum(1 for r in ROWS if (VIDEO_DIR / r["video"]).is_file())
    print(f"index: {len(ROWS)} scenes | clips on disk: {have}/{len(ROWS)}"
          f" | jev: {'configured' if core.jev_key() else 'NO KEY'}")
    print("http://127.0.0.1:8420")
    ThreadingHTTPServer(("127.0.0.1", 8420), Handler).serve_forever()
