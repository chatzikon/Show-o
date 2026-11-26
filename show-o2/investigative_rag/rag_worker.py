"""Isolated Sentence Transformers + FAISS worker. Started with: python -I rag_worker.py."""
import contextlib
import json
import sys
import traceback
from pathlib import Path

# Under -I, the script directory is not automatically importable.
# This directory has NO local `datasets` package; keep parent project paths out.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def reply(message):
    sys.stdout.write(json.dumps(message, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    retriever = None
    for raw in sys.stdin:
        try:
            req = json.loads(raw)
            command = req.get("command")
            if command == "init":
                from retrieval import Retriever
                with contextlib.redirect_stdout(sys.stderr):
                    retriever = Retriever(Path(req["root"]).resolve())
                reply({"ok": True, "ids": retriever.ids})
            elif command == "search":
                if retriever is None:
                    raise RuntimeError("Worker not initialized")
                with contextlib.redirect_stdout(sys.stderr):
                    hits = retriever.search(
                        req["observation"],
                        top_k=req.get("top_k", 5),
                        minimum_score=req.get("minimum_score"),
                        include_review_needed=False,
                    )
                reply({"ok": True, "hits": hits})
            elif command == "quit":
                reply({"ok": True})
                return
            else:
                raise ValueError(f"Unknown command: {command}")
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main()