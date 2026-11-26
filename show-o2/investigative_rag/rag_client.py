"""Thin, persistent subprocess client. Never imports Sentence Transformers."""
import atexit
import json
import subprocess
import sys
from pathlib import Path


class RAGClient:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        worker = Path(__file__).resolve().parent / "rag_worker.py"
        self.process = subprocess.Popen(
            [sys.executable, "-I", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # Display worker errors in the terminal.
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        atexit.register(self.close)
        try:
            response = self._request({"command": "init", "root": str(self.root)})
            self.ids = response["ids"]
        except Exception:
            self.close()
            raise

    def _request(self, request):
        if self.process.poll() is not None:
            raise RuntimeError(f"RAG worker exited with code {self.process.returncode}")
        try:
            self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            raise RuntimeError("RAG worker closed its input; see terminal errors") from exc
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("RAG worker terminated without a response; see terminal errors")
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(f"RAG worker error: {result.get('error')}")
        return result

    def search(self, observation, top_k=5, minimum_score=None, include_review_needed=False):
        if include_review_needed:
            raise ValueError("Unreviewed entries are disabled in the RAG worker")
        response = self._request({
            "command": "search",
            "observation": observation,
            "top_k": top_k,
            "minimum_score": minimum_score,
        })
        return response["hits"]

    def close(self):
        process = self.process
        if process.poll() is not None:
            return
        try:
            process.stdin.write('{"command":"quit"}\n')
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            if process.stdout:
                process.stdout.close()