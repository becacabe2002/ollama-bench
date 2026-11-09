"""Ollama HTTP wrapper and per-trial metric capture."""

from __future__ import annotations

import threading
from pathlib import Path

import psutil
import requests

PROMPT_TEMPLATE = "Summarize the following document in under 200 words.\n\nDocument:\n{doc}"


def build_prompt(doc: str) -> str:
    # str.format would choke on braces inside the document.
    return PROMPT_TEMPLATE.replace("{doc}", doc)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, url: str, timeout_s: float = 30):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, **kwargs) -> dict:
        resp = requests.request(method, self.url + path, timeout=kwargs.pop("timeout", 30), **kwargs)

        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text)
            except ValueError:
                detail = resp.text
            raise OllamaError(f"{method} {path} -> HTTP {resp.status_code}: {detail}")
        
        return resp.json()

    def version(self) -> str:
        return self._request("GET", "/api/version")["version"]

    def tags(self) -> dict[str, dict]:
        """Installed models keyed by full tag, e.g. `phi4-mini:latest`."""
        return {m["name"]: m for m in self._request("GET", "/api/tags").get("models", [])}

    def show(self, model: str) -> dict:
        return self._request("POST", "/api/show", json={"model": model})

    def generate(self, model: str, prompt: str, options: dict, think: bool | None = None) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
            "keep_alive": 0,
        }
        if think is not None:
            payload["think"] = think
        # Non-streaming sends no bytes until done, so the read timeout bounds the whole call.
        return self._request("POST", "/api/generate", json=payload, timeout=self.timeout_s)


def full_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def context_length(show_response: dict) -> int | None:
    for key, value in show_response.get("model_info", {}).items():
        if key.endswith(".context_length"):
            return int(value)
    return None


def response_metrics(resp: dict) -> dict:
    """Convert Ollama's nanosecond counters into the raw-row schema."""
    prompt_count = resp.get("prompt_eval_count", 0)
    prompt_s = resp.get("prompt_eval_duration", 0) / 1e9
    eval_count = resp.get("eval_count", 0)
    eval_s = resp.get("eval_duration", 0) / 1e9
    text = resp.get("response", "")
    return {
        "load_s": round(resp.get("load_duration", 0) / 1e9, 3),
        "prompt_eval_count": prompt_count,
        "prompt_eval_s": round(prompt_s, 3),
        "prompt_tps": round(prompt_count / prompt_s, 2) if prompt_s else None,
        "eval_count": eval_count,
        "eval_s": round(eval_s, 3),
        "gen_tps": round(eval_count / eval_s, 2) if eval_s else None,
        "word_count": len(text.split()),
        "response_text": text,
    }


def find_server_process() -> psutil.Process | None:
    for proc in psutil.process_iter(["name", "cmdline"]):
        name = (proc.info["name"] or "").lower()
        if name.startswith("ollama") and "serve" in (proc.info["cmdline"] or []):
            return proc
    return None


def read_vmhwm_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def read_swap_free_kb() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("SwapFree:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


class MemorySampler:
    """Samples RSS and VmHWM across the Ollama process tree on a background thread.

    Weights live in the `ollama runner` child, not the `ollama serve` PID, so the
    whole tree is summed. With keep_alive=0 the runner exits as soon as the response
    returns, so VmHWM has to be captured while it runs instead of read afterwards.
    """

    def __init__(self, server: psutil.Process | None, interval_s: float):
        self.server = server
        self.interval_s = interval_s
        self.peak_rss_bytes = 0
        self.vmhwm_kb: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> None:
        try:
            tree = [self.server, *self.server.children(recursive=True)]
        except psutil.Error:
            return
        total = 0
        for proc in tree:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                continue
            hwm = read_vmhwm_kb(proc.pid)
            if hwm is not None:
                self.vmhwm_kb[proc.pid] = max(hwm, self.vmhwm_kb.get(proc.pid, 0))
        self.peak_rss_bytes = max(self.peak_rss_bytes, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "MemorySampler":
        if self.server is not None:
            self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    @property
    def peak_rss_gb(self) -> float | None:
        return round(self.peak_rss_bytes / 2**30, 3) if self.peak_rss_bytes else None

    @property
    def vmhwm_gb(self) -> float | None:
        return round(sum(self.vmhwm_kb.values()) / 2**20, 3) if self.vmhwm_kb else None
