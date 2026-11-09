"""Benchmark runner: 3 trials interleaved across 7 models, one JSONL row per trial.

Runs on the Pouta VM; needs no API key. Re-running with the same --out resumes,
skipping every (model, trial) already in the file, failed ones included. Delete a
failed row's line to retry it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _ollama import (
    MemorySampler,
    OllamaClient,
    build_prompt,
    find_server_process,
    full_tag,
    read_swap_free_kb,
    response_metrics,
)
from _preflight import ROOT, PreflightError, load_yaml, run_preflight, sha256_file, write_manifest

RESTART_SCRIPT = ROOT / "scripts/restart_ollama.sh"
RESTART_TIMEOUT_S = 180

log = logging.getLogger("bench")


def restart_ollama(url: str) -> None:
    subprocess.run(
        ["bash", str(RESTART_SCRIPT)],
        check=True,
        timeout=RESTART_TIMEOUT_S,
        env={**os.environ, "OLLAMA_URL": url},
    )


def load_completed(path: Path) -> set[tuple[str, int]]:
    """Pairs already recorded. Repairs a final line cut off by a crash."""
    if not path.exists():
        return set()
    done = set()
    raw = path.read_text(encoding="utf-8")
    for line in raw.splitlines():
        try:
            row = json.loads(line)
            done.add((row["model"], row["trial"]))
        except (json.JSONDecodeError, KeyError):
            log.warning("ignoring malformed line in %s: %.80s", path, line)
    if raw and not raw.endswith("\n"):
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
    return done


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_trial(
    client: OllamaClient,
    model: str,
    prompt: str,
    trial: int,
    cfg: dict,
    base_row: dict,
    restart: bool,
) -> dict:
    options = {
        "temperature": cfg["temperature"],
        "seed": cfg["seed"],
        "num_ctx": cfg["num_ctx"],
        "num_predict": cfg["num_predict"],
    }
    row = {
        **base_row,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": model,
        "trial": trial,
        "status": "failed",
        "error": None,
        "wall_s": None,
        "load_s": None,
        "prompt_eval_count": None,
        "prompt_eval_s": None,
        "prompt_tps": None,
        "eval_count": None,
        "eval_s": None,
        "gen_tps": None,
        "peak_rss_gb": None,
        "vmhwm_gb": None,
        "word_count": None,
        "swap_touched": None,
        "response_text": None,
        "config": {**options, **({"think": cfg["think"]} if "think" in cfg else {})},
    }
    try:
        if restart:
            restart_ollama(cfg["ollama_url"])
        server = find_server_process()
        if server is None:
            log.warning("ollama server process not found; memory metrics will be null")

        swap_before = read_swap_free_kb()
        t0 = time.monotonic()
        try:
            with MemorySampler(server, cfg["rss_sample_interval_s"]) as sampler:
                resp = client.generate(model, prompt, options, think=cfg.get("think"))
        finally:
            row["wall_s"] = round(time.monotonic() - t0, 3)
            row["peak_rss_gb"] = sampler.peak_rss_gb
            row["vmhwm_gb"] = sampler.vmhwm_gb
            swap_after = read_swap_free_kb()
            if swap_before is not None and swap_after is not None:
                row["swap_touched"] = swap_after < swap_before

        row.update(response_metrics(resp))
        if not row["response_text"].strip():
            # Typically a thinking model that spent all of num_predict on reasoning.
            row["error"] = (
                f"empty response (done_reason={resp.get('done_reason')}, "
                f"thinking_chars={len(resp.get('thinking') or '')})"
            )
        else:
            row["status"] = "ok"
        if row["prompt_eval_count"] + cfg["num_predict"] > cfg["num_ctx"]:
            log.warning(
                "%s: prompt_eval_count=%d leaves under %d tokens of num_ctx=%d; prompt may be truncated",
                model, row["prompt_eval_count"], cfg["num_predict"], cfg["num_ctx"],
            )
    except Exception as exc:  # OOM, timeout, HTTP 500: record it and keep the loop alive
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/bench.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "config/models.yaml")
    parser.add_argument("--doc", type=Path, default=ROOT / "data/doc.txt")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/doc_questions.json")
    parser.add_argument("--manifests", type=Path, default=ROOT / "results/manifests")
    parser.add_argument("--out", type=Path, required=True, help="results/raw/run_<id>.jsonl")
    parser.add_argument("--no-restart", action="store_true", help="dev only: VmHWM is no longer per-trial")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    cfg = load_yaml(args.config)
    models = [m["name"] for m in load_yaml(args.models)["models"]]
    run_id = args.out.stem.removeprefix("run_")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # The manifest shares the run_id of this output file, so resumes reuse the first one.
    manifest_path = args.manifests / f"run_{run_id}.json"
    if not manifest_path.exists():
        try:
            manifest = run_preflight(args.config, args.models, args.doc, args.questions, run_id)
        except PreflightError as exc:
            log.error("preflight failed:\n%s", exc)
            return 1
        log.info("wrote %s", write_manifest(manifest, args.manifests))

    doc_text = args.doc.read_text(encoding="utf-8")
    prompt = build_prompt(doc_text)
    client = OllamaClient(cfg["ollama_url"], timeout_s=cfg["request_timeout_s"])
    digests = {full_tag(m): None for m in models}
    digests.update({name: info.get("digest") for name, info in client.tags().items()})

    base_row = {
        "run_id": run_id,
        "doc": args.doc.stem,
        "doc_sha256": sha256_file(args.doc),
    }
    completed = load_completed(args.out)
    total = cfg["n_trials"] * len(models)
    log.info("run %s: %d/%d trials already recorded", run_id, len(completed), total)

    for trial in range(1, cfg["n_trials"] + 1):
        for model in models:
            if (model, trial) in completed:
                continue
            digest = digests.get(full_tag(model))
            log.info("trial %d: %s starting", trial, model)
            row = run_trial(
                client, model, prompt, trial, cfg,
                {**base_row, "model_digest": f"sha256:{digest}" if digest else None},
                restart=not args.no_restart,
            )
            append_jsonl(args.out, row)
            if row["status"] == "ok":
                log.info(
                    "trial %d: %s ok wall=%.1fs gen_tps=%s peak_rss=%s GiB words=%d swap=%s",
                    trial, model, row["wall_s"], row["gen_tps"],
                    row["peak_rss_gb"], row["word_count"], row["swap_touched"],
                )
            else:
                log.error("trial %d: %s FAILED: %s", trial, model, row["error"])

    log.info("done: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
