"""Pre-flight checks; writes results/manifests/run_<id>.json when they all pass.

Usage: python src/_preflight.py [--run-id 20261107_1700]
Exit code 1 means a check failed and no manifest was written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil
import yaml

from _ollama import OllamaClient, build_prompt, context_length, full_tag

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ENV = {
    "OLLAMA_KEEP_ALIVE": "0",
    "OLLAMA_NUM_PARALLEL": "1",
    "OLLAMA_MAX_LOADED_MODELS": "1",
}


class PreflightError(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_questions(path: Path) -> list[str]:
    """Read the pre-generated question set: a JSON list of unique, non-empty strings."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise PreflightError(f"{path}: expected a non-empty JSON list of questions")
    if not all(isinstance(q, str) and q.strip() for q in data):
        raise PreflightError(f"{path}: every question must be a non-empty string")
    if len(set(data)) != len(data):
        raise PreflightError(f"{path}: duplicate questions")
    return data


def count_tokens(text: str) -> int:
    """Estimate only: candidates use their own tokenizers."""
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:  # tiktoken fetches its encoding over the network
        return int(len(text.split()) * 1.5)


def proc_meminfo_kb() -> dict[str, int]:
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
    except OSError:
        return {}
    return {k: int(v.split()[0]) for k, v in (line.split(":", 1) for line in lines)}


def vm_info() -> dict:
    cpu_model, avx512 = None, None
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        for line in cpuinfo.splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
        avx512 = "avx512f" in cpuinfo
    except OSError:
        cpu_model = platform.processor() or None
    cores = psutil.cpu_count(logical=True)
    ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
    try:
        os_name = platform.freedesktop_os_release().get("PRETTY_NAME")
    except OSError:
        os_name = platform.platform()
    return {
        "flavor_guess": "standard.xlarge" if cores == 8 and 28 <= ram_gb <= 33 else None,
        "cpu_model": cpu_model,
        "cores": cores,
        "ram_total_gb": ram_gb,
        "kernel": platform.release(),
        "os": os_name,
        "avx512": avx512,
    }


def ollama_service_env() -> dict[str, str] | None:
    # /proc/<pid>/environ is owner-only; systemd exposes the same data to any user.
    try:
        out = subprocess.run(
            ["systemctl", "show", "ollama", "--property=Environment", "--value"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    pairs = (item.split("=", 1) for item in shlex.split(out) if "=" in item)
    return {k: v for k, v in pairs if k.startswith("OLLAMA_")}


def run_preflight(config_path: Path, models_path: Path, doc_path: Path, questions_path: Path, run_id: str) -> dict:
    """Return the manifest, or raise PreflightError listing every failed check."""
    cfg = load_yaml(config_path)
    models = load_yaml(models_path)["models"]
    client = OllamaClient(cfg["ollama_url"])
    errors: list[str] = []
    warnings: list[str] = []

    # Swap
    meminfo = proc_meminfo_kb()
    if not meminfo:
        warnings.append("/proc/meminfo unavailable; swap check skipped (not on the Pouta VM?)")
    elif meminfo["SwapTotal"] != meminfo["SwapFree"]:
        errors.append(f"swap in use: {(meminfo['SwapTotal'] - meminfo['SwapFree']) // 1024} MiB")

    # Ollama service
    env = ollama_service_env()
    if env is None:
        warnings.append("could not read the ollama service environment")
    else:
        for key, want in EXPECTED_ENV.items():
            if env.get(key) != want:
                warnings.append(f"ollama service env {key}={env.get(key)!r}, expected {want!r}")

    try:
        version = client.version()
        installed = client.tags()
    except Exception as exc:
        raise PreflightError(f"Ollama unreachable at {cfg['ollama_url']}: {exc}") from exc

    # Models: resolve tag, check native context, record digest
    model_rows = []
    for model in models:
        name = model["name"]
        info = installed.get(full_tag(name))
        if info is None:
            errors.append(f"{name}: not pulled; fix the tag in models.yaml if the pull returned 404")
            continue
        shown = client.show(full_tag(name))
        ctx = context_length(shown)
        if ctx is None or ctx < cfg["num_ctx"]:
            errors.append(f"{name}: context_length {ctx} < num_ctx {cfg['num_ctx']}")
        if "thinking" in shown.get("capabilities", []) and "think" not in cfg:
            warnings.append(
                f"{name}: thinking model and bench.yaml has no `think` key; "
                f"reasoning tokens count against num_predict={cfg['num_predict']}"
            )
        model_rows.append({
            "name": name,
            "digest": info.get("digest"),
            "size_gb": round(info.get("size", 0) / 2**30, 2),
            "context_length": ctx,
        })

    # Document fits the context window, or Ollama truncates it silently
    doc_text = Path(doc_path).read_text(encoding="utf-8")
    prompt_tokens = count_tokens(build_prompt(doc_text))
    budget = cfg["num_ctx"] - cfg["num_predict"]
    if prompt_tokens > budget:
        errors.append(
            f"{Path(doc_path).name}: ~{prompt_tokens} prompt tokens exceeds {budget} "
            f"(num_ctx {cfg['num_ctx']} - num_predict {cfg['num_predict']}); Ollama would truncate it"
        )

    # Question set
    questions = load_questions(questions_path)
    if len(questions) != cfg["n_questions"]:
        errors.append(f"{len(questions)} questions, config expects {cfg['n_questions']}")

    for warning in warnings:
        print(f"[preflight] WARNING: {warning}")
    if errors:
        raise PreflightError("\n".join(f"  - {e}" for e in errors))

    return {
        "run_id": run_id,
        "vm": vm_info(),
        "ollama_version": version,
        "ollama_env": env,
        "models": model_rows,
        "bench_config": cfg,
        "doc": {"name": Path(doc_path).name, "sha256": sha256_file(doc_path), "token_count": prompt_tokens},
        "question_set": {"n": len(questions), "sha256": sha256_file(questions_path)},
    }


def write_manifest(manifest: dict, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"run_{manifest['run_id']}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/bench.yaml")
    parser.add_argument("--models", type=Path, default=ROOT / "config/models.yaml")
    parser.add_argument("--doc", type=Path, default=ROOT / "data/doc.txt")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/doc_questions.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/manifests")
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()

    try:
        manifest = run_preflight(args.config, args.models, args.doc, args.questions, args.run_id)
    except PreflightError as exc:
        print(f"[preflight] FAILED:\n{exc}", file=sys.stderr)
        return 1
    print(f"[preflight] OK: wrote {write_manifest(manifest, args.out_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
