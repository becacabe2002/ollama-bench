"""Quality grading with DeepEval SummarizationMetric. Needs OPENAI_API_KEY.

Writes <out>/<run_id>.csv after every row, and re-running skips rows that already
have scores from the same question set, so a crash never loses paid judge calls.
--dry-run needs no API key.

Coverage comes from the pre-generated list in data/doc_questions.json. Those questions
are vetted against the document and ask what the summary says, so the judge answers
them for the summary only.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from pathlib import Path

import pandas as pd

from _preflight import ROOT, count_tokens, load_questions, sha256_file

# DeepEval's default 88 s per attempt is too short for truth extraction over a long doc.
os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "300")
os.environ.setdefault("DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE", "900")

DEFAULT_JUDGE = "gpt-5.4-mini-2026-03-17"
GRADE_COLUMNS = [
    "alignment",
    "coverage",
    "summary_f1",
    "judge_model",
    "judge_prompt_tokens",
    "judge_completion_tokens",
    "questions_sha256",
]

# Dry-run assumptions, per graded row.
JUDGE_CALLS_PER_ROW = 4  # truths, claims, summary answers, alignment
TEMPLATE_TOKENS = 500  # DeepEval prompt boilerplate per call
TRUTHS_OUTPUT_RATIO = 0.25  # extracted truths relative to the doc

log = logging.getLogger("grade")


def registry_alias(name: str) -> str:
    """DeepEval's model table lists `gpt-5.4-mini` but not dated snapshots of it."""
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA

    if name in OPENAI_MODELS_DATA:
        return name
    alias = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)
    return alias if alias in OPENAI_MODELS_DATA else name


def make_judge(name: str, temperature: float):
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA
    from deepeval.models.llms.openai_model import OpenAIModel

    judge = OpenAIModel(model=name, temperature=temperature)
    # An unlisted snapshot gets blank capabilities: no pricing, so no token counts, no
    # structured output, and a temperature parameter the API may reject.
    judge.model_data = OPENAI_MODELS_DATA.get(registry_alias(name))
    if judge.model_data.supports_temperature is False:
        log.warning(
            "%s does not accept a temperature; judge runs at the API default, not T=%s",
            name, temperature,
        )
    return judge


def f1(alignment: float, coverage: float) -> float:
    if alignment <= 0 or coverage <= 0:
        return 0.0
    return 2 * alignment * coverage / (alignment + coverage)


def summary_answers_prompt(questions: list[str], summary: str) -> str:
    """Stands in for DeepEval's answer prompt, which asks if a text can answer a question.

    Any text can answer "does the summary state X?", so that wording cannot separate a
    summary that states X from one that omits it.
    """
    return (
        "You are checking a summary against yes/no questions about what it says.\n"
        "Answer 'yes' only if the summary itself states what the question asks, "
        "including every part of a question that asks for several things. "
        "Answer 'no' if it omits any part or contradicts it.\n\n"
        "Return only JSON with the key 'answers': a list holding exactly one 'yes' or "
        "'no' per question, in the order given.\n\n"
        f"Summary:\n{summary}\n\n"
        f"Questions:\n{json.dumps(questions, ensure_ascii=False, indent=2)}\n\n"
        "JSON:\n"
    )


def make_metric(questions: list[str], judge):
    from deepeval.metrics import SummarizationMetric
    from deepeval.metrics.summarization.schema import Answers, SummarizationCoverageVerdict
    from deepeval.metrics.utils import a_generate_with_schema_and_extract

    metric = SummarizationMetric(
        threshold=None,
        model=judge,
        eval_mode="llm",
        assessment_questions=questions,
        include_reason=False,  # unused here; saves one judge call per row
    )

    async def coverage_verdicts(test_case):
        answers = await a_generate_with_schema_and_extract(
            metric=metric,
            prompt=summary_answers_prompt(questions, test_case.actual_output),
            schema_cls=Answers,
            extract_schema=lambda s: s.answers,
            extract_json=lambda data: data["answers"],
        )
        if len(answers) != len(questions):
            raise ValueError(f"{len(answers)} answers for {len(questions)} questions")
        # DeepEval's score counts questions the document answers: all of ours.
        return [
            SummarizationCoverageVerdict(summary_verdict=answer, original_verdict="yes", question=question)
            for answer, question in zip(answers, questions)
        ]

    # Patched on the instance because DeepEval finds its prompts by class name, which
    # rules out a subclass. Private API: requirements.txt pins deepeval.
    metric._a_generate_coverage_verdicts = coverage_verdicts
    return metric


def grade_row(row: dict, doc_text: str, questions: list[str], judge) -> dict:
    from deepeval.metrics.summarization.schema import ScoreType
    from deepeval.test_case import LLMTestCase

    metric = make_metric(questions, judge)
    metric.measure(LLMTestCase(input=doc_text, actual_output=row["response_text"]), _show_indicator=False)

    breakdown = metric.score_breakdown
    keys = (ScoreType.ALIGNMENT.value, ScoreType.COVERAGE.value)
    missing = [k for k in keys if k not in breakdown]
    if missing:
        raise KeyError(f"score_breakdown is missing {missing}; got {list(breakdown)}")
    alignment, coverage = breakdown[keys[0]], breakdown[keys[1]]
    return {
        "alignment": alignment,
        "coverage": coverage,
        "summary_f1": f1(alignment, coverage),
        "judge_model": judge.get_model_name(),
        "judge_prompt_tokens": metric.input_tokens,
        "judge_completion_tokens": metric.output_tokens,
    }


def estimate_row_tokens(summary: str, doc_tokens: int, question_tokens: int) -> tuple[int, int]:
    """Rough (input, output) tokens for truths, claims, summary answers, alignment."""
    summary_tokens = count_tokens(summary)
    truths_tokens = int(doc_tokens * TRUTHS_OUTPUT_RATIO)
    input_tokens = (
        JUDGE_CALLS_PER_ROW * TEMPLATE_TOKENS
        + doc_tokens  # truths
        + summary_tokens  # claims
        + summary_tokens + question_tokens  # answers from the summary
        + summary_tokens + truths_tokens  # alignment
    )
    output_tokens = truths_tokens + summary_tokens + question_tokens + 2 * summary_tokens
    return input_tokens, output_tokens


def dry_run(pending_rows: list[dict], doc_text: str, questions: list[str], judge_model: str) -> None:
    from deepeval.models.llms.constants import OPENAI_MODELS_DATA

    doc_tokens = count_tokens(doc_text)
    question_tokens = count_tokens("\n".join(questions))
    estimates = [estimate_row_tokens(r["response_text"], doc_tokens, question_tokens) for r in pending_rows]
    total_in = sum(e[0] for e in estimates)
    total_out = sum(e[1] for e in estimates)
    print(f"rows to grade:         {len(pending_rows)}")
    print(f"judge calls (approx):  {JUDGE_CALLS_PER_ROW * len(pending_rows)}")
    print(f"input tokens (approx): {total_in:,}")
    print(f"output tokens (approx):{total_out:,}")
    data = OPENAI_MODELS_DATA.get(registry_alias(judge_model))
    if data.input_price and data.output_price:
        cost = total_in * data.input_price + total_out * data.output_price
        print(f"estimated cost:        ${cost:.2f} ({judge_model})")
    else:
        print(f"estimated cost:        unknown, DeepEval has no pricing for {judge_model}")


def load_graded(path: Path, questions_sha: str) -> dict[tuple[str, int], dict]:
    """Earlier scores from this question set; any others count as pending."""
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if "questions_sha256" not in df:
        df["questions_sha256"] = None
    scored = df[df["alignment"].notna()]
    current = scored[scored["questions_sha256"] == questions_sha]
    if len(current) < len(scored):
        log.warning("%s: %d scores used another question set; re-grading them", path, len(scored) - len(current))
    return {
        (r["model"], int(r["trial"])): {c: r[c] for c in GRADE_COLUMNS}
        for r in current.to_dict("records")
    }


def write_csv(rows: list[dict], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def csv_row(row: dict, grades: dict | None) -> dict:
    flat = {**row, "config": json.dumps(row.get("config"))}
    return {**flat, **{c: (grades or {}).get(c) for c in GRADE_COLUMNS}}


def read_raw(path: Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", required=True, help="JSONL files; globs allowed")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/doc_questions.json")
    parser.add_argument("--doc", type=Path, default=ROOT / "data/doc.txt")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=ROOT / "results/graded")
    parser.add_argument("--dry-run", action="store_true", help="estimate token spend, call nothing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    doc_text = args.doc.read_text(encoding="utf-8")
    doc_sha = sha256_file(args.doc)
    questions = load_questions(args.questions)
    questions_sha = sha256_file(args.questions)
    raw_paths = [Path(p) for pattern in args.raw for p in (glob.glob(pattern) or [pattern])]
    args.out.mkdir(parents=True, exist_ok=True)

    judge = None
    failures = 0
    all_pending: list[dict] = []
    for raw_path in raw_paths:
        rows = read_raw(raw_path)
        if not rows:
            log.warning("%s is empty; skipping", raw_path)
            continue
        out_path = args.out / f"{rows[0]['run_id']}.csv"
        previous = load_graded(out_path, questions_sha)
        pending = [r for r in rows if r["status"] == "ok" and (r["model"], r["trial"]) not in previous]
        all_pending.extend(pending)
        if args.dry_run:
            continue

        output = [csv_row(r, previous.get((r["model"], r["trial"]))) for r in rows]
        index = {(r["model"], r["trial"]): i for i, r in enumerate(rows)}
        write_csv(output, out_path)
        for row in pending:
            if row["doc_sha256"] != doc_sha:
                log.error("%s trial %s: doc hash differs from %s; not graded", row["model"], row["trial"], args.doc)
                failures += 1
                continue
            judge = judge or make_judge(args.judge_model, args.judge_temperature)
            key = (row["model"], row["trial"])
            try:
                grades = {**grade_row(row, doc_text, questions, judge), "questions_sha256": questions_sha}
            except Exception as exc:
                log.error("%s trial %s: judge failed: %s: %s", *key, type(exc).__name__, exc)
                failures += 1
                continue
            output[index[key]] = csv_row(row, grades)
            write_csv(output, out_path)
            log.info(
                "%s trial %s: alignment=%.3f coverage=%.3f f1=%.3f",
                *key, grades["alignment"], grades["coverage"], grades["summary_f1"],
            )
        log.info("wrote %s", out_path)

    if args.dry_run:
        dry_run(all_pending, doc_text, questions, args.judge_model)
        return 0
    if failures:
        log.error("%d rows not graded; re-run to retry them", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
