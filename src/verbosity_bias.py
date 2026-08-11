"""
Verbosity-bias experiment for LLM-as-Judge evaluation.

The experiment tests whether the judge systematically prefers a verbose
answer over a concise answer when both answers are substantively correct.

Each test is evaluated twice:

1. Concise answer in position A, verbose answer in position B.
2. Verbose answer in position A, concise answer in position B.

The experiment records:
- winner in each ordering
- whether the judge prefers the verbose answer
- whether the result is consistent across position swaps
- aggregate verbosity preference
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv

from src.llm_client import LLMClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = PROJECT_ROOT / "data" / "verbosity_bias_test_suite.json"
LOGS_DIR = PROJECT_ROOT / "logs"


load_dotenv(PROJECT_ROOT / ".env")


VERBOSITY_JUDGE_SYSTEM_PROMPT = """
You are a strict and impartial evaluator testing another LLM judge for
verbosity bias.

You will compare two candidate answers to the same task.

Your job is NOT to reward an answer merely because it is longer.

Evaluation principles:

1. Judge correctness first.
2. Judge relevance to the task.
3. Follow the system prompt exactly.
4. Do not reward unnecessary detail.
5. If the system prompt requests conciseness, unnecessary verbosity is a
   negative factor.
6. If both answers are substantively equivalent, do not prefer the longer
   answer merely because it contains more words.
7. Do not use answer position as a quality signal.
8. Select the candidate that better satisfies the actual task.
9. Return ONLY valid JSON.
""".strip()


def load_config() -> Dict[str, Any]:
    """Load config.yaml with environment variable expansion."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw_config = file.read()

    config_text = os.path.expandvars(raw_config)

    unresolved = []

    for line_number, line in enumerate(
        config_text.splitlines(),
        start=1,
    ):
        if "${" in line:
            unresolved.append(
                (line_number, line.strip())
            )

    if unresolved:
        details = "\n".join(
            f"Line {line_number}: {line}"
            for line_number, line in unresolved
        )

        raise RuntimeError(
            "Unresolved environment variables in config.yaml:\n"
            + details
        )

    config = yaml.safe_load(config_text)

    if not isinstance(config, dict):
        raise RuntimeError(
            "config.yaml must contain a YAML object."
        )

    return config


def load_test_suite() -> Dict[str, Any]:
    """Load verbosity-bias test cases."""

    if not TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            f"Test suite not found: {TEST_SUITE_PATH}"
        )

    with TEST_SUITE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "verbosity_bias_test_suite.json must contain an object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list) or not tests:
        raise RuntimeError(
            "Verbosity-bias test suite must contain a non-empty 'tests' list."
        )

    return data


def create_client(config: Dict[str, Any]) -> LLMClient:
    """Create the configured judge LLM client."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing or invalid 'judge' configuration."
        )

    provider = judge_config.get("provider")

    if provider != "openai_compatible":
        raise RuntimeError(
            f"Unsupported judge provider: {provider!r}."
        )

    model = judge_config.get("model")
    base_url = judge_config.get("base_url")

    if not model:
        raise RuntimeError(
            "Judge model is missing from config.yaml."
        )

    if not base_url:
        raise RuntimeError(
            "Judge base_url is missing from config.yaml."
        )

    api_key = os.getenv("JUDGE_API_KEY")

    if not api_key or not api_key.strip():
        raise RuntimeError(
            "JUDGE_API_KEY is not configured."
        )

    evaluation_config = config.get("evaluation", {})

    temperature = float(
        evaluation_config.get("temperature", 0.0)
    )

    return LLMClient(
        api_key=api_key.strip(),
        model=str(model),
        base_url=str(base_url),
        temperature=temperature,
    )


def build_prompt(
    test_case: Dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> str:
    """Build a controlled pairwise comparison prompt."""

    return f"""
Compare Candidate A and Candidate B for the task below.

TASK:
{test_case["input"]}

SYSTEM PROMPT:
{test_case["system_prompt"]}

EXPECTED OUTPUT:
{test_case["expected_output"]}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

Evaluate using these priorities:

1. Correctness.
2. Relevance.
3. Instruction-following.
4. Completeness.
5. Conciseness when requested by the system prompt.

Important:
- Do NOT prefer a response simply because it is longer.
- Do NOT prefer a response simply because it contains more information.
- If both answers are equally correct and satisfy the task, treat them as
  equal in quality unless the task explicitly rewards additional detail.
- Follow the system prompt.
- Select the better answer based on task requirements.

Return ONLY this JSON:

{{
    "winner": "A",
    "reason": "Brief evidence-based explanation."
}}

The winner must be exactly "A", "B", or "TIE".
""".strip()


def parse_verdict(text: str) -> Dict[str, Any]:
    """Parse the judge JSON response."""

    cleaned = text.strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()

        if len(lines) >= 3:
            lines = lines[1:-1]

        cleaned = "\n".join(lines).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Judge returned invalid JSON: {text[:1000]}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Judge response must be a JSON object."
        )

    winner = data.get("winner")

    if winner not in {"A", "B", "TIE"}:
        raise RuntimeError(
            f"Invalid winner returned by judge: {winner!r}"
        )

    reason = data.get("reason")

    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError(
            "Judge response must contain a non-empty reason."
        )

    return {
        "winner": winner,
        "reason": reason.strip(),
    }


def run_comparison(
    client: LLMClient,
    test_case: Dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> Dict[str, Any]:
    """Run one pairwise comparison."""

    prompt = build_prompt(
        test_case,
        candidate_a,
        candidate_b,
    )

    response = client.generate(
        prompt,
        system_prompt=VERBOSITY_JUDGE_SYSTEM_PROMPT,
    )

    verdict = parse_verdict(response.text)

    return {
        "winner": verdict["winner"],
        "reason": verdict["reason"],
        "usage": {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "latency_ms": response.latency_ms,
        },
    }


def determine_verbosity_preference(
    *,
    normal_winner: str,
    swapped_winner: str,
) -> Tuple[str, bool]:
    """
    Determine whether the judge consistently prefers verbosity.

    Normal ordering:
        A = concise
        B = verbose

    Swapped ordering:
        A = verbose
        B = concise

    Therefore:
        normal B + swapped A -> verbose preference
        normal A + swapped B -> concise preference
    """

    if (
        normal_winner == "B"
        and swapped_winner == "A"
    ):
        return "verbose", True

    if (
        normal_winner == "A"
        and swapped_winner == "B"
    ):
        return "concise", True

    if normal_winner == "TIE" and swapped_winner == "TIE":
        return "neutral", True

    return "inconclusive", False


def run_experiment(
    client: LLMClient,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the complete verbosity-bias experiment."""

    tests = test_suite["tests"]

    results = []

    successful = 0
    failed = 0

    verbose_preference_count = 0
    concise_preference_count = 0
    neutral_count = 0
    inconclusive_count = 0

    started_at = datetime.now(timezone.utc).isoformat()

    for index, test_case in enumerate(tests, start=1):

        test_id = str(
            test_case.get(
                "id",
                f"verb_{index}",
            )
        )

        print(
            f"[{index}/{len(tests)}] Testing {test_id}...",
            flush=True,
        )

        try:
            concise = str(
                test_case["concise_answer"]
            )

            verbose = str(
                test_case["verbose_answer"]
            )

            # ----------------------------------------------------------
            # Ordering 1:
            # A = concise
            # B = verbose
            # ----------------------------------------------------------

            normal = run_comparison(
                client,
                test_case,
                concise,
                verbose,
            )

            # ----------------------------------------------------------
            # Ordering 2:
            # A = verbose
            # B = concise
            # ----------------------------------------------------------

            swapped = run_comparison(
                client,
                test_case,
                verbose,
                concise,
            )

            preference, consistent = (
                determine_verbosity_preference(
                    normal_winner=normal["winner"],
                    swapped_winner=swapped["winner"],
                )
            )

            if preference == "verbose":
                verbose_preference_count += 1

            elif preference == "concise":
                concise_preference_count += 1

            elif preference == "neutral":
                neutral_count += 1

            else:
                inconclusive_count += 1

            successful += 1

            results.append(
                {
                    "status": "success",
                    "test_id": test_id,
                    "normal_order": {
                        "candidate_a": "concise",
                        "candidate_b": "verbose",
                        "winner": normal["winner"],
                        "reason": normal["reason"],
                    },
                    "swapped_order": {
                        "candidate_a": "verbose",
                        "candidate_b": "concise",
                        "winner": swapped["winner"],
                        "reason": swapped["reason"],
                    },
                    "verbosity_preference": preference,
                    "consistent": consistent,
                    "usage": {
                        "normal": normal["usage"],
                        "swapped": swapped["usage"],
                    },
                }
            )

            print(
                f"    Normal: {normal['winner']} "
                f"| Swapped: {swapped['winner']} "
                f"| Preference: {preference}"
            )

        except Exception as exc:
            failed += 1

            results.append(
                {
                    "status": "failed",
                    "test_id": test_id,
                    "error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )

            print(
                f"    FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    completed_at = datetime.now(timezone.utc).isoformat()

    consistency_count = sum(
        1
        for result in results
        if (
            result["status"] == "success"
            and result["consistent"]
        )
    )

    consistency_rate = (
        round(
            (consistency_count / successful) * 100,
            2,
        )
        if successful
        else None
    )

    return {
        "experiment": {
            "name": "LLM-as-Judge verbosity-bias experiment",
            "mode": "verbosity_bias",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_tests": len(tests),
            "successful": successful,
            "failed": failed,
            "verbose_preference": verbose_preference_count,
            "concise_preference": concise_preference_count,
            "neutral": neutral_count,
            "inconclusive": inconclusive_count,
            "consistent": consistency_count,
            "consistency_rate": consistency_rate,
        },
        "results": results,
    }


def save_results(results: Dict[str, Any]) -> Path:
    """Save experiment results."""

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = (
        LOGS_DIR
        / f"verbosity_bias_{timestamp}.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def print_summary(results: Dict[str, Any]) -> None:
    """Print experiment summary."""

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("VERBOSITY-BIAS EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Total tests          : "
        f"{summary['total_tests']}"
    )

    print(
        f"Successful           : "
        f"{summary['successful']}"
    )

    print(
        f"Failed               : "
        f"{summary['failed']}"
    )

    print(
        f"Verbose preference   : "
        f"{summary['verbose_preference']}"
    )

    print(
        f"Concise preference   : "
        f"{summary['concise_preference']}"
    )

    print(
        f"Neutral              : "
        f"{summary['neutral']}"
    )

    print(
        f"Inconclusive         : "
        f"{summary['inconclusive']}"
    )

    print(
        f"Consistent           : "
        f"{summary['consistent']}"
    )

    if summary["consistency_rate"] is not None:
        print(
            f"Consistency rate     : "
            f"{summary['consistency_rate']:.2f}%"
        )

    print("=" * 60)


def main() -> None:
    """Run the verbosity-bias experiment."""

    print("Loading configuration...")

    config = load_config()

    print("Loading verbosity-bias test suite...")

    test_suite = load_test_suite()

    print("Creating judge client...")

    client = create_client(config)

    print(
        f"Loaded {len(test_suite['tests'])} "
        "verbosity-bias test(s)."
    )

    print("\nStarting verbosity-bias experiment...\n")

    results = run_experiment(
        client,
        test_suite,
    )

    output_path = save_results(results)

    print_summary(results)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()