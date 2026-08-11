"""
Entry point for pairwise A/B LLM-as-Judge evaluation.

Loads the existing configuration and environment variables,
loads the pairwise test suite, evaluates Candidate A vs Candidate B,
and saves auditable pairwise results.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.pairwise_judge import PairwiseJudge, PairwiseJudgeError


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
PAIRWISE_TEST_SUITE_PATH = (
    PROJECT_ROOT / "data" / "pairwise_test_suite.json"
)
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load YAML configuration and resolve environment variables."""

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
            "Environment variables referenced in config.yaml "
            "are not set:\n"
            + details
        )

    config = yaml.safe_load(config_text)

    if not isinstance(config, dict):
        raise RuntimeError(
            "config.yaml must contain a YAML object."
        )

    return config


def get_required_environment_variable(name: str) -> str:
    """Return a required environment variable."""

    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )

    return value.strip()


# ---------------------------------------------------------------------------
# Pairwise test suite
# ---------------------------------------------------------------------------

def load_pairwise_test_suite() -> Dict[str, Any]:
    """Load and validate the pairwise test suite."""

    if not PAIRWISE_TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            "Pairwise test suite not found: "
            f"{PAIRWISE_TEST_SUITE_PATH}"
        )

    with PAIRWISE_TEST_SUITE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "pairwise_test_suite.json must contain a JSON object."
        )

    if data.get("mode") != "pairwise":
        raise RuntimeError(
            "pairwise_test_suite.json must have "
            "'mode': 'pairwise'."
        )

    tests = data.get("tests")

    if not isinstance(tests, list):
        raise RuntimeError(
            "pairwise_test_suite.json must contain a 'tests' list."
        )

    if not tests:
        raise RuntimeError(
            "pairwise_test_suite.json contains no test cases."
        )

    required_fields = {
        "id",
        "input",
        "system_prompt",
        "candidate_a",
        "candidate_b",
    }

    for index, test_case in enumerate(tests, start=1):
        if not isinstance(test_case, dict):
            raise RuntimeError(
                f"Pairwise test #{index} must be an object."
            )

        missing = required_fields - set(test_case.keys())

        if missing:
            raise RuntimeError(
                f"Pairwise test #{index} is missing fields: "
                + ", ".join(sorted(missing))
            )

    return data


# ---------------------------------------------------------------------------
# Pairwise judge creation
# ---------------------------------------------------------------------------

def create_pairwise_judge(
    config: Dict[str, Any],
) -> PairwiseJudge:
    """Create the pairwise judge using the existing LLM client."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing or invalid 'judge' configuration."
        )

    provider = judge_config.get("provider")

    if provider != "openai_compatible":
        raise RuntimeError(
            f"Unsupported judge provider: {provider!r}. "
            "Expected 'openai_compatible'."
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

    api_key = get_required_environment_variable(
        "JUDGE_API_KEY"
    )

    evaluation_config = config.get(
        "evaluation",
        {},
    )

    if not isinstance(evaluation_config, dict):
        raise RuntimeError(
            "'evaluation' section must be an object."
        )

    temperature = float(
        evaluation_config.get(
            "temperature",
            0.0,
        )
    )

    client = LLMClient(
        api_key=api_key,
        model=str(model),
        base_url=str(base_url),
        temperature=temperature,
    )

    return PairwiseJudge(
        client=client,
        tie_margin=0.25,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_pairwise_suite(
    judge: PairwiseJudge,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate every A/B pair."""

    tests = test_suite["tests"]

    results = []

    successful = 0
    failed = 0

    started_at = datetime.now(
        timezone.utc
    ).isoformat()

    for index, test_case in enumerate(
        tests,
        start=1,
    ):
        test_id = str(
            test_case.get(
                "id",
                f"pair_{index}",
            )
        )

        print(
            f"[{index}/{len(tests)}] "
            f"Comparing {test_id}...",
            flush=True,
        )

        try:
            result = judge.evaluate(
                test_id=test_id,
                user_input=str(
                    test_case["input"]
                ),
                system_prompt=str(
                    test_case["system_prompt"]
                ),
                candidate_a=str(
                    test_case["candidate_a"]
                ),
                candidate_b=str(
                    test_case["candidate_b"]
                ),
                expected_output=str(
                    test_case.get(
                        "expected_output",
                        "",
                    )
                ),
            )

            results.append(
                {
                    "status": "success",
                    "result": result,
                }
            )

            successful += 1

            print(
                f"    A: "
                f"{result['candidate_a']['average_score']:.2f}"
                f"/5"
            )

            print(
                f"    B: "
                f"{result['candidate_b']['average_score']:.2f}"
                f"/5"
            )

            print(
                f"    Winner: {result['winner']}"
            )

        except PairwiseJudgeError as exc:
            failed += 1

            results.append(
                {
                    "status": "failed",
                    "test_id": test_id,
                    "error": str(exc),
                }
            )

            print(
                f"    FAILED: {exc}",
                file=sys.stderr,
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
                f"    FAILED: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    completed_at = datetime.now(
        timezone.utc
    ).isoformat()

    successful_results = [
        item["result"]
        for item in results
        if item["status"] == "success"
    ]

    a_wins = sum(
        1
        for result in successful_results
        if result["winner"] == "A"
    )

    b_wins = sum(
        1
        for result in successful_results
        if result["winner"] == "B"
    )

    ties = sum(
        1
        for result in successful_results
        if result["winner"] == "TIE"
    )

    total_successful = len(
        successful_results
    )

    a_win_rate = (
        round(
            (a_wins / total_successful) * 100,
            2,
        )
        if total_successful
        else None
    )

    b_win_rate = (
        round(
            (b_wins / total_successful) * 100,
            2,
        )
        if total_successful
        else None
    )

    tie_rate = (
        round(
            (ties / total_successful) * 100,
            2,
        )
        if total_successful
        else None
    )

    if a_wins > b_wins:
        overall_winner = "A"
    elif b_wins > a_wins:
        overall_winner = "B"
    else:
        overall_winner = "TIE"

    return {
        "experiment": {
            "name": "Pairwise A/B LLM-as-Judge evaluation",
            "mode": "pairwise",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_pairs": len(tests),
            "successful": successful,
            "failed": failed,
            "a_wins": a_wins,
            "b_wins": b_wins,
            "ties": ties,
            "a_win_rate": a_win_rate,
            "b_win_rate": b_win_rate,
            "tie_rate": tie_rate,
            "overall_winner": overall_winner,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, Any],
) -> Path:
    """Save pairwise results to logs."""

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        LOGS_DIR
        / f"pairwise_evaluation_{timestamp}.json"
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


def print_summary(
    results: Dict[str, Any],
) -> None:
    """Print pairwise evaluation summary."""

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("PAIRWISE LLM-AS-JUDGE EVALUATION COMPLETE")
    print("=" * 60)

    print(
        f"Total pairs : "
        f"{summary['total_pairs']}"
    )

    print(
        f"Successful  : "
        f"{summary['successful']}"
    )

    print(
        f"Failed      : "
        f"{summary['failed']}"
    )

    print(
        f"A wins      : "
        f"{summary['a_wins']}"
    )

    print(
        f"B wins      : "
        f"{summary['b_wins']}"
    )

    print(
        f"Ties        : "
        f"{summary['ties']}"
    )

    if summary["a_win_rate"] is not None:
        print(
            f"A win rate  : "
            f"{summary['a_win_rate']:.2f}%"
        )

    if summary["b_win_rate"] is not None:
        print(
            f"B win rate  : "
            f"{summary['b_win_rate']:.2f}%"
        )

    if summary["tie_rate"] is not None:
        print(
            f"Tie rate    : "
            f"{summary['tie_rate']:.2f}%"
        )

    print(
        f"Winner      : "
        f"{summary['overall_winner']}"
    )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete pairwise evaluation."""

    print("Loading configuration...")

    config = load_config()

    print("Loading pairwise test suite...")

    test_suite = load_pairwise_test_suite()

    print("Creating pairwise judge...")

    judge = create_pairwise_judge(config)

    print(
        f"Loaded "
        f"{len(test_suite['tests'])} pair(s)."
    )

    print("\nStarting pairwise evaluation...\n")

    results = evaluate_pairwise_suite(
        judge=judge,
        test_suite=test_suite,
    )

    output_path = save_results(results)

    print_summary(results)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()