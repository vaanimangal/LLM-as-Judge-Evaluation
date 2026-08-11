"""
Gold-label validation for the LLM-as-Judge.

Validates whether the judge agrees with independently defined
gold verdicts and score expectations.

This experiment uses a small number of API calls because it is
intended as a validation experiment rather than a large benchmark.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from src.judge import Judge, JudgeError
from src.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = (
    PROJECT_ROOT / "data" / "judge_validation_test_suite.json"
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
    """Load YAML configuration."""

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


def get_required_environment_variable(name: str) -> str:
    """Get a required environment variable."""

    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"Required environment variable '{name}' is not set."
        )

    return value.strip()


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

def load_test_suite() -> Dict[str, Any]:
    """Load gold-label validation cases."""

    if not TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            f"Validation test suite not found: {TEST_SUITE_PATH}"
        )

    with TEST_SUITE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Validation test suite must contain a JSON object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list) or not tests:
        raise RuntimeError(
            "Validation test suite must contain a non-empty 'tests' list."
        )

    return data


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def create_judge(config: Dict[str, Any]) -> Judge:
    """Create the configured judge."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing or invalid 'judge' configuration."
        )

    if judge_config.get("provider") != "openai_compatible":
        raise RuntimeError(
            "Unsupported judge provider."
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

    evaluation_config = config.get("evaluation", {})

    temperature = float(
        evaluation_config.get("temperature", 0.0)
    )

    client = LLMClient(
        api_key=api_key,
        model=str(model),
        base_url=str(base_url),
        temperature=temperature,
    )

    return Judge(client)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_result(
    result: Dict[str, Any],
    test_case: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare judge output with gold labels."""

    actual_verdict = result["verdict"]
    actual_score = float(result["overall_score"])

    gold_verdict = test_case["gold_verdict"]

    verdict_match = (
        actual_verdict == gold_verdict
    )

    score_match = True

    if "gold_min_score" in test_case:
        score_match = (
            actual_score >= float(
                test_case["gold_min_score"]
            )
        )

    if "gold_max_score" in test_case:
        score_match = score_match and (
            actual_score <= float(
                test_case["gold_max_score"]
            )
        )

    agreement = (
        verdict_match and score_match
    )

    return {
        "gold_verdict": gold_verdict,
        "judge_verdict": actual_verdict,
        "gold_score_requirement": {
            "min": test_case.get("gold_min_score"),
            "max": test_case.get("gold_max_score"),
        },
        "judge_score": actual_score,
        "verdict_match": verdict_match,
        "score_match": score_match,
        "agreement": agreement,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_validation(
    judge: Judge,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Run gold-label validation."""

    tests = test_suite["tests"]

    results: List[Dict[str, Any]] = []

    successful = 0
    failed = 0
    agreements = 0

    started_at = datetime.now(timezone.utc).isoformat()

    for index, test_case in enumerate(
        tests,
        start=1,
    ):
        test_id = str(
            test_case.get(
                "id",
                f"validation_{index}",
            )
        )

        print(
            f"[{index}/{len(tests)}] "
            f"Validating {test_id}...",
            flush=True,
        )

        try:
            judge_result = judge.evaluate(
                test_case
            )

            validation = validate_result(
                judge_result,
                test_case,
            )

            if validation["agreement"]:
                agreements += 1

            successful += 1

            results.append(
                {
                    "status": "success",
                    "test_id": test_id,
                    "validation": validation,
                    "judge_result": judge_result,
                }
            )

            print(
                f"    Judge: "
                f"{validation['judge_verdict']} "
                f"{validation['judge_score']:.2f}/100 "
                f"| Gold: "
                f"{validation['gold_verdict']} "
                f"| Agreement: "
                f"{'YES' if validation['agreement'] else 'NO'}",
                flush=True,
            )

        except JudgeError as exc:

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

    completed_at = datetime.now(timezone.utc).isoformat()

    agreement_rate = (
        round(
            (agreements / successful) * 100,
            2,
        )
        if successful
        else None
    )

    return {
        "experiment": {
            "name": "Gold-label judge validation",
            "mode": "judge_validation",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_tests": len(tests),
            "successful": successful,
            "failed": failed,
            "agreements": agreements,
            "disagreements": (
                successful - agreements
            ),
            "agreement_rate": agreement_rate,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, Any],
) -> Path:
    """Save validation results."""

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = (
        LOGS_DIR
        / f"judge_validation_{timestamp}.json"
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
    """Print validation summary."""

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("JUDGE VALIDATION COMPLETE")
    print("=" * 60)

    print(
        f"Total tests       : "
        f"{summary['total_tests']}"
    )

    print(
        f"Successful        : "
        f"{summary['successful']}"
    )

    print(
        f"Failed            : "
        f"{summary['failed']}"
    )

    print(
        f"Agreements        : "
        f"{summary['agreements']}"
    )

    print(
        f"Disagreements     : "
        f"{summary['disagreements']}"
    )

    if summary["agreement_rate"] is not None:
        print(
            f"Agreement rate    : "
            f"{summary['agreement_rate']:.2f}%"
        )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run gold-label judge validation."""

    print("Loading configuration...")

    config = load_config()

    print("Loading validation test suite...")

    test_suite = load_test_suite()

    print("Creating judge client...")

    judge = create_judge(config)

    print(
        f"Loaded {len(test_suite['tests'])} "
        f"validation test(s)."
    )

    print("\nStarting judge validation...\n")

    results = run_validation(
        judge=judge,
        test_suite=test_suite,
    )

    output_path = save_results(results)

    print_summary(results)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()