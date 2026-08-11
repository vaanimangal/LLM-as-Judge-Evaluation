"""
Entry point for the LLM-as-Judge evaluation framework.

Loads configuration and environment variables, creates the judge LLM client,
runs the configured test suite, and writes auditable evaluation results.
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

from src.judge import Judge, JudgeError
from src.llm_client import LLMClient


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = PROJECT_ROOT / "data" / "test_suite.json"
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load YAML configuration and resolve ${ENV_VAR} placeholders."""

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw_config = file.read()

    config_text = os.path.expandvars(raw_config)

    unresolved = []

    for line_number, line in enumerate(config_text.splitlines(), start=1):
        if "${" in line:
            unresolved.append((line_number, line.strip()))

    if unresolved:
        details = "\n".join(
            f"Line {line_number}: {line}"
            for line_number, line in unresolved
        )

        raise RuntimeError(
            "One or more environment variables referenced in config.yaml "
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
# Test suite
# ---------------------------------------------------------------------------

def load_test_suite() -> Dict[str, Any]:
    """Load the evaluation test suite."""

    if not TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            f"Test suite not found: {TEST_SUITE_PATH}"
        )

    with TEST_SUITE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "test_suite.json must contain a JSON object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list):
        raise RuntimeError(
            "test_suite.json must contain a 'tests' list."
        )

    if not tests:
        raise RuntimeError(
            "test_suite.json contains no test cases."
        )

    return data


# ---------------------------------------------------------------------------
# Judge creation
# ---------------------------------------------------------------------------

def create_judge(config: Dict[str, Any]) -> Judge:
    """Create the configured judge LLM client."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing or invalid 'judge' configuration in config.yaml."
        )

    provider = judge_config.get("provider")

    if provider != "openai_compatible":
        raise RuntimeError(
            f"Unsupported judge provider: {provider!r}. "
            "Expected 'openai_compatible'."
        )

    model_value = judge_config.get("model")
    base_url_value = judge_config.get("base_url")

    if not model_value:
        raise RuntimeError(
            "Judge model is missing from config.yaml."
        )

    if not base_url_value:
        raise RuntimeError(
            "Judge base_url is missing from config.yaml."
        )

    model = str(model_value)
    base_url = str(base_url_value)

    api_key = get_required_environment_variable(
        "JUDGE_API_KEY"
    )

    evaluation_config = config.get("evaluation", {})

    if not isinstance(evaluation_config, dict):
        raise RuntimeError(
            "'evaluation' section in config.yaml must be an object."
        )

    temperature = float(
        evaluation_config.get("temperature", 0.0)
    )

    client = LLMClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    return Judge(client)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_test_suite(
    judge: Judge,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate every test case in the suite."""

    tests = test_suite["tests"]

    results = []

    successful = 0
    failed = 0

    started_at = datetime.now(timezone.utc).isoformat()

    for index, test_case in enumerate(tests, start=1):
        test_id = str(test_case.get("id", f"test_{index}"))

        print(
            f"[{index}/{len(tests)}] Evaluating {test_id}...",
            flush=True,
        )

        try:
            result = judge.evaluate(test_case)

            results.append(
                {
                    "status": "success",
                    "result": result,
                }
            )

            successful += 1

            print(
                f"    Score: {result['overall_score']:.2f}/100 "
                f"| Verdict: {result['verdict']}"
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
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(
                f"    FAILED: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    completed_at = datetime.now(timezone.utc).isoformat()

    successful_results = [
        item["result"]
        for item in results
        if item["status"] == "success"
    ]

    scores = [
        float(result["overall_score"])
        for result in successful_results
    ]

    pass_count = sum(
        1
        for result in successful_results
        if result["verdict"] == "PASS"
    )

    fail_count = sum(
        1
        for result in successful_results
        if result["verdict"] == "FAIL"
    )

    mean_score = (
        round(sum(scores) / len(scores), 2)
        if scores
        else None
    )

    pass_rate = (
        round((pass_count / len(successful_results)) * 100, 2)
        if successful_results
        else None
    )

    return {
        "experiment": {
            "name": "LLM-as-Judge evaluation",
            "mode": test_suite.get("mode", "pointwise"),
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_tests": len(tests),
            "successful": successful,
            "failed": failed,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "mean_score": mean_score,
            "pass_rate": pass_rate,
        },
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(results: Dict[str, Any]) -> Path:
    """Save evaluation results to the logs directory."""

    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = LOGS_DIR / f"evaluation_{timestamp}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def print_summary(results: Dict[str, Any]) -> None:
    """Print a concise evaluation summary."""

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("LLM-AS-JUDGE EVALUATION COMPLETE")
    print("=" * 60)

    print(f"Total tests : {summary['total_tests']}")
    print(f"Successful  : {summary['successful']}")
    print(f"Failed      : {summary['failed']}")
    print(f"PASS        : {summary['pass_count']}")
    print(f"FAIL        : {summary['fail_count']}")

    if summary["mean_score"] is not None:
        print(f"Mean score  : {summary['mean_score']:.2f}/100")

    if summary["pass_rate"] is not None:
        print(f"Pass rate   : {summary['pass_rate']:.2f}%")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the complete pointwise evaluation."""

    print("Loading configuration...")

    config = load_config()

    print("Loading test suite...")

    test_suite = load_test_suite()

    print("Creating judge...")

    judge = create_judge(config)

    print(
        f"Loaded {len(test_suite['tests'])} test case(s)."
    )

    print("\nStarting evaluation...\n")

    results = evaluate_test_suite(
        judge=judge,
        test_suite=test_suite,
    )

    output_path = save_results(results)

    print_summary(results)

    print(f"\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()