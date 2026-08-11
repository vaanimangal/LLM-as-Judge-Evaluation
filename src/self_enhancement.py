"""
Self-enhancement bias experiment for LLM-as-Judge.

Tests whether self-promotional language causes the judge to assign
higher scores when the underlying answer quality is essentially the same.
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

from src.llm_client import LLMClient
from src.judge import Judge, JudgeError


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = (
    PROJECT_ROOT / "data" / "self_enhancement_test_suite.json"
)
LOGS_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


def load_config() -> Dict[str, Any]:
    """Load YAML configuration."""

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        raw_config = file.read()

    config_text = os.path.expandvars(raw_config)

    if "${" in config_text:
        raise RuntimeError(
            "Unresolved environment variable found in config.yaml."
        )

    config = yaml.safe_load(config_text)

    if not isinstance(config, dict):
        raise RuntimeError(
            "config.yaml must contain a YAML object."
        )

    return config


def get_api_key() -> str:
    """Load the judge API key."""

    value = os.getenv("JUDGE_API_KEY")

    if not value or not value.strip():
        raise RuntimeError(
            "JUDGE_API_KEY is not set."
        )

    return value.strip()


def load_test_suite() -> Dict[str, Any]:
    """Load self-enhancement test cases."""

    with TEST_SUITE_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "Self-enhancement test suite must be an object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list) or not tests:
        raise RuntimeError(
            "Self-enhancement test suite must contain tests."
        )

    return data


def create_judge(config: Dict[str, Any]) -> Judge:
    """Create the existing configured judge."""

    judge_config = config.get("judge", {})
    evaluation_config = config.get("evaluation", {})

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

    client = LLMClient(
        api_key=get_api_key(),
        model=str(model),
        base_url=str(base_url),
        temperature=float(
            evaluation_config.get("temperature", 0.0)
        ),
    )

    return Judge(client)


def build_test_case(
    original: Dict[str, Any],
    candidate: str,
) -> Dict[str, Any]:
    """Convert an experiment case into the existing Judge format."""

    return {
        "id": original["id"],
        "input": original["input"],
        "system_prompt": original["system_prompt"],
        "model_output": candidate,
        "expected_output": original["expected_output"],
        "criteria": original["criteria"],
    }


def run_experiment(
    judge: Judge,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Run neutral and self-enhanced evaluations."""

    results: List[Dict[str, Any]] = []

    neutral_scores: List[float] = []
    enhanced_scores: List[float] = []

    started_at = datetime.now(timezone.utc).isoformat()

    tests = test_suite["tests"]

    for index, test in enumerate(tests, start=1):

        test_id = str(test["id"])

        print(
            f"[{index}/{len(tests)}] Testing {test_id}...",
            flush=True,
        )

        try:
            neutral_case = build_test_case(
                test,
                test["candidate_neutral"],
            )

            enhanced_case = build_test_case(
                test,
                test["candidate_self_enhanced"],
            )

            neutral_result = judge.evaluate(
                neutral_case
            )

            enhanced_result = judge.evaluate(
                enhanced_case
            )

            neutral_score = float(
                neutral_result["overall_score"]
            )

            enhanced_score = float(
                enhanced_result["overall_score"]
            )

            difference = round(
                enhanced_score - neutral_score,
                4,
            )

            neutral_scores.append(
                neutral_score
            )

            enhanced_scores.append(
                enhanced_score
            )

            if difference > 0:
                direction = "SELF_ENHANCED_HIGHER"
            elif difference < 0:
                direction = "NEUTRAL_HIGHER"
            else:
                direction = "NO_DIFFERENCE"

            results.append(
                {
                    "status": "success",
                    "test_id": test_id,
                    "neutral_score": neutral_score,
                    "self_enhanced_score": enhanced_score,
                    "score_difference": difference,
                    "direction": direction,
                }
            )

            print(
                f"    Neutral: {neutral_score:.2f}"
                f" | Self-enhanced: {enhanced_score:.2f}"
                f" | Difference: {difference:+.2f}"
            )

        except JudgeError as exc:

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

    successful = [
        item
        for item in results
        if item["status"] == "success"
    ]

    enhanced_higher = sum(
        1
        for item in successful
        if item["direction"] == "SELF_ENHANCED_HIGHER"
    )

    neutral_higher = sum(
        1
        for item in successful
        if item["direction"] == "NEUTRAL_HIGHER"
    )

    no_difference = sum(
        1
        for item in successful
        if item["direction"] == "NO_DIFFERENCE"
    )

    differences = [
        item["score_difference"]
        for item in successful
    ]

    mean_difference = (
        round(
            sum(differences) / len(differences),
            4,
        )
        if differences
        else None
    )

    return {
        "experiment": {
            "name": "Self-enhancement bias experiment",
            "mode": "self_enhancement",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_tests": len(tests),
            "successful": len(successful),
            "failed": len(tests) - len(successful),
            "self_enhanced_higher": enhanced_higher,
            "neutral_higher": neutral_higher,
            "no_difference": no_difference,
            "mean_score_difference": mean_difference,
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
        / f"self_enhancement_{timestamp}.json"
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
    """Print final experiment summary."""

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("SELF-ENHANCEMENT BIAS EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Total tests            : "
        f"{summary['total_tests']}"
    )

    print(
        f"Successful             : "
        f"{summary['successful']}"
    )

    print(
        f"Failed                 : "
        f"{summary['failed']}"
    )

    print(
        f"Self-enhanced higher   : "
        f"{summary['self_enhanced_higher']}"
    )

    print(
        f"Neutral higher         : "
        f"{summary['neutral_higher']}"
    )

    print(
        f"No difference          : "
        f"{summary['no_difference']}"
    )

    print(
        f"Mean score difference  : "
        f"{summary['mean_score_difference']}"
    )

    print("=" * 60)


def main() -> None:
    """Run the self-enhancement experiment."""

    print("Loading configuration...")

    config = load_config()

    print(
        "Loading self-enhancement test suite..."
    )

    test_suite = load_test_suite()

    print("Creating judge client...")

    judge = create_judge(config)

    print(
        f"Loaded {len(test_suite['tests'])} "
        f"self-enhancement test(s)."
    )

    print(
        "\nStarting self-enhancement experiment...\n"
    )

    results = run_experiment(
        judge,
        test_suite,
    )

    output_path = save_results(results)

    print_summary(results)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()