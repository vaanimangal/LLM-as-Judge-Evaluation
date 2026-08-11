"""
Score-clustering experiment for an LLM-as-Judge system.

Purpose
-------
Tests whether the judge disproportionately concentrates scores around
a small portion of the available 1-5 scoring scale.

The experiment:
1. Evaluates deliberately varied candidate responses.
2. Collects criterion-level scores.
3. Calculates score-frequency distribution.
4. Calculates mean and standard deviation.
5. Measures the proportion of scores in the dominant cluster.
6. Flags potential score-clustering when the distribution is unusually
   concentrated.

This experiment does NOT modify the existing pointwise or pairwise
evaluation pipeline.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from src.llm_client import LLMClient
from src.judge import Judge, JudgeError


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = PROJECT_ROOT / "data" / "score_clustering_test_suite.json"
LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config() -> Dict[str, Any]:
    """Load configuration and resolve environment variables."""

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
    """Load the score-clustering test suite."""

    if not TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            f"Score-clustering test suite not found: "
            f"{TEST_SUITE_PATH}"
        )

    with TEST_SUITE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "score_clustering_test_suite.json must contain an object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list):
        raise RuntimeError(
            "Test suite must contain a 'tests' list."
        )

    if not tests:
        raise RuntimeError(
            "Score-clustering test suite contains no tests."
        )

    return data


# ---------------------------------------------------------------------------
# Judge creation
# ---------------------------------------------------------------------------

def create_judge(config: Dict[str, Any]) -> Judge:
    """Create the configured judge."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing or invalid 'judge' configuration."
        )

    provider = judge_config.get("provider")

    if provider != "openai_compatible":
        raise RuntimeError(
            f"Unsupported judge provider: {provider!r}"
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

    if not isinstance(evaluation_config, dict):
        raise RuntimeError(
            "'evaluation' configuration must be an object."
        )

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
# Score extraction
# ---------------------------------------------------------------------------

def extract_scores(
    successful_results: List[Dict[str, Any]],
) -> List[int]:
    """
    Extract all criterion-level 1-5 scores from successful evaluations.
    """

    scores: List[int] = []

    for result in successful_results:
        criteria = result.get("criteria", {})

        if not isinstance(criteria, dict):
            continue

        for criterion_result in criteria.values():

            if not isinstance(criterion_result, dict):
                continue

            score = criterion_result.get("score")

            if isinstance(score, int) and not isinstance(score, bool):
                if 1 <= score <= 5:
                    scores.append(score)

    return scores


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------

def calculate_distribution(
    scores: List[int],
) -> Dict[str, Any]:
    """Calculate score distribution and clustering statistics."""

    if not scores:
        return {
            "total_scores": 0,
            "distribution": {
                "1": 0,
                "2": 0,
                "3": 0,
                "4": 0,
                "5": 0,
            },
            "mean": None,
            "standard_deviation": None,
            "dominant_score": None,
            "dominant_score_count": 0,
            "dominant_score_percentage": None,
            "clustered": False,
        }

    counter = Counter(scores)

    total = len(scores)

    distribution = {
        str(score): counter.get(score, 0)
        for score in range(1, 6)
    }

    mean = sum(scores) / total

    variance = sum(
        (score - mean) ** 2
        for score in scores
    ) / total

    standard_deviation = variance ** 0.5

    dominant_score, dominant_count = counter.most_common(1)[0]

    dominant_percentage = (
        dominant_count / total
    ) * 100

    # A simple, transparent experimental threshold:
    #
    # If >= 60% of all criterion scores are concentrated
    # on one score, flag possible score clustering.
    #
    # This is a diagnostic threshold, not a universal statistical
    # definition of bias.
    clustered = dominant_percentage >= 60.0

    return {
        "total_scores": total,
        "distribution": distribution,
        "mean": round(mean, 4),
        "standard_deviation": round(
            standard_deviation,
            4,
        ),
        "dominant_score": dominant_score,
        "dominant_score_count": dominant_count,
        "dominant_score_percentage": round(
            dominant_percentage,
            2,
        ),
        "clustered": clustered,
    }


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run_experiment(
    judge: Judge,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the complete score-clustering experiment."""

    tests = test_suite["tests"]

    results: List[Dict[str, Any]] = []

    successful = 0
    failed = 0

    started_at = datetime.now(timezone.utc).isoformat()

    for index, test_case in enumerate(tests, start=1):

        test_id = str(
            test_case.get(
                "id",
                f"score_cluster_{index}",
            )
        )

        print(
            f"[{index}/{len(tests)}] Testing {test_id}...",
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
                f"    Overall score: "
                f"{result['overall_score']:.2f}/100",
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

    successful_results = [
        item["result"]
        for item in results
        if item["status"] == "success"
    ]

    scores = extract_scores(
        successful_results
    )

    distribution = calculate_distribution(
        scores
    )

    return {
        "experiment": {
            "name": "Score-clustering experiment",
            "mode": "score_clustering",
            "started_at": started_at,
            "completed_at": completed_at,
        },
        "summary": {
            "total_tests": len(tests),
            "successful": successful,
            "failed": failed,
            "total_criterion_scores": len(scores),
            "clustered": distribution["clustered"],
        },
        "score_analysis": distribution,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(
    results: Dict[str, Any],
) -> Path:
    """Save score-clustering results."""

    LOGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d_%H%M%S")

    output_path = (
        LOGS_DIR
        / f"score_clustering_{timestamp}.json"
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
    """Print a concise experiment summary."""

    summary = results["summary"]
    analysis = results["score_analysis"]

    print("\n" + "=" * 60)
    print("SCORE-CLUSTERING EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Total tests           : "
        f"{summary['total_tests']}"
    )

    print(
        f"Successful            : "
        f"{summary['successful']}"
    )

    print(
        f"Failed                : "
        f"{summary['failed']}"
    )

    print(
        f"Criterion scores      : "
        f"{analysis['total_scores']}"
    )

    print("\nScore distribution:")

    for score, count in analysis[
        "distribution"
    ].items():

        print(
            f"    Score {score}: "
            f"{count}"
        )

    if analysis["mean"] is not None:

        print(
            f"\nMean score            : "
            f"{analysis['mean']:.4f}"
        )

        print(
            f"Standard deviation    : "
            f"{analysis['standard_deviation']:.4f}"
        )

        print(
            f"Dominant score        : "
            f"{analysis['dominant_score']}"
        )

        print(
            f"Dominant percentage   : "
            f"{analysis['dominant_score_percentage']:.2f}%"
        )

        print(
            f"Clustering detected   : "
            f"{'YES' if analysis['clustered'] else 'NO'}"
        )

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the score-clustering experiment."""

    print("Loading configuration...")

    config = load_config()

    print(
        "Loading score-clustering test suite..."
    )

    test_suite = load_test_suite()

    print("Creating judge client...")

    judge = create_judge(config)

    print(
        f"Loaded {len(test_suite['tests'])} "
        f"score-clustering test(s)."
    )

    print(
        "\nStarting score-clustering experiment...\n"
    )

    results = run_experiment(
        judge=judge,
        test_suite=test_suite,
    )

    output_path = save_results(
        results
    )

    print_summary(results)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()