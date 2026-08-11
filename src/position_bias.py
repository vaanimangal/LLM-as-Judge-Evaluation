"""
Position-bias experiment for the LLM-as-Judge framework.

The experiment evaluates the same pair of candidate responses twice:

Run 1:
    Candidate A -> Position A
    Candidate B -> Position B

Run 2:
    Candidate B -> Position A
    Candidate A -> Position B

A position-biased judge may change its winner when only the
candidate positions are swapped.

The experiment records both decisions and calculates whether
the underlying winner remained consistent.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

from .llm_client import LLMClient


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_SUITE_PATH = PROJECT_ROOT / "data" / "position_bias_test_suite.json"
LOGS_DIR = PROJECT_ROOT / "logs"

load_dotenv(PROJECT_ROOT / ".env")


POSITION_BIAS_SYSTEM_PROMPT = """
You are an impartial pairwise evaluator.

Compare two candidate responses to the same task.

Your job is to determine which candidate is better according to:
1. Correctness
2. Faithfulness
3. Completeness
4. Instruction-following
5. Relevance
6. Safety

Important rules:

- Judge the responses based only on their content and the task.
- Do not prefer a response because it appears first.
- Do not prefer a response because it appears second.
- Do not reward verbosity unless the task requires it.
- Do not penalize concise answers when they fully satisfy the task.
- Use the expected output as reference evidence when available.
- Evaluate semantic correctness rather than word overlap.
- Return ONLY valid JSON.
""".strip()

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
            "One or more environment variables referenced in "
            "config.yaml are not set:\n"
            + details
        )

    config = yaml.safe_load(config_text)

    if not isinstance(config, dict):
        raise RuntimeError(
            "config.yaml must contain a YAML object."
        )

    return config


def load_test_suite() -> Dict[str, Any]:
    """Load position-bias test cases."""

    if not TEST_SUITE_PATH.exists():
        raise FileNotFoundError(
            f"Position-bias test suite not found: "
            f"{TEST_SUITE_PATH}"
        )

    with TEST_SUITE_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise RuntimeError(
            "position_bias_test_suite.json must contain an object."
        )

    tests = data.get("tests")

    if not isinstance(tests, list) or not tests:
        raise RuntimeError(
            "Position-bias test suite must contain a non-empty 'tests' list."
        )

    return data


def create_client(config: Dict[str, Any]) -> LLMClient:
    """Create the configured judge client."""

    judge_config = config.get("judge")

    if not isinstance(judge_config, dict):
        raise RuntimeError(
            "Missing 'judge' configuration in config.yaml."
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

    if not api_key:
        raise RuntimeError(
            "JUDGE_API_KEY is not set in .env."
        )

    evaluation_config = config.get("evaluation", {})

    temperature = float(
        evaluation_config.get("temperature", 0.0)
    )

    return LLMClient(
        api_key=api_key,
        model=str(model),
        base_url=str(base_url),
        temperature=temperature,
    )


def build_prompt(
    test_case: Dict[str, Any],
    candidate_a: str,
    candidate_b: str,
) -> str:
    """Build the pairwise position-bias evaluation prompt."""

    return f"""
Evaluate the two candidate responses below.

TEST ID:
{test_case["id"]}

TASK INPUT:
{test_case["input"]}

SYSTEM PROMPT:
{test_case["system_prompt"]}

EXPECTED OUTPUT:
{test_case["expected_output"]}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

Determine which candidate is better overall.

IMPORTANT:
- Candidate A and Candidate B are anonymous.
- Do not favor the first or second position.
- Base the decision on quality relative to the task.
- If the candidates are effectively equal, return TIE.
- Do not explain outside the JSON.

Return exactly:

{{
    "winner": "A",
    "reason": "Brief evidence-based explanation."
}}

The winner must be exactly one of:
"A", "B", "TIE".
""".strip()


def parse_verdict(text: str) -> Dict[str, Any]:
    """Parse and validate the judge's JSON response."""

    cleaned = text.strip()

    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()

        if len(lines) >= 3:
            cleaned = "\n".join(lines[1:-1]).strip()

        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Judge returned invalid JSON: {text[:500]}"
        ) from exc

    if not isinstance(data, dict):
        raise RuntimeError(
            "Judge verdict must be a JSON object."
        )

    winner = data.get("winner")

    if winner not in {"A", "B", "TIE"}:
        raise RuntimeError(
            "Judge winner must be A, B, or TIE."
        )

    reason = data.get("reason")

    if not isinstance(reason, str) or not reason.strip():
        raise RuntimeError(
            "Judge reason must be a non-empty string."
        )

    return {
        "winner": winner,
        "reason": reason.strip(),
    }


def evaluate_position_bias(
    client: LLMClient,
    test_case: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Evaluate one pair in both candidate positions.
    """

    candidate_a = test_case["candidate_a"]
    candidate_b = test_case["candidate_b"]

    # ---------------------------------------------------------
    # Run 1
    #
    # Position A = original Candidate A
    # Position B = original Candidate B
    # ---------------------------------------------------------

    prompt_original = build_prompt(
        test_case=test_case,
        candidate_a=candidate_a,
        candidate_b=candidate_b,
    )

    response_original = client.generate(
        prompt_original,
        system_prompt=POSITION_BIAS_SYSTEM_PROMPT,
    )

    verdict_original = parse_verdict(
        response_original.text
    )

    # ---------------------------------------------------------
    # Run 2
    #
    # Position A = original Candidate B
    # Position B = original Candidate A
    # ---------------------------------------------------------

    prompt_swapped = build_prompt(
        test_case=test_case,
        candidate_a=candidate_b,
        candidate_b=candidate_a,
    )

    response_swapped = client.generate(
        prompt_swapped,
        system_prompt=POSITION_BIAS_SYSTEM_PROMPT,
    )

    verdict_swapped = parse_verdict(
        response_swapped.text
    )

    # Convert positional winners into original candidate identity.
    #
    # Original run:
    #   A -> candidate_a
    #   B -> candidate_b
    #
    # Swapped run:
    #   A -> candidate_b
    #   B -> candidate_a

    original_winner = verdict_original["winner"]

    if original_winner == "A":
        original_candidate_winner = "candidate_a"
    elif original_winner == "B":
        original_candidate_winner = "candidate_b"
    else:
        original_candidate_winner = "TIE"

    swapped_winner = verdict_swapped["winner"]

    if swapped_winner == "A":
        swapped_candidate_winner = "candidate_b"
    elif swapped_winner == "B":
        swapped_candidate_winner = "candidate_a"
    else:
        swapped_candidate_winner = "TIE"

    # A consistent judge should select the same underlying
    # candidate after the positions are swapped.

    position_consistent = (
        original_candidate_winner
        == swapped_candidate_winner
    )

    return {
        "test_id": str(test_case["id"]),

        "original_order": {
            "position_A": "candidate_a",
            "position_B": "candidate_b",
            "winner": original_winner,
            "underlying_winner": original_candidate_winner,
            "reason": verdict_original["reason"],
            "usage": response_original.to_dict(),
        },

        "swapped_order": {
            "position_A": "candidate_b",
            "position_B": "candidate_a",
            "winner": swapped_winner,
            "underlying_winner": swapped_candidate_winner,
            "reason": verdict_swapped["reason"],
            "usage": response_swapped.to_dict(),
        },

        "position_consistent": position_consistent,
    }


def run_experiment(
    client: LLMClient,
    test_suite: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the complete position-bias experiment."""

    started_at = datetime.now(timezone.utc).isoformat()

    results = []

    consistent_count = 0
    inconsistent_count = 0

    tests = test_suite["tests"]

    for index, test_case in enumerate(tests, start=1):
        test_id = test_case.get("id", f"pos_{index}")

        print(
            f"[{index}/{len(tests)}] Testing {test_id}...",
            flush=True,
        )

        try:
            result = evaluate_position_bias(
                client=client,
                test_case=test_case,
            )

            results.append(
                {
                    "status": "success",
                    "result": result,
                }
            )

            if result["position_consistent"]:
                consistent_count += 1
                print("    Position consistent")
            else:
                inconsistent_count += 1
                print("    POSITION INCONSISTENCY DETECTED")

        except Exception as exc:
            results.append(
                {
                    "status": "failed",
                    "test_id": str(test_id),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

            print(
                f"    FAILED: {type(exc).__name__}: {exc}"
            )

    completed_at = datetime.now(timezone.utc).isoformat()

    successful = sum(
        1
        for item in results
        if item["status"] == "success"
    )

    failed = len(results) - successful

    consistency_rate = (
        round(
            (consistent_count / successful) * 100,
            2,
        )
        if successful
        else None
    )

    return {
        "experiment": {
            "name": "Position Bias Experiment",
            "mode": "position_bias",
            "started_at": started_at,
            "completed_at": completed_at,
        },

        "summary": {
            "total_tests": len(tests),
            "successful": successful,
            "failed": failed,
            "consistent": consistent_count,
            "inconsistent": inconsistent_count,
            "position_consistency_rate": consistency_rate,
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
        / f"position_bias_{timestamp}.json"
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


def main() -> None:
    """Run the position-bias experiment."""

    print("Loading configuration...")

    config = load_config()

    print("Loading position-bias test suite...")

    test_suite = load_test_suite()

    print("Creating judge client...")

    client = create_client(config)

    print(
        f"Loaded {len(test_suite['tests'])} position-bias test(s)."
    )

    print(
        "\nStarting position-bias experiment...\n"
    )

    results = run_experiment(
        client=client,
        test_suite=test_suite,
    )

    output_path = save_results(results)

    summary = results["summary"]

    print("\n" + "=" * 60)
    print("POSITION-BIAS EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Total tests       : {summary['total_tests']}"
    )
    print(
        f"Successful        : {summary['successful']}"
    )
    print(
        f"Failed            : {summary['failed']}"
    )
    print(
        f"Consistent        : {summary['consistent']}"
    )
    print(
        f"Inconsistent      : {summary['inconsistent']}"
    )

    if summary["position_consistency_rate"] is not None:
        print(
            "Consistency rate  : "
            f"{summary['position_consistency_rate']:.2f}%"
        )

    print("=" * 60)

    print("\nResults saved to:")
    print(output_path)


if __name__ == "__main__":
    main()