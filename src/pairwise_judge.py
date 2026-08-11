"""
Pairwise LLM-as-Judge evaluator.

Compares two candidate responses (A and B) for the same task and asks
the configured judge LLM to determine which response is better.

The judge returns a structured JSON verdict containing:
- winner: A, B, or TIE
- score for candidate A
- score for candidate B
- criterion-level rationale
- overall rationale

The final winner is determined from the validated criterion scores.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .llm_client import LLMClient


PAIRWISE_SYSTEM_PROMPT = """
You are an impartial and rigorous LLM-as-Judge evaluator.

Your task is to compare two candidate responses to the same user task.

Evaluation principles:

1. Evaluate the responses based only on the task and supplied evidence.
2. Do not identify or infer which model produced either response.
3. Do not prefer a response merely because it is longer.
4. Do not prefer a response merely because it uses more sophisticated language.
5. Prefer correctness, faithfulness, completeness, instruction-following,
   relevance, and safety.
6. Treat semantically equivalent answers as equivalent.
7. Do not use superficial word overlap as the primary basis for judgment.
8. Give evidence-based reasons for every criterion score.
9. Do not invent facts.
10. Return ONLY valid JSON.
11. Do not use Markdown code fences.
12. Do not include text before or after the JSON object.
""".strip()


PAIRWISE_CRITERIA = [
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "relevance",
    "safety",
]


class PairwiseJudgeError(RuntimeError):
    """Raised when pairwise judging fails or produces an invalid verdict."""


class PairwiseJudge:
    """
    Pairwise A/B LLM-as-Judge evaluator.

    Uses the existing LLMClient to compare candidate A and candidate B.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        tie_margin: float = 0.25,
    ) -> None:
        if tie_margin < 0:
            raise ValueError("tie_margin cannot be negative.")

        self.client = client
        self.tie_margin = tie_margin

    def evaluate(
        self,
        *,
        test_id: str,
        user_input: str,
        system_prompt: str,
        candidate_a: str,
        candidate_b: str,
        expected_output: str = "",
    ) -> Dict[str, Any]:
        """
        Compare candidate A and candidate B.

        Returns a validated pairwise verdict.
        """

        self._validate_inputs(
            test_id=test_id,
            user_input=user_input,
            system_prompt=system_prompt,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
        )

        prompt = self._build_prompt(
            test_id=test_id,
            user_input=user_input,
            system_prompt=system_prompt,
            candidate_a=candidate_a,
            candidate_b=candidate_b,
            expected_output=expected_output,
        )

        response = self.client.generate(
            prompt,
            system_prompt=PAIRWISE_SYSTEM_PROMPT,
        )

        verdict = self._parse_json(response.text)

        validated = self._validate_verdict(
            verdict=verdict,
            test_id=test_id,
        )

        validated["usage"] = {
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "total_tokens": response.total_tokens,
            "latency_ms": response.latency_ms,
        }

        return validated

    @staticmethod
    def _validate_inputs(
        *,
        test_id: str,
        user_input: str,
        system_prompt: str,
        candidate_a: str,
        candidate_b: str,
    ) -> None:
        """Validate pairwise evaluation inputs."""

        fields = {
            "test_id": test_id,
            "user_input": user_input,
            "system_prompt": system_prompt,
            "candidate_a": candidate_a,
            "candidate_b": candidate_b,
        }

        for name, value in fields.items():
            if not isinstance(value, str):
                raise PairwiseJudgeError(
                    f"'{name}' must be a string."
                )

            if not value.strip():
                raise PairwiseJudgeError(
                    f"'{name}' cannot be empty."
                )

    def _build_prompt(
        self,
        *,
        test_id: str,
        user_input: str,
        system_prompt: str,
        candidate_a: str,
        candidate_b: str,
        expected_output: str,
    ) -> str:
        """Build the pairwise judge prompt."""

        reference_section = (
            expected_output
            if expected_output.strip()
            else "No reference answer is available."
        )

        return f"""
Compare Candidate A and Candidate B for the following task.

TEST ID:
{test_id}

USER INPUT:
{user_input}

SYSTEM PROMPT:
{system_prompt}

REFERENCE / EXPECTED OUTPUT:
{reference_section}

CANDIDATE A:
{candidate_a}

CANDIDATE B:
{candidate_b}

EVALUATION CRITERIA:
{json.dumps(PAIRWISE_CRITERIA, indent=2)}

SCORING SCALE:
1 = Fundamentally inadequate
2 = Major problems
3 = Partially acceptable
4 = Good, minor issues
5 = Excellent

Evaluate BOTH candidates independently on every criterion.

For each criterion:
- assign Candidate A a score from 1 to 5
- assign Candidate B a score from 1 to 5
- provide concise evidence-based reasoning

Then calculate the average score for A and B.

WINNER RULE:
- A wins if A's average score is greater than B's average score
  by more than the tie margin.
- B wins if B's average score is greater than A's average score
  by more than the tie margin.
- Otherwise the result is TIE.

TIE MARGIN:
{self.tie_margin}

IMPORTANT:
- Do not choose a winner based only on verbosity.
- Do not choose a winner based only on writing style.
- Do not assume A is better because it appears first.
- Do not assume B is better because it appears second.
- Compare the actual quality of both responses.
- The winner must be supported by the criterion scores.
- Return ONLY JSON.

RETURN EXACTLY THIS STRUCTURE:

{{
  "test_id": "{test_id}",
  "candidate_a": {{
    "criteria": {{
      "correctness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "faithfulness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "completeness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "instruction_following": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "relevance": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "safety": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }}
    }},
    "average_score": 0.0
  }},
  "candidate_b": {{
    "criteria": {{
      "correctness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "faithfulness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "completeness": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "instruction_following": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "relevance": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }},
      "safety": {{
        "score": 1,
        "rationale": "Evidence-based reason."
      }}
    }},
    "average_score": 0.0
  }},
  "winner": "A",
  "overall_rationale": "Brief explanation of why the winner is better."
}}

Rules:
- Every criterion score MUST be an integer from 1 through 5.
- Every criterion MUST appear for both candidates.
- average_score MUST be between 1 and 5.
- winner MUST be exactly "A", "B", or "TIE".
- overall_rationale MUST be a non-empty string.
""".strip()

    @staticmethod
    def _parse_json(text: str) -> Dict[str, Any]:
        """Parse JSON returned by the judge."""

        cleaned = text.strip()

        if cleaned.startswith("```") and cleaned.endswith("```"):
            lines = cleaned.splitlines()

            if len(lines) >= 3:
                lines = lines[1:-1]

            cleaned = "\n".join(lines).strip()

            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        try:
            parsed = json.loads(cleaned)

        except json.JSONDecodeError as exc:
            raise PairwiseJudgeError(
                "Pairwise judge returned invalid JSON. "
                f"Raw response: {text[:1000]}"
            ) from exc

        if not isinstance(parsed, dict):
            raise PairwiseJudgeError(
                "Pairwise judge response must be a JSON object."
            )

        return parsed

    def _validate_verdict(
        self,
        *,
        verdict: Dict[str, Any],
        test_id: str,
    ) -> Dict[str, Any]:
        """Validate and normalize the pairwise verdict."""

        required_fields = {
            "test_id",
            "candidate_a",
            "candidate_b",
            "winner",
            "overall_rationale",
        }

        missing = required_fields - set(verdict.keys())

        if missing:
            raise PairwiseJudgeError(
                "Pairwise verdict is missing fields: "
                + ", ".join(sorted(missing))
            )

        if str(verdict["test_id"]) != test_id:
            raise PairwiseJudgeError(
                f"Expected test_id '{test_id}', "
                f"got '{verdict['test_id']}'."
            )

        normalized_a = self._validate_candidate(
            verdict["candidate_a"],
            candidate_name="candidate_a",
        )

        normalized_b = self._validate_candidate(
            verdict["candidate_b"],
            candidate_name="candidate_b",
        )

        winner = verdict["winner"]

        if winner not in {"A", "B", "TIE"}:
            raise PairwiseJudgeError(
                "Winner must be exactly 'A', 'B', or 'TIE'."
            )

        overall_rationale = verdict["overall_rationale"]

        if (
            not isinstance(overall_rationale, str)
            or not overall_rationale.strip()
        ):
            raise PairwiseJudgeError(
                "overall_rationale must be a non-empty string."
            )

        calculated_a = self._calculate_average(
            normalized_a["criteria"]
        )

        calculated_b = self._calculate_average(
            normalized_b["criteria"]
        )

        # The LLM-reported averages are retained for auditability.
        reported_a = normalized_a["average_score"]
        reported_b = normalized_b["average_score"]

        # Use locally calculated averages as authoritative values.
        authoritative_winner = self._determine_winner(
            calculated_a,
            calculated_b,
        )

        return {
            "test_id": test_id,
            "candidate_a": {
                "criteria": normalized_a["criteria"],
                "average_score": calculated_a,
                "reported_average_score": reported_a,
            },
            "candidate_b": {
                "criteria": normalized_b["criteria"],
                "average_score": calculated_b,
                "reported_average_score": reported_b,
            },
            "winner": authoritative_winner,
            "reported_winner": winner,
            "winner_agrees_with_report": (
                authoritative_winner == winner
            ),
            "overall_rationale": overall_rationale.strip(),
        }

    @staticmethod
    def _validate_candidate(
        candidate: Any,
        *,
        candidate_name: str,
    ) -> Dict[str, Any]:
        """Validate one candidate's criterion scores."""

        if not isinstance(candidate, dict):
            raise PairwiseJudgeError(
                f"'{candidate_name}' must be an object."
            )

        if "criteria" not in candidate:
            raise PairwiseJudgeError(
                f"'{candidate_name}' is missing 'criteria'."
            )

        if "average_score" not in candidate:
            raise PairwiseJudgeError(
                f"'{candidate_name}' is missing 'average_score'."
            )

        criteria = candidate["criteria"]

        if not isinstance(criteria, dict):
            raise PairwiseJudgeError(
                f"'{candidate_name}.criteria' must be an object."
            )

        expected = set(PAIRWISE_CRITERIA)
        actual = set(criteria.keys())

        missing = expected - actual
        unexpected = actual - expected

        if missing:
            raise PairwiseJudgeError(
                f"{candidate_name} missing criteria: "
                + ", ".join(sorted(missing))
            )

        if unexpected:
            raise PairwiseJudgeError(
                f"{candidate_name} contains unexpected criteria: "
                + ", ".join(sorted(unexpected))
            )

        normalized: Dict[str, Dict[str, Any]] = {}

        for criterion in PAIRWISE_CRITERIA:
            item = criteria[criterion]

            if not isinstance(item, dict):
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion} must be an object."
                )

            if "score" not in item:
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion} is missing score."
                )

            if "rationale" not in item:
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion} is missing rationale."
                )

            score = item["score"]

            if not isinstance(score, int) or isinstance(score, bool):
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion}.score "
                    "must be an integer."
                )

            if not 1 <= score <= 5:
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion}.score "
                    "must be between 1 and 5."
                )

            rationale = item["rationale"]

            if (
                not isinstance(rationale, str)
                or not rationale.strip()
            ):
                raise PairwiseJudgeError(
                    f"{candidate_name}.{criterion}.rationale "
                    "must be a non-empty string."
                )

            normalized[criterion] = {
                "score": score,
                "rationale": rationale.strip(),
            }

        try:
            average_score = float(candidate["average_score"])

        except (TypeError, ValueError) as exc:
            raise PairwiseJudgeError(
                f"{candidate_name}.average_score must be numeric."
            ) from exc

        if not 1 <= average_score <= 5:
            raise PairwiseJudgeError(
                f"{candidate_name}.average_score must be "
                "between 1 and 5."
            )

        return {
            "criteria": normalized,
            "average_score": average_score,
        }

    @staticmethod
    def _calculate_average(
        criteria: Dict[str, Dict[str, Any]],
    ) -> float:
        """Calculate the authoritative average criterion score."""

        scores = [
            float(criteria[criterion]["score"])
            for criterion in PAIRWISE_CRITERIA
        ]

        return round(sum(scores) / len(scores), 4)

    def _determine_winner(
        self,
        score_a: float,
        score_b: float,
    ) -> str:
        """Determine the authoritative pairwise winner."""

        difference = score_a - score_b

        if difference > self.tie_margin:
            return "A"

        if difference < -self.tie_margin:
            return "B"

        return "TIE"