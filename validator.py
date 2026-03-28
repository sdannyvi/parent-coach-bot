"""
Answer validation module for the parent coaching chatbot.

This file belongs to the parent_coach_bot module and is responsible for all
validation logic applied to parent answers during the protocol flow.

Validation runs in three layers:
1. Hard rules — deterministic checks applied before the LLM (Q1 word ban, Q4 format).
2. LLM evaluation — chain-of-thought reasoning hidden inside the model; the model
   returns a structured JSON verdict with alignment, confidence, and feedback.
3. should_accept override — post-LLM logic that decides whether to accept the answer
   based on alignment + confidence, and applies the soft-pass mechanism after 2 attempts.

Update this file when validation rules, acceptance policy, or the LLM prompt change.
"""

import json
import logging
from openai import OpenAI

from llm_client import get_deployment, get_temperature, get_max_tokens

logger = logging.getLogger(__name__)

# Q4 only accepts one of these exact strings (stripped)
_Q4_VALID = {"כן", "לא"}

# Confidence threshold above which a not_aligned verdict causes a hard reject
_CONFIDENCE_REJECT_THRESHOLD = 0.8

# Number of failed attempts after which the answer is force-accepted (soft pass)
_SOFT_PASS_AFTER = 2

_EVAL_SYSTEM_PROMPT = (
    "You are a precise evaluator of parenting protocol responses. "
    "Carefully analyze the parent's answer internally step by step before deciding. "
    "Be permissive — avoid over-rejecting reasonable or interpretable answers. "
    "Do not expose your internal reasoning. "
    "Respond with valid JSON only, no extra text."
)


def _call_llm(client: OpenAI, messages: list[dict]) -> str:
    """
    Send messages to the LLM and return the raw response string.

    Args:
        client: Authenticated OpenAI client.
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Raw response text from the model.
    """
    response = client.chat.completions.create(
        model=get_deployment(),
        messages=messages,
        temperature=get_temperature(),
        max_tokens=get_max_tokens(),
    )
    return response.choices[0].message.content.strip()


def _parse_json(raw: str) -> dict:
    """
    Parse a JSON string returned by the LLM, stripping markdown fences if present.

    Args:
        raw: Raw string from the LLM.

    Returns:
        Parsed dict.
    """
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def _compute_should_accept(alignment: str, confidence: float, attempt_count: int) -> bool:
    """
    Decide whether to accept the parent's answer based on alignment + confidence.

    Acceptance policy:
    - aligned → always accept
    - partially_aligned → accept (close enough)
    - not_aligned + confidence < threshold → accept (benefit of the doubt)
    - not_aligned + confidence >= threshold → reject (model is confident it's wrong)
    - Soft pass override: accept regardless after attempt_count >= _SOFT_PASS_AFTER

    Args:
        alignment: One of "aligned", "partially_aligned", "not_aligned".
        confidence: Float 0.0–1.0 from the LLM.
        attempt_count: How many times the parent has already tried this question.

    Returns:
        True if the answer should be accepted and the flow should advance.
    """
    if attempt_count >= _SOFT_PASS_AFTER:
        logger.info("Soft pass triggered after %d attempts.", attempt_count)
        return True

    if alignment in ("aligned", "partially_aligned"):
        return True

    # not_aligned: only hard-reject when model is highly confident
    return confidence < _CONFIDENCE_REJECT_THRESHOLD


def validate_answer(
    client: OpenAI,
    question_id: int,
    question: str,
    guidelines: str,
    answer: str,
    attempt_count: int,
) -> dict:
    """
    Validate a parent's answer against the protocol guidelines.

    Applies hard rules first, then calls the LLM for evaluation, then
    computes the final should_accept decision.

    Args:
        client: Authenticated OpenAI client.
        question_id: Protocol question id (1–4).
        question: Hebrew question text shown to the parent.
        guidelines: English alignment guidelines for this question.
        answer: The parent's raw answer string.
        attempt_count: Number of previous failed attempts on this question
                       (0 on the first try).

    Returns:
        Dict with keys:
          - alignment (str): "aligned" | "partially_aligned" | "not_aligned" | "invalid_format"
          - confidence (float): 0.0–1.0 (0.0 for hard-rule results)
          - reasoning_summary (str): brief internal reasoning (not shown to user)
          - feedback (str): Hebrew explanation shown to the parent when not accepted
          - should_accept (bool): whether the flow should advance past this question
    """
    # ── Layer 1: Hard rules ───────────────────────────────────────────────────

    if question_id == 1 and "לא" in answer.split():
        return {
            "alignment": "not_aligned",
            "confidence": 1.0,
            "reasoning_summary": "Hard rule: answer contains the word 'לא'.",
            "feedback": "התשובה צריכה לתאר מה הילד כן עשה, ולא מה הוא לא עשה.",
            "should_accept": False,
        }

    if question_id == 4 and answer.strip() not in _Q4_VALID:
        return {
            "alignment": "invalid_format",
            "confidence": 1.0,
            "reasoning_summary": "Hard rule: Q4 requires exactly כן or לא.",
            "feedback": "אנא ענה/י רק כן או לא.",
            "should_accept": False,
        }

    # ── Layer 2: LLM evaluation ───────────────────────────────────────────────

    prompt = (
        f"Protocol question:\n{question}\n\n"
        f"Alignment guidelines:\n{guidelines}\n\n"
        f"Parent's answer:\n{answer}\n\n"
        "Evaluate the answer. Think step by step internally, then respond ONLY with "
        "this JSON (no extra text):\n"
        "{\n"
        '  "alignment": "aligned | partially_aligned | not_aligned",\n'
        '  "confidence": <float 0.0-1.0>,\n'
        '  "reasoning_summary": "<one sentence internal summary>",\n'
        '  "feedback": "<short Hebrew explanation for the parent if not aligned>",\n'
        '  "should_accept": <true|false>\n'
        "}"
    )
    messages = [
        {"role": "system", "content": _EVAL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    raw = _call_llm(client, messages)
    result = _parse_json(raw)

    # ── Layer 3: Override should_accept with our acceptance policy ────────────

    alignment = result.get("alignment", "not_aligned")
    confidence = float(result.get("confidence", 0.5))
    result["should_accept"] = _compute_should_accept(alignment, confidence, attempt_count)

    return result
