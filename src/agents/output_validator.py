"""
Output validator and retry logic for agent LLM calls.

Every LLM response goes through this before being used downstream.
Handles: JSON parsing, Pydantic validation, field normalization, and retries.

This is what separates a demo from a production system.
"""

import json
import time
import re
from typing import TypeVar, Type, Callable, Any
from pydantic import BaseModel, ValidationError
from groq import Groq, RateLimitError, APITimeoutError, APIConnectionError

from src.config import GROQ_API_KEY, MAX_RETRIES_PER_AGENT, AGENT_TIMEOUT_SECONDS, SEVERITY_LEVELS


T = TypeVar("T", bound=BaseModel)


# --- JSON Extraction ---

def extract_json_from_response(text: str) -> str:
    """
    Extract JSON from an LLM response that might have extra text around it.

    LLMs sometimes wrap JSON in markdown or add preamble like "Sure! Here you go:".
    This function finds the first valid JSON object in the response.

    Examples of what this handles:
        "Sure! Here's the JSON: {"key": "value"}"  → '{"key": "value"}'
        "```json\n{"key": "value"}\n```"            → '{"key": "value"}'
        '{"key": "value"}'                          → '{"key": "value"}'
    """
    # Strip markdown code blocks
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()

    # Find the first { and last } to extract JSON object
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]

    return text


def normalize_severity(value: str) -> str:
    """
    Normalize severity to lowercase and validate it's a known value.

    LLMs sometimes return "HIGH" or "High" instead of "high".
    """
    normalized = value.lower().strip()
    if normalized in SEVERITY_LEVELS:
        return normalized
    # Default to medium if unknown value
    return "medium"


def parse_and_validate(raw_text: str, schema: Type[T]) -> T:
    """
    Parse LLM output and validate against a Pydantic schema.

    Args:
        raw_text: raw string from LLM
        schema: Pydantic model class to validate against

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: if parsing or validation fails
    """
    # Extract JSON
    json_str = extract_json_from_response(raw_text)

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in LLM response: {e}\nRaw: {raw_text[:200]}")

    # Normalize severity if present
    if "severity" in data:
        data["severity"] = normalize_severity(data["severity"])

    # Validate with Pydantic
    try:
        return schema(**data)
    except ValidationError as e:
        raise ValueError(f"Schema validation failed: {e}")


# --- Retry Logic ---

RETRY_SYSTEM_PROMPT_SUFFIX = """

IMPORTANT: Your previous response was not valid JSON or did not match the required schema.
You MUST respond with ONLY a valid JSON object. No text before or after the JSON.
No markdown formatting. No explanations. Just the raw JSON object."""


def call_llm_with_retry(
    client: Groq,
    model: str,
    system_prompt: str,
    user_message: str,
    output_schema: Type[T],
    max_retries: int = MAX_RETRIES_PER_AGENT,
) -> T:
    """
    Call the LLM with automatic retry on failure.

    Retry strategy:
    - Attempt 1: normal call
    - Attempt 2: add stricter JSON instruction to system prompt
    - Attempt 3: simplify user message, add even stronger instruction
    - After 3 failures: raise exception (coordinator handles graceful degradation)

    Args:
        client: Groq client instance
        model: model name to use
        system_prompt: system prompt for the agent
        user_message: user message with the actual task
        output_schema: Pydantic model to validate response against
        max_retries: maximum number of attempts

    Returns:
        Validated Pydantic model instance

    Raises:
        RuntimeError: if all retries exhausted
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            # On retry, add stronger JSON instruction
            effective_system = system_prompt
            if attempt > 1:
                effective_system = system_prompt + RETRY_SYSTEM_PROMPT_SUFFIX

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": effective_system},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                timeout=AGENT_TIMEOUT_SECONDS,
            )

            raw_text = response.choices[0].message.content
            return parse_and_validate(raw_text, output_schema)

        except RateLimitError:
            # Rate limited — wait and retry
            wait_time = 2 ** attempt  # exponential backoff: 2s, 4s, 8s
            print(f"  Rate limited on attempt {attempt}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            last_error = "Rate limit exceeded"

        except APITimeoutError:
            print(f"  Timeout on attempt {attempt}. Retrying...")
            last_error = "API timeout"

        except APIConnectionError:
            print(f"  Connection error on attempt {attempt}. Retrying...")
            time.sleep(1)
            last_error = "Connection error"

        except ValueError as e:
            # JSON parsing or validation error — retry with stricter prompt
            print(f"  Validation error on attempt {attempt}: {e}")
            last_error = str(e)

        except Exception as e:
            print(f"  Unexpected error on attempt {attempt}: {e}")
            last_error = str(e)

    raise RuntimeError(
        f"LLM call failed after {max_retries} attempts. Last error: {last_error}"
    )
