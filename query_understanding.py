"""Query understanding: parse natural language into structured filters, then apply to a DataFrame."""

import json
import os
from pathlib import Path

import httpx
import pandas as pd

PROMPT_FILE = Path(__file__).parent / "prompts" / "query_understanding.txt"

FILTER_FIELDS = [
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "season",
    "year",
    "usage",
]

# OpenRouter config — override with env vars as needed.
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")


def _load_prompt_template() -> str:
    return PROMPT_FILE.read_text()


def get_field_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """Extract sorted unique non-null values for each filterable metadata field."""
    values: dict[str, list[str]] = {}
    for field in FILTER_FIELDS:
        if field not in df.columns:
            continue
        unique = df[field].dropna().unique().tolist()
        values[field] = sorted(str(v) for v in unique)
    return values


def build_prompt(user_query: str, field_values: dict[str, list[str]]) -> str:
    """Fill the prompt template with the actual catalog values and user query."""
    template = _load_prompt_template()

    replacements = {f"{field}_values": ", ".join(field_values.get(field, [])) for field in FILTER_FIELDS}
    replacements["user_query"] = user_query

    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(f"{{{key}}}", value)

    return prompt


def parse_query(user_query: str, field_values: dict[str, list[str]]) -> dict:
    """Send the user query to the LLM and return the structured filter JSON.

    Requires OPENROUTER_API_KEY environment variable.
    """
    prompt = build_prompt(user_query, field_values)

    response = httpx.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wraps its response.
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]  # drop opening fence line
        raw = raw.rsplit("```", 1)[0]  # drop closing fence
        raw = raw.strip()

    return json.loads(raw)


def filter_dataframe(df: pd.DataFrame, parsed: dict) -> pd.DataFrame:
    """Apply the structured filters from parse_query to a product DataFrame.

    Returns the filtered DataFrame. Fields set to null in the parsed dict are skipped.
    """
    mask = pd.Series(True, index=df.index)

    for field in FILTER_FIELDS:
        value = parsed.get(field)
        if value is None:
            continue

        if field not in df.columns:
            continue

        if isinstance(value, list):
            mask = mask & df[field].isin(value)
        else:
            mask = mask & (df[field] == value)

    return df[mask]
