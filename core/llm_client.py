# pyright: reportExplicitAny=false, reportUnknownVariableType=false, reportCallIssue=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportArgumentType=false
import json
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def get_client(
    base_url: str = "http://localhost:11434/v1/", api_key: str = "ollama"
) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


def extract_json_from_text(raw_text: str) -> dict[str, Any] | None:
    candidates = [raw_text.strip()]
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw_text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def generate_structured_output(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    response_model: type[T],
    thinking: bool = False,
) -> T:
    """Converts raw output to structured due to smaller models not supported native structured output"""
    schema_dict = response_model.model_json_schema()
    template_dict = {}
    for key, prop in schema_dict.get("properties", {}).items():
        template_dict[key] = f"<{prop.get('type', 'value')}>"

    template_str = json.dumps(template_dict, indent=2)

    schema_instruction = (
        f"\n\nIMPORTANT: You must respond ONLY with a raw, valid JSON object containing the actual data. "
        f"Do NOT output schema definitions (do not use 'properties', 'type', or 'title' keys). "
        f"Your output must exactly match this structure:\n{template_str}"
    )

    mod_messages = list(messages)
    if mod_messages:
        mod_messages[0] = {
            "role": mod_messages[0]["role"],
            "content": mod_messages[0]["content"] + schema_instruction,
        }

    response = client.chat.completions.create(
        model=model,
        messages=mod_messages,
        stream=False,
        response_format={"type": "json_object"},
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )

    content: str = response.choices[0].message.content or "{}"

    try:
        return response_model.model_validate_json(content)
    except ValidationError:
        parsed = extract_json_from_text(content)
        if parsed is not None:
            try:
                return response_model.model_validate(parsed)
            except ValidationError:
                pass

    raise ValueError(
        f"Failed to parse structured output into {response_model.__name__}. Raw: {content}"
    )
