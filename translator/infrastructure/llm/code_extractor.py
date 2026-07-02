"""Extraction of source code from LLM responses."""

from __future__ import annotations

from translator.exceptions import LLMEmptyResponseError
from translator.infrastructure.llm.models import ChatCompletion, Message

_CODE_START_PREFIXES = (
    "#",           # C/C++ preprocessor (#include, #define, …)
    "package ",    # Go
    "import ",     # Go (file without package, e.g. tests)
    "using ",      # C++
    "namespace ",  # C++
)


def _resolve_message_text(message: Message) -> str:
    """Return the text content of an LLM message; some models leave content=null."""
    if message.content and message.content.strip():
        return message.content

    for val in (message.reasoning, message.reasoning_content):
        if val and val.strip():
            return val

    if message.reasoning_details:
        for item in reversed(message.reasoning_details):
            if item.type == "reasoning.text" and item.text and item.text.strip():
                return item.text

    return ""


def strip_code_fences(content: str | None) -> str:
    """Remove opening and closing ``` fences if the LLM wrapped the code in them."""
    if not content:
        return ""
    lines = content.splitlines()
    if not lines or not lines[0].strip().startswith("```"):
        return content
    if len(lines) >= 2 and lines[-1].strip().startswith("```"):
        return "\n".join(lines[1:-1])
    return "\n".join(lines[1:])


def strip_leading_preamble(content: str) -> str:
    """Drop leading lines until the first line that looks like source code."""
    lines = content.splitlines()
    while lines:
        stripped = lines[0].lstrip()
        if any(stripped.startswith(prefix) for prefix in _CODE_START_PREFIXES):
            break
        lines.pop(0)
    return "\n".join(lines)


def strip_trailing_fence(content: str) -> str:
    """Remove a trailing closing ``` line if it was left after stripping."""
    lines = content.splitlines()
    if lines and lines[-1].strip().startswith("```"):
        lines.pop()
    return "\n".join(lines)


def extract_code(content: str | None) -> str:
    """Apply all stripping steps to produce clean source code."""
    code = strip_code_fences(content)
    code = strip_leading_preamble(code)
    return strip_trailing_fence(code)


def extract_message(data: ChatCompletion) -> Message:
    """Return the first choice message from a ChatCompletion response."""
    return data.choices[0].message


def extract_content(data: ChatCompletion) -> str:
    """Extract and clean source code from a ChatCompletion response.

    Raises:
        LLMEmptyResponseError: If the response is empty or contains no recognisable code.
    """
    message = extract_message(data)
    raw = _resolve_message_text(message)
    if not raw:
        raise LLMEmptyResponseError(
            "El modelo devolvió una respuesta vacía (content=null).\n"
            "        Prueba otro modelo o reintenta la traducción."
        )
    code = extract_code(raw)
    if not code.strip():
        raise LLMEmptyResponseError(
            "No se pudo extraer código de la respuesta del modelo.\n"
            "        La respuesta no contenía código reconocible tras el post-procesado."
        )
    return code
