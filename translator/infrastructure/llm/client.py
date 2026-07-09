"""HTTP client for the LLM API (OpenRouter or Hugging Face) with retry logic."""

from __future__ import annotations

import json
import logging
import time

import requests
from pydantic import ValidationError

from translator.config.settings import API_URL
from translator.exceptions import LLMError
from translator.infrastructure.llm.models import ChatCompletion, ErrorResponse

logger = logging.getLogger(__name__)

RETRY_WAIT_SECONDS = 10
_RECOVERABLE_CODES = {429, 502, 503, 504}


def _is_recoverable(code: int | str) -> bool:
    try:
        return int(code) in _RECOVERABLE_CODES
    except (ValueError, TypeError):
        return False


def _api_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _reasoning_payload(*, thinking: bool, effort: str | None) -> dict | None:
    if not thinking:
        return None
    reasoning: dict = {"enabled": True}
    if effort is not None:
        reasoning["effort"] = effort
    return reasoning


def _do_request(
    api_key: str,
    payload: dict,
    api_url: str = API_URL,
) -> ChatCompletion | tuple[int | str, str]:
    """Send a single HTTP request to the configured API endpoint.

    Returns ChatCompletion on success, or (code, message) tuple on error.
    Does not raise exceptions so the caller can retry on recoverable errors.
    """
    response = requests.post(
        api_url,
        headers=_api_headers(api_key),
        json=payload,
        timeout=120,
    )

    logger.info("\nHTTP %s", response.status_code)

    try:
        data = response.json()
    except json.JSONDecodeError:
        logger.info("Respuesta no JSON:")
        logger.debug("%s", response.text)
        logger.info("%s", response.text[:500])
        return response.status_code, "Respuesta no JSON"

    logger.debug(
        "\n[DEBUG] Respuesta completa de la API:\n%s",
        json.dumps(data, indent=2, ensure_ascii=False),
    )

    # Explicit error in the HTTP response (4xx/5xx)
    if not response.ok:
        logger.info("Error de API:")
        logger.info("%s", json.dumps(data, indent=2, ensure_ascii=False))
        return response.status_code, data.get("error", {}).get("message", "Error de API")

    # Error embedded in an HTTP 200 body (e.g. rate limit, provider down)
    if "error" in data:
        try:
            err = ErrorResponse.model_validate(data)
            return err.error.code, err.error.message
        except ValidationError:
            return 500, "Error desconocido en el cuerpo de la respuesta"

    try:
        return ChatCompletion.model_validate(data)
    except ValidationError as exc:
        logger.info("La respuesta de la API no tiene el formato esperado:")
        logger.info("%s", exc)
        return 500, "Formato de respuesta inesperado"


def chat_completion(
    api_key: str,
    messages: list[dict],
    *,
    model: str,
    api_url: str = API_URL,
    thinking: bool = False,
    thinking_effort: str | None = None,
    label: str = "",
) -> ChatCompletion:
    """Send a chat completion request with one automatic retry on recoverable errors.

    Args:
        api_key: API key for the selected provider.
        messages: Conversation history in OpenAI format.
        model: Model ID as expected by the provider.
        api_url: Endpoint URL (OpenRouter or Hugging Face).
        thinking: Whether to enable extended reasoning (OpenRouter only).
        thinking_effort: Reasoning effort level (requires ``thinking=True``).
        label: Human-readable label for log output.

    Raises:
        LLMError: On unrecoverable API errors.
    """
    payload: dict = {
        "model": model,
        "messages": messages,
    }
    reasoning = _reasoning_payload(thinking=thinking, effort=thinking_effort)
    if reasoning is not None:
        payload["reasoning"] = reasoning

    if label:
        logger.info("\n\n%s", "-" * 50)
        logger.info("  %s", label)
        logger.info("%s", "-" * 50)

    for attempt in (1, 2):
        result = _do_request(api_key, payload, api_url)

        if isinstance(result, ChatCompletion):
            return result

        code, message = result
        if attempt == 1 and _is_recoverable(code):
            logger.warning("[AVISO] Error recuperable (code %s): %s", code, message)
            logger.warning("        Reintentando en %s s…", RETRY_WAIT_SECONDS)
            time.sleep(RETRY_WAIT_SECONDS)
            continue

        raise LLMError(f"API: {message} (code {code})", code=code)

    raise LLMError("API: error irrecuperable tras reintentos")  # pragma: no cover
