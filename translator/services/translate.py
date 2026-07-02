"""Use case: translate a serial source file to a parallel backend with iterative error fixing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from translator.config.settings import MAX_RETRIES
from translator.domain.backend import Backend
from translator.exceptions import MaxRetriesExhaustedError
from translator.infrastructure.compilation import compile_file
from translator.infrastructure.execution.benchmark import verify_and_benchmark
from translator.infrastructure.llm.client import chat_completion
from translator.infrastructure.llm.code_extractor import extract_content, extract_message
from translator.infrastructure.llm.models import Message

logger = logging.getLogger(__name__)


class LLMConfig(NamedTuple):
    """Parameters forwarded to every chat_completion() call."""

    model: str
    thinking: bool = False
    thinking_effort: str | None = None


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _assistant_turn(message: Message) -> dict:
    turn: dict = {
        "role": "assistant",
        "content": message.content,
    }
    if message.reasoning_details is not None:
        turn["reasoning_details"] = [rd.model_dump() for rd in message.reasoning_details]
    return turn


def _write_source(target_path: Path, code: str) -> None:
    target_path.write_text(code, encoding="utf-8")
    logger.info("  Fichero actualizado: %s", target_path)


def _on_compile_success(
    original_path: Path,
    backend: Backend,
    iteration: int,
    *,
    extra_flags: list[str] | None = None,
) -> None:
    logger.info("\n\nCompilado correctamente en la iteración %s.", iteration)
    verify_and_benchmark(original_path, backend, extra_flags=extra_flags)


def translate_file(
    source_path: Path,
    api_key: str,
    *,
    backend: Backend,
    model: str,
    thinking: bool = False,
    thinking_effort: str | None = None,
    extra_flags: list[str] | None = None,
) -> None:
    """Translate a serial source file to the given backend, fixing compilation errors iteratively."""
    translated_path = backend.translated_path(source_path)
    source_code = source_path.read_text(encoding="utf-8")
    translate_prompt = _load_prompt(backend.prompts_dir / "translate.txt")
    fix_prompt = _load_prompt(backend.prompts_dir / "fix_errors.txt")

    logger.info("\n\n  Original:  %s", source_path)
    logger.info("  Traducido: %s\n", translated_path)

    user_msg = {"role": "user", "content": f"{translate_prompt}\n\n{source_code}"}
    messages: list[dict] = [user_msg]

    llm_cfg = LLMConfig(
        model=model,
        thinking=thinking,
        thinking_effort=thinking_effort,
    )

    data = chat_completion(
        api_key,
        messages,
        label="Iteración 1 — traducción inicial",
        model=llm_cfg.model,
        thinking=llm_cfg.thinking,
        thinking_effort=llm_cfg.thinking_effort,
    )
    assistant_msg = extract_message(data)
    messages.append(_assistant_turn(assistant_msg))

    translated_code = extract_content(data)
    _write_source(translated_path, translated_code)

    ok, error_output = compile_file(
        translated_path, backend, variant="translated", extra_flags=extra_flags,
    )
    if ok:
        _on_compile_success(source_path, backend, 1, extra_flags=extra_flags)
        return

    for iteration in range(2, MAX_RETRIES + 1):
        current_code = translated_path.read_text(encoding="utf-8")
        fix_user_content = (
            f"{fix_prompt}\n\n{current_code}\n\nCompilation errors:\n{error_output}"
        )
        messages.append({"role": "user", "content": fix_user_content})

        data = chat_completion(
            api_key,
            messages,
            label=f"Iteración {iteration} — corrección de errores",
            model=llm_cfg.model,
            thinking=llm_cfg.thinking,
            thinking_effort=llm_cfg.thinking_effort,
        )
        assistant_msg = extract_message(data)
        messages.append(_assistant_turn(assistant_msg))

        translated_code = extract_content(data)
        _write_source(translated_path, translated_code)

        ok, error_output = compile_file(
            translated_path, backend, variant="translated", extra_flags=extra_flags,
        )
        if ok:
            _on_compile_success(
                source_path, backend, iteration, extra_flags=extra_flags,
            )
            return

    raise MaxRetriesExhaustedError(
        f"No se pudo compilar tras {MAX_RETRIES} iteraciones. Último error:\n{error_output}"
    )
