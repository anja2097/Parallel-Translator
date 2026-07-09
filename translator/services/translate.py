"""Use case: translate a serial source file to a parallel backend with iterative error fixing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

from translator.config.settings import API_URL, MAX_RETRIES
from translator.domain.backend import Backend
from translator.exceptions import CompilationError, MaxRetriesExhaustedError
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.benchmark import run_and_check, verify_and_benchmark
from translator.infrastructure.execution.output_parser import parse_checksum
from translator.infrastructure.execution.process import run_once
from translator.infrastructure.llm.client import chat_completion
from translator.infrastructure.llm.code_extractor import extract_content, extract_message
from translator.infrastructure.llm.models import Message

logger = logging.getLogger(__name__)


class LLMConfig(NamedTuple):
    """Parameters forwarded to every chat_completion() call."""

    model: str
    api_url: str = API_URL
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


def _run_serial_reference(
    source_path: Path,
    backend: Backend,
    *,
    extra_flags: list[str] | None = None,
) -> tuple[str, float | None]:
    """Compile and run the serial binary once to obtain a reference stdout and checksum.

    Returns:
        (serial_stdout, serial_checksum)  — checksum is None when not printed by the program.

    Raises:
        CompilationError: if the serial source fails to compile.
    """
    logger.info("\n\n  Compilando versión serial para referencia…")
    ok, err = compile_file(source_path, backend, variant="serial", extra_flags=extra_flags)
    if not ok:
        raise CompilationError("No se pudo compilar el fichero original:", output=err)

    serial_exe = executable_path(source_path)
    serial_cmd = backend.run_serial_cmd(serial_exe)
    logger.info("  Ejecutando serial (referencia)…")
    serial_stdout, _serial_stderr, _elapsed = run_once(serial_exe, serial_cmd)
    if serial_exe.is_file():
        serial_exe.unlink()

    serial_checksum = parse_checksum(serial_stdout)
    if serial_checksum is not None:
        logger.info("  Checksum de referencia serial: %.6g", serial_checksum)
    else:
        logger.info("  (Checksum no disponible en serial — se comparará stdout exacto)")
    return serial_stdout, serial_checksum


def translate_file(
    source_path: Path,
    api_key: str,
    *,
    backend: Backend,
    model: str,
    api_url: str = API_URL,
    thinking: bool = False,
    thinking_effort: str | None = None,
    extra_flags: list[str] | None = None,
    compare: bool = False,
) -> None:
    """Translate a serial source file to the given backend, fixing errors iteratively.

    When *compare* is True the loop also verifies execution correctness: after each
    successful compilation the translated binary is run and its output is compared to
    the serial reference.  If the result is outside tolerance, ``fix_execution_errors.txt``
    is used to ask the LLM to fix runtime errors.  The total number of LLM iterations
    (compilation fixes + execution fixes) is bounded by MAX_RETRIES.
    """
    translated_path = backend.translated_path(source_path)
    source_code = source_path.read_text(encoding="utf-8")
    translate_prompt = _load_prompt(backend.prompts_dir / "translate.txt")
    fix_compile_prompt = _load_prompt(backend.prompts_dir / "fix_errors.txt")

    fix_exec_prompt: str | None = None
    if compare:
        fix_exec_prompt_path = backend.prompts_dir / "fix_execution_errors.txt"
        if not fix_exec_prompt_path.exists():
            logger.warning(
                "  [AVISO] --compare activo pero no se encontró %s. "
                "Se usará verify_and_benchmark() sin bucle de ejecución.",
                fix_exec_prompt_path,
            )
            compare = False
        else:
            fix_exec_prompt = _load_prompt(fix_exec_prompt_path)

    logger.info("\n\n  Original:  %s", source_path)
    logger.info("  Traducido: %s\n", translated_path)

    user_msg = {"role": "user", "content": f"{translate_prompt}\n\n{source_code}"}
    messages: list[dict] = [user_msg]

    llm_cfg = LLMConfig(
        model=model,
        api_url=api_url,
        thinking=thinking,
        thinking_effort=thinking_effort,
    )

    # ── Iteración 1: traducción inicial ──────────────────────────────────────
    data = chat_completion(
        api_key,
        messages,
        label="Iteración 1 — traducción inicial",
        model=llm_cfg.model,
        api_url=llm_cfg.api_url,
        thinking=llm_cfg.thinking,
        thinking_effort=llm_cfg.thinking_effort,
    )
    assistant_msg = extract_message(data)
    messages.append(_assistant_turn(assistant_msg))

    translated_code = extract_content(data)
    _write_source(translated_path, translated_code)

    ok, compile_error = compile_file(
        translated_path, backend, variant="translated", extra_flags=extra_flags,
    )

    # Obtain serial reference once (outside the retry loop) when compare is active.
    serial_stdout: str = ""
    serial_checksum: float | None = None
    serial_reference_obtained = False

    if ok and compare:
        serial_stdout, serial_checksum = _run_serial_reference(
            source_path, backend, extra_flags=extra_flags,
        )
        serial_reference_obtained = True

    if ok:
        exec_ok, exec_error = True, ""
        if compare:
            exec_ok, exec_error = run_and_check(
                source_path, backend,
                serial_stdout=serial_stdout,
                serial_checksum=serial_checksum,
                extra_flags=extra_flags,
            )
        if not compare or exec_ok:
            logger.info("\n\nCompilado correctamente en la iteración 1.")
            verify_and_benchmark(source_path, backend, extra_flags=extra_flags)
            return
        # Execution failed: fall into the retry loop with error_type="execution"
        error_type = "execution"
        last_error = exec_error
    else:
        error_type = "compilation"
        last_error = compile_error

    # ── Iteraciones 2..MAX_RETRIES: corrección de compilación o ejecución ────
    for iteration in range(2, MAX_RETRIES + 1):
        current_code = translated_path.read_text(encoding="utf-8")

        if error_type == "compilation":
            fix_user_content = (
                f"{fix_compile_prompt}\n\n{current_code}"
                f"\n\nCompilation errors:\n{last_error}"
            )
            label = f"Iteración {iteration} — corrección de compilación"
        else:
            fix_user_content = (
                f"{fix_exec_prompt}\n\n**Current code:**\n\n{current_code}"
                f"\n\n**Execution error details:**\n{last_error}"
            )
            label = f"Iteración {iteration} — corrección de errores de ejecución"

        messages.append({"role": "user", "content": fix_user_content})

        data = chat_completion(
            api_key,
            messages,
            label=label,
            model=llm_cfg.model,
            api_url=llm_cfg.api_url,
            thinking=llm_cfg.thinking,
            thinking_effort=llm_cfg.thinking_effort,
        )
        assistant_msg = extract_message(data)
        messages.append(_assistant_turn(assistant_msg))

        translated_code = extract_content(data)
        _write_source(translated_path, translated_code)

        ok, compile_error = compile_file(
            translated_path, backend, variant="translated", extra_flags=extra_flags,
        )
        if not ok:
            error_type = "compilation"
            last_error = compile_error
            continue

        # Compiled successfully.
        if not compare:
            logger.info("\n\nCompilado correctamente en la iteración %s.", iteration)
            verify_and_benchmark(source_path, backend, extra_flags=extra_flags)
            return

        # Obtain the serial reference on the first successful compile (if not yet done).
        if not serial_reference_obtained:
            serial_stdout, serial_checksum = _run_serial_reference(
                source_path, backend, extra_flags=extra_flags,
            )
            serial_reference_obtained = True

        exec_ok, exec_error = run_and_check(
            source_path, backend,
            serial_stdout=serial_stdout,
            serial_checksum=serial_checksum,
            extra_flags=extra_flags,
        )
        if exec_ok:
            logger.info("\n\nCompilado y verificado correctamente en la iteración %s.", iteration)
            verify_and_benchmark(source_path, backend, extra_flags=extra_flags)
            return

        error_type = "execution"
        last_error = exec_error

    if error_type == "compilation":
        raise MaxRetriesExhaustedError(
            f"No se pudo compilar tras {MAX_RETRIES} iteraciones. "
            f"Último error:\n{last_error}"
        )
    raise MaxRetriesExhaustedError(
        f"No se pudo corregir el resultado de ejecución tras {MAX_RETRIES} iteraciones. "
        f"Último error:\n{last_error}"
    )
