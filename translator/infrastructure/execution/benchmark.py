"""Benchmarking and correctness verification: serial vs translated binary."""

from __future__ import annotations

import logging
from pathlib import Path

from translator.config.settings import BENCHMARK_RUNS, CHECKSUM_RTOL, EXECUTION_TIMEOUT_SECONDS
from translator.domain.backend import Backend
from translator.exceptions import CompilationError, ExecutionError, ExecutionTimeoutError
from translator.infrastructure.compilation import compile_file, executable_path
from translator.infrastructure.execution.output_parser import parse_checksum, parse_stdout_time
from translator.infrastructure.execution.process import run_once

logger = logging.getLogger(__name__)


def _format_execute_cmd(cmd: list[str], env: dict[str, str] | None = None) -> str:
    parts = [f"{key}={value}" for key, value in (env or {}).items()]
    parts.extend(cmd)
    return " ".join(parts)


def _log_execute_header(
    label: str,
    cmd: list[str],
    runs: int,
    *,
    env: dict[str, str] | None = None,
) -> None:
    logger.info("\n\n  Ejecutando (%s): %s", label, _format_execute_cmd(cmd, env))
    logger.info("  Límite por ejecución: %s s", EXECUTION_TIMEOUT_SECONDS)
    if runs > 1:
        logger.info("  Repeticiones: %s", runs)


def _cleanup_executables(*exe_paths: Path) -> None:
    removed: list[Path] = []
    for exe_path in exe_paths:
        if exe_path.is_file():
            exe_path.unlink()
            removed.append(exe_path)
    if removed:
        logger.info("\n  Ejecutables eliminados:")
        for exe_path in removed:
            logger.info("    %s", exe_path)


def benchmark(
    exe_path: Path,
    cmd: list[str],
    runs: int,
    *,
    label: str,
    env: dict[str, str] | None = None,
) -> tuple[str, str, float, float | None]:
    """Run the binary ``runs`` times and return (stdout, stderr, wall_avg_s, stdout_avg_ms).

    - wall_avg_s: mean wall-clock time measured from Python (seconds).
    - stdout_avg_ms: mean internal time reported by the program (ms),
                     or None if the program does not print 'Tiempo de ejecucion'.
    """
    _log_execute_header(label, cmd, runs, env=env)
    stdout, stderr = "", ""
    wall_total = 0.0
    stdout_total = 0.0
    stdout_time_available = True
    for _ in range(runs):
        stdout, stderr, elapsed = run_once(exe_path, cmd, env=env)
        wall_total += elapsed
        parsed = parse_stdout_time(stdout)
        if parsed is None:
            stdout_time_available = False
        else:
            stdout_total += parsed
    logger.info("  Resultado: ejecución OK")
    stdout_avg = (stdout_total / runs) if stdout_time_available else None
    return stdout, stderr, wall_total / runs, stdout_avg


def run_and_check(
    original_path: Path,
    backend: Backend,
    *,
    serial_stdout: str,
    serial_checksum: float | None,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Execute the translated binary once and compare its output against the serial reference.

    This function is designed for the ``--compare`` feedback loop: it runs the translated
    binary a single time (not benchmarking), captures any runtime failure, and returns a
    structured error description suitable for feeding back to the LLM.

    Args:
        original_path: Path to the original serial source file.
        backend: The active backend.
        serial_stdout: Captured stdout of the serial reference run.
        serial_checksum: Pre-parsed checksum from the serial run (or None if absent).
        extra_flags: Extra compilation/run flags.

    Returns:
        ``(True, "")`` if the translated binary ran successfully and its output is within
        tolerance of the serial reference.
        ``(False, error_description)`` otherwise, where *error_description* contains
        enough context (exit signal, stderr, checksum values) for the LLM to diagnose the
        problem.
    """
    translated_path = backend.translated_path(original_path)
    translated_exe = executable_path(translated_path)
    translated_cmd = backend.run_cmd(translated_exe)

    _log_execute_header(backend.name, translated_cmd, 1, env=backend.run_env())

    try:
        trans_stdout, trans_stderr, _ = run_once(
            translated_exe, translated_cmd, env=backend.run_env(),
        )
    except ExecutionTimeoutError as exc:
        error_info = (
            f"The translated program exceeded the execution timeout "
            f"({EXECUTION_TIMEOUT_SECONDS} s).\n"
            f"This usually indicates an infinite loop, excessive computational load, or a "
            f"deadlock in the parallel code.\n\n"
            f"Error details:\n{exc}"
        )
        logger.info("  Resultado: timeout de ejecución")
        return False, error_info
    except ExecutionError as exc:
        error_info = (
            f"The translated program crashed at runtime (non-zero exit code).\n"
            f"Common causes: segmentation fault, stack overflow, assertion failure, "
            f"MPI error, or unhandled exception.\n\n"
            f"Error details:\n{exc}"
        )
        logger.info("  Resultado: error de ejecución (crash)")
        return False, error_info

    logger.info("  Resultado: ejecución OK")

    trans_checksum = parse_checksum(trans_stdout)

    if serial_checksum is not None and trans_checksum is not None:
        denom = max(abs(serial_checksum), 1e-10)
        rel_diff = abs(serial_checksum - trans_checksum) / denom
        logger.info("  Checksum serial:     %.6g", serial_checksum)
        logger.info("  Checksum %-10s %.6g", backend.name, trans_checksum)
        if rel_diff <= CHECKSUM_RTOL:
            logger.info(
                "  Diferencia relativa: %.2e  (dentro de tolerancia %.0e)",
                rel_diff, CHECKSUM_RTOL,
            )
            return True, ""
        logger.warning(
            "  Diferencia relativa: %.2e  [AVISO] fuera de tolerancia (%.0e)",
            rel_diff, CHECKSUM_RTOL,
        )
        error_info = (
            f"The translated program produced an incorrect numerical result.\n"
            f"The checksum differs from the serial reference beyond the allowed tolerance "
            f"({CHECKSUM_RTOL:.0e} relative).\n\n"
            f"Serial checksum:     {serial_checksum:.6g}\n"
            f"Translated checksum: {trans_checksum:.6g}\n"
            f"Relative difference: {rel_diff:.2e}\n\n"
            f"This is typically caused by data races, incorrect reduction operations, "
            f"uninitialized memory, or incorrect parallelization boundaries."
        )
        if trans_stderr:
            error_info += f"\n\nProgram stderr:\n{trans_stderr}"
        return False, error_info

    # No checksum available — fall back to exact stdout comparison
    if serial_stdout == trans_stdout:
        logger.info("  Salida idéntica (stdout)")
        return True, ""

    logger.warning("  [AVISO] stdout difiere del serial")
    error_info = (
        f"The translated program produced different output than the serial reference.\n\n"
        f"Expected stdout (serial):\n{serial_stdout!r}\n\n"
        f"Actual stdout (translated):\n{trans_stdout!r}"
    )
    if trans_stderr:
        error_info += f"\n\nProgram stderr:\n{trans_stderr}"
    return False, error_info


def verify_and_benchmark(
    original_path: Path,
    backend: Backend,
    *,
    extra_flags: list[str] | None = None,
) -> None:
    """Compile and run both the serial original and the translated version, then report results."""
    translated_path = backend.translated_path(original_path)
    serial_exe = executable_path(original_path)
    translated_exe = executable_path(translated_path)

    logger.info("\n\n%s", "=" * 50)
    logger.info("  Verificación: serial vs %s", backend.name)
    logger.info("%s", "=" * 50)

    logger.info("\n\n  Compilando versión serial…")
    ok, err = compile_file(
        original_path, backend, variant="serial", extra_flags=extra_flags,
    )
    if not ok:
        raise CompilationError("No se pudo compilar el fichero original:", output=err)

    runs = BENCHMARK_RUNS
    serial_cmd = backend.run_serial_cmd(serial_exe)
    translated_cmd = backend.run_cmd(translated_exe)

    serial_stdout, serial_stderr, serial_wall, serial_stdout_ms = benchmark(
        serial_exe, serial_cmd, runs, label="serial",
    )

    trans_stdout, trans_stderr, trans_wall, trans_stdout_ms = benchmark(
        translated_exe,
        translated_cmd,
        runs,
        label=backend.name,
        env=backend.run_env(),
    )

    logger.info("\n\n%s", "-" * 50)
    logger.info("  Verificación de correctitud")
    logger.info("%s", "-" * 50)

    serial_checksum = parse_checksum(serial_stdout)
    trans_checksum = parse_checksum(trans_stdout)

    if serial_checksum is not None and trans_checksum is not None:
        denom = max(abs(serial_checksum), 1e-10)
        rel_diff = abs(serial_checksum - trans_checksum) / denom
        logger.info("  Checksum serial:     %.6g", serial_checksum)
        logger.info("  Checksum %-10s %.6g", backend.name, trans_checksum)
        if rel_diff <= CHECKSUM_RTOL:
            logger.info(
                "  Diferencia relativa: %.2e  (dentro de tolerancia %.0e)",
                rel_diff, CHECKSUM_RTOL,
            )
        else:
            logger.warning(
                "  Diferencia relativa: %.2e  [AVISO] fuera de tolerancia (%.0e)",
                rel_diff, CHECKSUM_RTOL,
            )
    else:
        logger.info("  (Checksum no disponible — comparación exacta de stdout)")
        stdout_match = serial_stdout == trans_stdout
        if stdout_match:
            logger.info("  Salida idéntica (stdout)")
        else:
            logger.warning("  [AVISO] stdout difiere:")
            logger.warning("     serial:    %r", serial_stdout)
            logger.warning("     %s: %r", backend.slug, trans_stdout)

    stderr_match = serial_stderr == trans_stderr
    if not stderr_match:
        logger.warning("  [AVISO] stderr difiere:")
        logger.warning("     serial:    %r", serial_stderr)
        logger.warning("     %s: %r", backend.slug, trans_stderr)

    # ── Wall-clock times (full process measured from Python) ──────────────
    logger.info("\n\n%s", "-" * 50)
    logger.info("  Tiempos de ejecución (proceso completo)")
    logger.info("%s", "-" * 50)
    logger.info("  Serial      (media de %s): %.3f ms", runs, serial_wall * 1000)
    logger.info("  %-10s (media de %s): %.3f ms", backend.name, runs, trans_wall * 1000)

    if trans_wall > 0:
        speedup_wall = serial_wall / trans_wall
        logger.info("  Speedup: %.2fx", speedup_wall)
        if speedup_wall < 1.0:
            logger.info("  (El paralelo es más lento. Puede ser debido a poca carga computacional)")
    else:
        logger.info("  Speedup: N/A (tiempo traducido ~ 0)")

    # ── Self-reported times from the program ───────────────────────────────
    logger.info("\n\n%s", "-" * 50)
    logger.info("  Tiempos de ejecución (reportados por el programa)")
    logger.info("%s", "-" * 50)
    if serial_stdout_ms is not None and trans_stdout_ms is not None:
        logger.info("  Serial      (media de %s): %.3f ms", runs, serial_stdout_ms)
        logger.info("  %-10s (media de %s): %.3f ms", backend.name, runs, trans_stdout_ms)
        if trans_stdout_ms > 0:
            speedup_stdout = serial_stdout_ms / trans_stdout_ms
            logger.info("  Speedup: %.2fx", speedup_stdout)
            if speedup_stdout < 1.0:
                logger.info(
                    "  (El paralelo es más lento. Puede ser debido a poca carga computacional)"
                )
        else:
            logger.info("  Speedup: N/A (tiempo traducido ~ 0)")
    else:
        logger.info("  (No disponible: el programa no imprime 'Tiempo de ejecucion ... ms')")

    _cleanup_executables(serial_exe, translated_exe)

    logger.info("\n%s\n", "=" * 50)
