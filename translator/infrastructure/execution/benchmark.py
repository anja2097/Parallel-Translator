"""Benchmarking and correctness verification: serial vs translated binary."""

from __future__ import annotations

import logging
from pathlib import Path

from translator.config.settings import BENCHMARK_RUNS, CHECKSUM_RTOL, EXECUTION_TIMEOUT_SECONDS
from translator.domain.backend import Backend
from translator.exceptions import CompilationError
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
