"""Compilación OpenMP y serial genérica."""

from __future__ import annotations

from pathlib import Path

from translator.infrastructure.compilation.common import (
    executable_path,
    resolve_c_compiler,
    run_compile,
)


def compile_openmp(
    source_path: Path,
    *,
    serial: bool,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    compiler = resolve_c_compiler(source_path)
    exe = executable_path(source_path)
    flags = ["-O3", "-Wall", "-fopenmp", "-march=native"]
    extra = extra_flags or []
    label = "serial" if serial else "OpenMP"
    return run_compile(
        [compiler, *flags, *extra, str(source_path), "-o", str(exe), "-lm"],
        label=label,
    )
