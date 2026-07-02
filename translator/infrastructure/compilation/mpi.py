"""Compilación con MPI."""

from __future__ import annotations

import shutil
from pathlib import Path

from translator.exceptions import CompilerNotFoundError
from translator.infrastructure.compilation.common import executable_path, run_compile


def _mpi_compiler(source_path: Path) -> str:
    if source_path.suffix == ".cpp":
        candidates = ["mpicxx", "mpiCC", "mpic++"]
    else:
        candidates = ["mpicc"]
    for c in candidates:
        if shutil.which(c):
            return c
    raise CompilerNotFoundError("No se encontró compilador MPI (mpicc/mpicxx) en PATH.")


def compile_mpi(
    source_path: Path,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    compiler = _mpi_compiler(source_path)
    exe = executable_path(source_path)
    extra = extra_flags or []
    cmd = [compiler, "-O2", "-Wall", *extra, str(source_path), "-o", str(exe), "-lm"]
    return run_compile(cmd, label="MPI")
