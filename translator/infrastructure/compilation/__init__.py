"""Compilation façade: dispatches to the appropriate backend toolchain."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from translator.domain.backend import Backend
from translator.infrastructure.compilation import ghighway, go, kokkos, mpi, openmp
from translator.infrastructure.compilation.common import executable_path

# Maps each backend slug to its compile function.
_COMPILERS: dict[str, Callable[[], tuple[bool, str]]] = {}


def compile_serial(
    source_path: Path,
    *,
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile the serial source file using OpenMP flags (without -fopenmp at link time)."""
    return openmp.compile_openmp(source_path, serial=True, extra_flags=extra_flags or [])


def compile_file(
    source_path: Path,
    backend: Backend,
    *,
    variant: str = "translated",
    extra_flags: list[str] | None = None,
) -> tuple[bool, str]:
    """Compile a source file for the given backend.

    Args:
        variant: ``"serial"`` compiles the original with serial flags;
                 ``"translated"`` uses the backend-specific toolchain.
    """
    if variant == "serial":
        return compile_serial(source_path, extra_flags=extra_flags)

    extra = extra_flags or []
    dispatch: dict[str, Callable[[], tuple[bool, str]]] = {
        "openmp": lambda: openmp.compile_openmp(source_path, serial=False, extra_flags=extra),
        "kokkos": lambda: kokkos.compile_kokkos(source_path, extra_flags=extra),
        "mpi": lambda: mpi.compile_mpi(source_path, extra_flags=extra),
        "go": lambda: go.compile_go(source_path),
        "ghighway": lambda: ghighway.compile_ghighway(source_path, extra_flags=extra),
    }
    compile_fn = dispatch.get(backend.slug)
    if compile_fn is None:
        raise ValueError(f"Backend sin toolchain de compilación: {backend.name!r}")
    return compile_fn()


__all__ = ["compile_file", "compile_serial", "executable_path"]
