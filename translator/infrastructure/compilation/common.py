"""Shared utilities for C/C++ compilation."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from translator.exceptions import CompilerNotFoundError

logger = logging.getLogger(__name__)


@dataclass
class CompilationResult:
    """Return value of every compile_* function."""

    success: bool
    error_output: str = ""

    def __bool__(self) -> bool:
        return self.success

    def as_tuple(self) -> tuple[bool, str]:
        """Compatibility helper for callers that still unpack (ok, err)."""
        return self.success, self.error_output


def resolve_c_compiler(source_path: Path, *, cxx: bool = False) -> str:
    if cxx or source_path.suffix == ".cpp":
        candidates = ["g++", "g++-15"]
    else:
        candidates = ["gcc", "gcc-15"]
    for c in candidates:
        if shutil.which(c):
            return c
    raise CompilerNotFoundError("No se encontró compilador C/C++ en PATH.")


def run_compile(cmd: list[str], *, label: str) -> tuple[bool, str]:
    logger.info("\n\n  Compilando (%s): %s", label, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if result.returncode == 0:
        logger.info("  Resultado: compilación OK")
        return True, ""
    logger.info("  Resultado: error de compilación")
    return False, output


def executable_path(source_path: Path) -> Path:
    return source_path.with_suffix("")
