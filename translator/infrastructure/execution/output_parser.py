"""Parsing of program stdout: checksum and self-reported execution time."""

from __future__ import annotations

import re

_STDOUT_TIME_RE = re.compile(r"Tiempo de ejecucion[^:]*:\s*([\d.]+)\s*ms", re.IGNORECASE)
_CHECKSUM_RE = re.compile(r"Checksum:\s*([\d.eE+\-]+)", re.IGNORECASE)


def parse_checksum(stdout: str) -> float | None:
    """Extract the float from a 'Checksum: <value>' line in stdout.

    Supports fixed and scientific notation. Returns None if not found.
    """
    for line in stdout.splitlines():
        m = _CHECKSUM_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None


def parse_stdout_time(stdout: str) -> float | None:
    """Extract the time in ms from a 'Tiempo de ejecucion ...: X.XX ms' line.

    Returns None if the line is not present or the value cannot be parsed.
    """
    for line in stdout.splitlines():
        m = _STDOUT_TIME_RE.search(line)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None
