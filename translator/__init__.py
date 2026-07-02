"""Traductor serial → backend paralelo (OpenMP, Kokkos, MPI)."""

from translator.cli.main import main
from translator.services.translate import translate_file

__all__ = ["main", "translate_file"]
