"""Backend registry — discovers available backends and resolves them by name."""

from __future__ import annotations

from pathlib import Path

from translator.config.env import get_env
from translator.config.settings import PROMPTS_DIR
from translator.domain.backend import Backend

# Extension produced by each backend slug (default: inherit from source file).
# Slugs absent from this map use the source extension unchanged.
_EXTENSION_BY_SLUG: dict[str, str] = {
    "kokkos": ".cpp",
    "ghighway": ".cpp",
    "go": ".go",
}

# Extra environment variables injected at runtime for specific backends.
_RUN_ENV_BY_SLUG: dict[str, dict[str, str]] = {
    "kokkos": {
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
    },
}


def _build_backend(subdir: Path) -> Backend:
    """Construct a Backend from a prompts sub-directory, injecting runtime config."""
    name = subdir.name
    slug = name.lower()
    extension = _EXTENSION_BY_SLUG.get(slug, "")  # "" means inherit source extension

    run_prefix: list[str] = []
    if slug == "mpi":
        nprocs = get_env("MPI_PROCS", "4")
        run_prefix = ["mpirun", "-np", nprocs]

    return Backend(
        name=name,
        slug=slug,
        prompts_dir=subdir,
        extension=extension,
        run_prefix=run_prefix,
        run_env_vars=_RUN_ENV_BY_SLUG.get(slug, {}),
    )


def discover_backends(prompts_dir: Path = PROMPTS_DIR) -> dict[str, Backend]:
    """Scan a prompts directory and return backends that have both required prompt files."""
    backends: dict[str, Backend] = {}
    if not prompts_dir.is_dir():
        return backends
    for subdir in sorted(prompts_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if (subdir / "translate.txt").exists() and (subdir / "fix_errors.txt").exists():
            backend = _build_backend(subdir)
            backends[backend.slug] = backend
    return backends


class BackendRegistry:
    """Lazy registry that discovers backends once and caches the result."""

    def __init__(self, prompts_dir: Path = PROMPTS_DIR) -> None:
        self._prompts_dir = prompts_dir
        self._cache: dict[str, Backend] | None = None

    def _load(self) -> dict[str, Backend]:
        if self._cache is None:
            self._cache = discover_backends(self._prompts_dir)
        return self._cache

    def all(self) -> dict[str, Backend]:
        """Return all discovered backends keyed by slug."""
        return self._load()

    def resolve(self, name: str) -> Backend:
        """Resolve a backend name (case-insensitive) to a Backend instance.

        Raises:
            ValueError: If the name does not match any known backend.
        """
        backends = self._load()
        key = name.lower()
        if key in backends:
            return backends[key]
        known = ", ".join(b.name for b in backends.values())
        raise ValueError(f"Backend desconocido: {name!r}. Usa uno de: {known}")


# Module-level default registry used by CLI and tests.
_default_registry = BackendRegistry()


def resolve_backend(name: str) -> Backend:
    """Resolve a backend name using the default registry."""
    return _default_registry.resolve(name)
