"""Punto de entrada CLI del traductor."""

import argparse
import logging
import os
import shlex
import sys
from pathlib import Path

from translator.config.env import load_env
from translator.config.settings import (
    DEFAULT_BACKEND_NAME,
    DEFAULT_MODEL,
    ENV_PATH,
    MODELS,
    THINKING_EFFORTS,
    resolve_model,
)
from translator.exceptions import TranslatorError
from translator.registry import discover_backends, resolve_backend
from translator.services.translate import translate_file

logger = logging.getLogger(__name__)


def parse_compile_flags(raw: str) -> list[str]:
    """Convierte la cadena de --flags en tokens de compilación."""
    return shlex.split(raw) if raw.strip() else []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Traduce código C/C++ serial a un backend paralelo usando un LLM vía OpenRouter.",
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Fichero fuente .c o .cpp a traducir",
    )
    parser.add_argument(
        "-b",
        "--backend",
        default=DEFAULT_BACKEND_NAME,
        metavar="NOMBRE",
        help=f"Backend de paralelización (por defecto: {DEFAULT_BACKEND_NAME})",
    )
    parser.add_argument(
        "--list-backends",
        action="store_true",
        help="Muestra los backends disponibles y sale",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        metavar="ALIAS",
        help=f"Modelo a usar (por defecto: {DEFAULT_MODEL}). Acepta alias corto o ID completo de OpenRouter.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Muestra los modelos disponibles y sale",
    )
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--thinking",
        action="store_true",
        help="Activa pensamiento extendido (reasoning) en la API",
    )
    thinking.add_argument(
        "--no-thinking",
        action="store_true",
        help="Desactiva pensamiento extendido (por defecto)",
    )
    parser.add_argument(
        "--thinking-effort",
        choices=THINKING_EFFORTS,
        metavar="NIVEL",
        help="Nivel de esfuerzo del reasoning (requiere --thinking)",
    )
    parser.add_argument(
        "--flags",
        type=str,
        default="",
        metavar="FLAGS",
        help=(
            "Flags extra antes del fichero fuente, como una cadena entre comillas "
            '(p. ej. "-DPOLYBENCH_TIME utilities/polybench.c")'
        ),
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help=(
            "Activa verificación de correctitud en bucle: si el checksum del programa "
            "traducido difiere del serial (o el programa falla en ejecución), reintenta "
            "con correcciones de errores de ejecución hasta agotar MAX_RETRIES."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Muestra la respuesta completa de la API de OpenRouter tras cada petición",
    )
    return parser


def _print_backends() -> None:
    backends = discover_backends()
    print("Backends disponibles:\n")
    for backend in sorted(backends.values(), key=lambda b: b.name):
        default = " (por defecto)" if backend.name == DEFAULT_BACKEND_NAME else ""
        print(f"  {backend.name:<14} [{backend.slug}]  prompts: {backend.prompts_dir}{default}")
    print()


def _print_models() -> None:
    print("Modelos disponibles:\n")
    for alias, model_id in sorted(MODELS.items()):
        default = " (por defecto)" if alias == DEFAULT_MODEL else ""
        print(f"  {alias:<20} → {model_id}{default}")
    print("\nTambién puedes pasar el ID completo de OpenRouter con -m/--model.")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(message)s",
    )

    if args.list_backends:
        _print_backends()
        return

    if args.list_models:
        _print_models()
        return

    if not args.source:
        parser.error("se requiere la ruta al fichero fuente (o usa --list-backends / --list-models)")

    if args.thinking_effort and not args.thinking:
        parser.error("--thinking-effort requiere --thinking")

    try:
        backend = resolve_backend(args.backend)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("Usa --list-backends para ver los backends disponibles.")
        sys.exit(1)

    try:
        model_id = resolve_model(args.model)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        print("Usa --list-models para ver los alias disponibles.")
        sys.exit(1)

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[ERROR] Fichero no encontrado: {source_path}")
        sys.exit(1)
    if source_path.suffix not in (".c", ".cpp"):
        print("[ERROR] Solo se admiten ficheros .c o .cpp")
        sys.exit(1)

    load_env(ENV_PATH)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY no encontrada en .env")
        sys.exit(1)

    thinking = args.thinking
    extra_flags = parse_compile_flags(args.flags)
    translated_name = backend.translated_path(source_path).name
    logger.info("\n\nTraduciendo %s -> %s", source_path.name, translated_name)
    logger.info("  Backend:    %s", backend.name)
    logger.info("  Modelo:     %s", model_id)
    if thinking:
        effort = args.thinking_effort or "medium (por defecto)"
        logger.info("  Reasoning:  activado (esfuerzo: %s)", effort)
    else:
        logger.info("  Reasoning:  desactivado")
    if extra_flags:
        logger.info("  Flags comp.: %s", " ".join(extra_flags))
    if args.compare:
        logger.info("  Compare:    activado (verificación de checksums en bucle)")
    if args.debug:
        logger.info("  Debug:      activado (respuesta completa de OpenRouter)")
    logger.info("\n")

    try:
        translate_file(
            source_path,
            api_key,
            backend=backend,
            model=model_id,
            thinking=thinking,
            thinking_effort=args.thinking_effort,
            extra_flags=extra_flags,
            compare=args.compare,
        )
    except TranslatorError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
