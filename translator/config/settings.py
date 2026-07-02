"""Constantes y resolución de configuración del traductor."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODELS: dict[str, str] = {
    "qwen3-coder": "qwen/qwen3-coder:free",
    "deepseek-r1": "deepseek/deepseek-r1",
    "kimi-k2.6": "moonshotai/kimi-k2.6:free",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "gpt-oss-120b": "openai/gpt-oss-120b:free",
    "gemma-4-31b": "google/gemma-4-31b-it:free",
    "llama-3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "nemotron-3-nano": "nvidia/nemotron-3-nano-30b-a3b:free",
}

DEFAULT_MODEL = "gpt-oss-120b"
DEFAULT_BACKEND_NAME = "OpenMP"

THINKING_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

MAX_RETRIES = 5
BENCHMARK_RUNS = 3
EXECUTION_TIMEOUT_SECONDS = 60
CHECKSUM_RTOL = 1e-3  # tolerancia relativa para comparar checksums entre serial y paralelo

PROMPTS_DIR = PROJECT_ROOT / "prompts"
ENV_PATH = PROJECT_ROOT / ".env"


def resolve_model(name: str) -> str:
    """Resuelve un alias corto o un ID completo al ID de OpenRouter."""
    if name in MODELS:
        return MODELS[name]
    if name in MODELS.values():
        return name
    known = ", ".join(sorted(MODELS))
    raise ValueError(
        f"Modelo desconocido: {name!r}. "
        f"Usa uno de: {known} o el ID completo de OpenRouter."
    )
