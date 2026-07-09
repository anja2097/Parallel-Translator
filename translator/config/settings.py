"""Constantes y resolución de configuración del traductor."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

API_URL = "https://openrouter.ai/api/v1/chat/completions"
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"

MODELS: dict[str, str] = {
    "qwen3-coder": "qwen/qwen3-coder",
    "deepseek-r1": "deepseek/deepseek-r1",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "kimi-k2.6": "moonshotai/kimi-k2.6:free",
    "nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b:free",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gemma-4-31b": "google/gemma-4-31b-it:free",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "nemotron-3-nano": "nvidia/nemotron-3-nano-30b-a3b:free",
    "sonnet-5": "anthropic/claude-sonnet-5",
}

HF_MODELS: dict[str, str] = {
    "qwen3-coder": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "qwen2.5-coder-32b": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gpt-oss-120b": "openai/gpt-oss-120b",
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


def resolve_model(name: str, provider: str = "openrouter") -> str:
    """Resuelve un alias corto o un ID completo al ID del proveedor indicado.

    Args:
        name: Alias corto o ID completo del modelo.
        provider: ``"openrouter"`` (por defecto) o ``"hf"`` para Hugging Face.
    """
    model_map = HF_MODELS if provider == "hf" else MODELS
    if name in model_map:
        return model_map[name]
    if name in model_map.values():
        return name
    known = ", ".join(sorted(model_map))
    provider_label = "Hugging Face" if provider == "hf" else "OpenRouter"
    raise ValueError(
        f"Modelo desconocido para {provider_label}: {name!r}. "
        f"Usa uno de: {known} o el ID completo."
    )
