import os
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ModelProvider(str, Enum):
    OLLAMA = "ollama"
    TOGETHER = "together"
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"
    GROQ = "groq"


@dataclass
class ModelConfig:
    name: str
    temperature: float
    provider: ModelProvider
    # Rough free-tier ceiling, used only to order fallbacks sensibly.
    context_window: int = 32_000


QWEN_2_5 = ModelConfig("qwen2.5:latest", 0.0, ModelProvider.OLLAMA)
EXAONE = ModelConfig("lgai/exaone-3-5-32b-instruct", 0.0, ModelProvider.TOGETHER)

# gemini-2.0-flash and gemini-2.0-flash-lite were confirmed working when this
# app was first built, but Google has since zeroed their free-tier quota
# ("limit: 0" on every call) in favor of newer generations. gemini-3.5-flash-lite
# is the current free-tier model as of 2026-07; the "-latest" aliases below
# track whichever generation Google currently points them at, so they serve as
# a fallback that shouldn't go stale the same way. Pin an exact version instead
# of an alias if you need reproducible output across a deployment's lifetime.
GEMINI_FLASH = ModelConfig("gemini-3.5-flash-lite", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH_LITE = ModelConfig("gemini-flash-lite-latest", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH_LATEST = ModelConfig("gemini-flash-latest", 0.0, ModelProvider.GEMINI, 1_000_000)
PPLX_SONAR = ModelConfig("sonar", 0.0, ModelProvider.PERPLEXITY)
LLAMA_3_3_70B = ModelConfig("llama-3.3-70b-versatile", 0.0, ModelProvider.GROQ, 128_000)


def _read_key_pool(*env_names: str) -> List[str]:
    """Collect API keys for one provider from the environment.

    Two forms are supported and combined, in this order:

      1. A comma-separated list in the base variable:
             GOOGLE_API_KEY=key_one,key_two,key_three
      2. Numbered siblings, which are easier to manage in a .env file:
             GOOGLE_API_KEY_1=key_one
             GOOGLE_API_KEY_2=key_two

    Duplicates are dropped while preserving order, so a key that appears in
    both forms is only ever tried once.
    """
    keys: List[str] = []
    for env_name in env_names:
        raw = os.getenv(env_name, "")
        keys.extend(part.strip() for part in raw.split(",") if part.strip())
        for index in range(1, 21):
            numbered = os.getenv(f"{env_name}_{index}", "").strip()
            if numbered:
                keys.append(numbered)

    seen = set()
    unique: List[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


@dataclass(frozen=True)
class ProviderCredentials:
    """The pool of interchangeable API keys available for a single provider."""

    provider: ModelProvider
    keys: List[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        # Ollama runs locally and needs no key, so an empty pool is still usable.
        return bool(self.keys) or self.provider == ModelProvider.OLLAMA


class Config:
    """Configuration for the application."""

    SEED = 42

    # The model tried first. Any provider in FALLBACK_MODELS is used only after
    # every key for the preceding model has been exhausted or rate-limited.
    MODEL = GEMINI_FLASH

    # Ordered fallbacks. Each entry must have credentials to be considered.
    FALLBACK_MODELS = [
        GEMINI_FLASH_LITE,
        GEMINI_FLASH_LATEST,
        LLAMA_3_3_70B,
        PPLX_SONAR,
        QWEN_2_5,
    ]

    OLLAMA_CONTEXT_WINDOW = 8192

    # How long a key is benched after the provider reports a rate limit.
    KEY_COOLDOWN_SECONDS = 60

    # Ceiling on a single model call, so an unreachable provider fails over
    # instead of hanging the app.
    REQUEST_TIMEOUT_SECONDS = 30

    # Agent loop limits. Each iteration is one LLM call, so this directly bounds
    # the request cost of a single user question.
    MAX_AGENT_ITERATIONS = 8

    # Truncation limits applied to tool output before it re-enters the prompt.
    MAX_TOOL_RESULT_CHARS = 4_000
    MAX_SQL_RESULT_ROWS = 50

    # Number of prior conversation turns kept in context. Older turns are
    # dropped so a long chat cannot grow the prompt without bound.
    MAX_HISTORY_MESSAGES = 12

    API_KEYS = {
        ModelProvider.GEMINI: _read_key_pool("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        ModelProvider.GROQ: _read_key_pool("GROQ_API_KEY"),
        ModelProvider.TOGETHER: _read_key_pool("TOGETHER_API_KEY"),
        ModelProvider.PERPLEXITY: _read_key_pool("PERPLEXITY_API_KEY", "PPLX_API_KEY"),
        ModelProvider.OLLAMA: [],
    }

    @classmethod
    def credentials(cls, provider: ModelProvider) -> ProviderCredentials:
        return ProviderCredentials(provider=provider, keys=list(cls.API_KEYS.get(provider, [])))

    @classmethod
    def model_chain(cls) -> List[ModelConfig]:
        """The primary model followed by every fallback that has credentials."""
        chain = [cls.MODEL]
        for model in cls.FALLBACK_MODELS:
            if model.provider == cls.MODEL.provider and model.name == cls.MODEL.name:
                continue
            chain.append(model)
        return [model for model in chain if cls.credentials(model.provider).available]

    # How many tables the database overview lists before summarising. Raise it
    # for very large schemas at the cost of more tokens per call.
    MAX_OVERVIEW_TABLES = 60

    # Similarity required before a mistyped table or column name is silently
    # corrected to a real one. Raising it makes corrections more conservative;
    # lowering it risks rewriting a query to the wrong table. See schema.py.
    NAME_MATCH_CUTOFF = 0.75

    # Database credentials are NOT read from configuration. They are entered in
    # the app and held in per-session memory only, so a deployed instance never
    # holds one user's credentials while serving another. See connection.py.
    #
    # For local convenience only, DATABASE_URL may pre-fill the connection form.
    # It is never required, and its value is not treated as a stored credential.
    DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", "")


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
