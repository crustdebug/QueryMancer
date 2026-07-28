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
# ("limit: 0" on every call) in favor of newer generations.
#
# gemini-2.5-flash-lite returns HTTP 200 on a raw REST call but consistently
# 404s ("no longer available to new users") through the actual chat-completion
# path used here - confirmed by hitting it live twice, minutes apart, both
# times 404. It is kept in the chain (RotatingChatModel now falls through a
# 404 to the next model rather than crashing) but deliberately NOT listed
# first, so a normal request doesn't eat a wasted round trip on it every time.
# gemini-3.1-flash-lite is listed first instead: confirmed working live.
# The "-latest" alias at the tail tracks whichever generation Google currently
# points it at, so it serves as a fallback that shouldn't go stale the same
# way a pinned version eventually does. Pin an exact version instead of an
# alias if you need reproducible output across a deployment's lifetime.
GEMINI_FLASH_LITE_31 = ModelConfig("gemini-3.1-flash-lite", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH = ModelConfig("gemini-2.5-flash", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH_LITE = ModelConfig("gemini-2.5-flash-lite", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH_LATEST = ModelConfig("gemini-flash-latest", 0.0, ModelProvider.GEMINI, 1_000_000)
PPLX_SONAR = ModelConfig("sonar", 0.0, ModelProvider.PERPLEXITY)

# Groq free-tier models capable of the multi-step tool calling this agent
# relies on, smallest/cheapest first for the same reason as the Gemini order
# above: llama-3.1-8b-instant has Groq's highest free-tier RPM, so it takes
# most of the traffic and leaves gpt-oss-20b's quota for when 8b is limited.
GROQ_LLAMA_3_1_8B = ModelConfig("llama-3.1-8b-instant", 0.0, ModelProvider.GROQ, 128_000)
GROQ_GPT_OSS_20B = ModelConfig("openai/gpt-oss-20b", 0.0, ModelProvider.GROQ, 131_000)
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
    MODEL = GEMINI_FLASH_LITE_31

    # Ordered fallbacks. Each entry must have credentials to be considered.
    FALLBACK_MODELS = [
        GEMINI_FLASH,
        GEMINI_FLASH_LITE,
        GEMINI_FLASH_LATEST,
        GROQ_LLAMA_3_1_8B,
        GROQ_GPT_OSS_20B,
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
