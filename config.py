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
GEMINI_FLASH = ModelConfig("gemini-2.0-flash", 0.0, ModelProvider.GEMINI, 1_000_000)
GEMINI_FLASH_LITE = ModelConfig("gemini-2.0-flash-lite", 0.0, ModelProvider.GEMINI, 1_000_000)
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
    FALLBACK_MODELS = [GEMINI_FLASH_LITE, LLAMA_3_3_70B, PPLX_SONAR, QWEN_2_5]

    OLLAMA_CONTEXT_WINDOW = 8192

    # How long a key is benched after the provider reports a rate limit.
    KEY_COOLDOWN_SECONDS = 60

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

    # Tables the agent is permitted to read. An empty list means every table in
    # the 'public' schema is allowed. This is enforced in tools.py, not merely
    # displayed in the UI.
    ALLOWED_TABLES = [
        "advance",
        "ap_set_up",
        "arap_set_up",
        "asset",
        "asset_depreciation_entry",
        "bank_account",
        "bank_gurantee",
        "budget",
        "budget_adjustment",
        "budget_coa_mapping",
        "budget_set_up",
        "chart_of_accounts",
        "cheque_book",
        "cost_center",
        "countries",
        "credit_note",
        "credit_note_summary",
        "currency_chart",
        "customer_invoice",
        "customer_invoice_summary",
        "customer_profile",
        "debit_note",
        "debit_note_summary",
        "department",
        "depreciation",
        "depreciation_setup",
        "employee",
        "employee_expense",
        "employee_invoice_summary",
        "financial_year",
        "gst_return",
        "gst_settlement",
        "gst_setup",
        "historical_pl_bs",
        "hsnsac",
        "invoice_tax",
        "invoice_terms_and_annexure",
        "journal",
        "journal_entry",
        "journal_entry_breakdown",
        "petty_cash",
        "project",
        "provisions",
        "regions",
        "roc_return",
        "salary_invoice",
        "salary_invoice_summary",
        "sop",
        "standard_coa_mapping",
        "subregions",
        "tax_settlement_entry",
        "tds_return",
        "tds_setup",
        "vendor_bank",
        "vendor_categories",
        "vendor_detail",
        "vendor_invoice",
        "vendor_invoice_deduction",
        "vendor_invoice_reallocation",
        "vendor_invoice_summary",
        "vendor_profile",
        "vendor_tax",
    ]

    class Postgres:
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")


def seed_everything(seed: int = Config.SEED):
    random.seed(seed)
