import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class ModelProvider(str, Enum):
    OLLAMA="ollama"
    TOGETHER="together"
    GEMINI="gemini"
    PERPLEXITY="perplexity"

@dataclass
class ModelConfig:
    name:str
    temperature: float
    provider: ModelProvider


QWEN_2_5 = ModelConfig("qwen2.5:latest",0.0,ModelProvider.OLLAMA)
EXAONE = ModelConfig("lgai/exaone-3-5-32b-instruct", 0.0, ModelProvider.TOGETHER)
GEMINI_FLASH = ModelConfig("gemini-2.0-flash", 0.0, ModelProvider.GEMINI)
PPLX_7B_ONLINE = ModelConfig("pplx-7b-online", 0.0, ModelProvider.PERPLEXITY)

class Config:
    """
    Configuration class for the application.
    """
    SEED = 42
    MODEL = GEMINI_FLASH
    OLLAMA_CONTEXT_WINDOW = 1024
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    """
    A list of table names that the model is allowed to access.
    If the list is empty, all tables in the database will be accessible.
    Example: ["users", "products"]
    """
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
    "vendor_tax"
    ]

    class Postgres:
        dbname = os.getenv("POSTGRES_DB")
        user = os.getenv("POSTGRES_USER")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT")

def seed_everything(seed: int = Config.SEED):
    random.seed(seed)