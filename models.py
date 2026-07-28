import os
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_together import ChatTogether
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatPerplexity

from config import Config, ModelConfig, ModelProvider

def create_llm(model_config: ModelConfig) -> BaseChatModel:
    if model_config.provider == ModelProvider.OLLAMA:
        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            verbose=True,
            keep_alive=-1,
        )
    elif model_config.provider == ModelProvider.TOGETHER:
        return ChatTogether(
            model=model_config.name,
            temperature=model_config.temperature,
            together_api_key=Config.TOGETHER_API_KEY,
        )
    elif model_config.provider == ModelProvider.GEMINI:
        return ChatGoogleGenerativeAI(
            model=model_config.name,
            temperature=model_config.temperature,
            google_api_key=Config.GOOGLE_API_KEY,
            streaming=True,
        )
    elif model_config.provider == ModelProvider.PERPLEXITY:
        try:
            return ChatPerplexity(
                model=model_config.name,
                temperature=model_config.temperature,
                pplx_api_key=Config.OPENAI_API_KEY,
            )
        except Exception as e:
            raise ValueError("PERPLEXITY_API_KEY not found or is invalid. Please set the PERPLEXITY_API_KEY environment variable.") from e