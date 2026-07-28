"""Model construction and key-rotating invocation.

`create_llm` builds a client bound to one specific API key. `RotatingChatModel`
wraps that factory so a single logical model can transparently move between the
keys in its pool, and then between fallback models, as free-tier quotas run out.
"""

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool

from config import Config, ModelConfig, ModelProvider
from custom_logging import log
from key_pool import KeyPool, PoolExhausted


def create_llm(model_config: ModelConfig, api_key: Optional[str] = None) -> BaseChatModel:
    """Build a chat model for one provider, using the supplied API key."""
    provider = model_config.provider

    if provider == ModelProvider.OLLAMA:
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_config.name,
            temperature=model_config.temperature,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            keep_alive=-1,
        )

    if provider == ModelProvider.GEMINI:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model_config.name,
            temperature=model_config.temperature,
            google_api_key=api_key,
        )

    if provider == ModelProvider.GROQ:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=model_config.name,
            temperature=model_config.temperature,
            api_key=api_key,
        )

    if provider == ModelProvider.TOGETHER:
        from langchain_together import ChatTogether

        return ChatTogether(
            model=model_config.name,
            temperature=model_config.temperature,
            together_api_key=api_key,
        )

    if provider == ModelProvider.PERPLEXITY:
        from langchain_perplexity import ChatPerplexity

        return ChatPerplexity(
            model=model_config.name,
            temperature=model_config.temperature,
            pplx_api_key=api_key,
        )

    raise ValueError(f"Unsupported model provider: {provider}")


# Providers whose tool-calling support is unreliable enough that we prefer not
# to bind tools to them. Perplexity is search-oriented and does not do the
# multi-step tool loop this agent depends on.
NO_TOOL_BINDING = {ModelProvider.PERPLEXITY}


class RotatingChatModel:
    """A chat model that rotates API keys, then falls back to other models.

    Exposes the small slice of the BaseChatModel surface the agent uses
    (`invoke` and `bind_tools`), so agent.py needs no special handling.
    """

    def __init__(self, models: Optional[Sequence[ModelConfig]] = None):
        self.models: List[ModelConfig] = list(models or Config.model_chain())
        if not self.models:
            raise RuntimeError(
                "No usable model configured. Set at least one provider API key "
                "(for example GOOGLE_API_KEY) in your .env file."
            )

        self._tools: Optional[List[BaseTool]] = None
        self._pools: Dict[str, KeyPool] = {}
        self._clients: Dict[str, BaseChatModel] = {}
        self.active_model: ModelConfig = self.models[0]
        self.last_usage: Dict[str, int] = {}
        self.session_usage: Dict[str, int] = {}

    # -- pool / client management ----------------------------------------

    def _pool_for(self, model: ModelConfig) -> KeyPool:
        key = f"{model.provider.value}:{model.name}"
        if key not in self._pools:
            credentials = Config.credentials(model.provider)
            # Ollama needs no key; a single empty slot keeps the loop uniform.
            keys = credentials.keys or [""]
            self._pools[key] = KeyPool(
                provider=model.provider.value,
                keys=keys,
                cooldown_seconds=Config.KEY_COOLDOWN_SECONDS,
            )
        return self._pools[key]

    def _client_for(self, model: ModelConfig, api_key: str) -> BaseChatModel:
        """Cache one client per (model, key) so we do not rebuild per call."""
        cache_key = f"{model.provider.value}:{model.name}:{api_key}"
        if cache_key not in self._clients:
            client = create_llm(model, api_key or None)
            if self._tools and model.provider not in NO_TOOL_BINDING:
                client = client.bind_tools(self._tools)
            self._clients[cache_key] = client
        return self._clients[cache_key]

    # -- BaseChatModel-compatible surface --------------------------------

    def bind_tools(self, tools: Sequence[BaseTool]) -> "RotatingChatModel":
        self._tools = list(tools)
        self._clients.clear()  # Rebuild clients so they carry the new tools.
        return self

    def invoke(self, messages: List[BaseMessage], **kwargs: Any) -> BaseMessage:
        """Invoke the first model whose pool still has capacity."""
        errors: List[str] = []

        for model in self.models:
            pool = self._pool_for(model)
            if not pool.has_capacity:
                continue

            def call(api_key: str, _model: ModelConfig = model) -> BaseMessage:
                client = self._client_for(_model, api_key)
                return client.invoke(messages, **kwargs)

            try:
                response = pool.run(call)
            except PoolExhausted as exhausted:
                errors.append(str(exhausted))
                if model is not self.models[-1]:
                    log(
                        f"[yellow]{model.provider.value}/{model.name} exhausted; "
                        f"falling back to the next model.[/yellow]"
                    )
                continue

            if self.active_model is not model:
                log(f"[cyan]Now serving from {model.provider.value}/{model.name}.[/cyan]")
            self.active_model = model
            self._record_usage(response)
            return response

        raise PoolExhausted(
            provider=", ".join(m.provider.value for m in self.models),
            pool_size=sum(len(self._pool_for(m)) for m in self.models),
        ) from RuntimeError("; ".join(errors))

    # -- usage accounting -------------------------------------------------

    def _record_usage(self, response: BaseMessage) -> None:
        """Capture real token counts reported by the provider, if present."""
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            metadata = getattr(response, "response_metadata", {}) or {}
            usage = metadata.get("token_usage") or metadata.get("usage") or {}

        def pick(*names: str) -> int:
            for name in names:
                value = usage.get(name)
                if isinstance(value, int):
                    return value
            return 0

        input_tokens = pick("input_tokens", "prompt_tokens")
        output_tokens = pick("output_tokens", "completion_tokens")
        total = pick("total_tokens") or (input_tokens + output_tokens)

        if total:
            self.last_usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total,
            }
            for field, value in self.last_usage.items():
                self.session_usage[field] = self.session_usage.get(field, 0) + value

    def pool_status(self) -> List[dict]:
        """Per-model, per-key status for display in the UI."""
        rows = []
        for model in self.models:
            for stat in self._pool_for(model).stats():
                rows.append({"model": f"{model.provider.value}/{model.name}", **stat})
        return rows
