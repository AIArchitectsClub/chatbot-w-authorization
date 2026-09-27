"""LLM provider selection. Swapping providers is a config change (LLM_PROVIDER)."""

from agno.models.base import Model
from agno.models.groq import Groq
from agno.models.ollama import Ollama
from agno.models.openai import OpenAIChat

from app.config import Settings


def make_model(settings: Settings) -> Model:
    # Low temperature: this is a policy Q&A bot, so we want figures quoted, not invented.
    temperature = settings.llm_temperature
    match settings.llm_provider:
        case "ollama":
            return Ollama(
                id=settings.ollama_model,
                host=settings.ollama_host,
                options={"temperature": temperature},
            )
        case "groq":
            if not settings.groq_api_key:
                raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
            return Groq(
                id=settings.groq_model, api_key=settings.groq_api_key, temperature=temperature
            )
        case "openai":
            if not settings.openai_api_key:
                raise ValueError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
            return OpenAIChat(
                id=settings.openai_model, api_key=settings.openai_api_key, temperature=temperature
            )
