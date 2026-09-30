from langchain_openai import ChatOpenAI

from uriel.config import ModelSpec


def build_chat_model(spec: ModelSpec) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=spec.base_url,
        model=spec.model,
        api_key=spec.key(),
        timeout=spec.timeout_s,
        max_retries=spec.max_retries,
        **spec.params,
    )
