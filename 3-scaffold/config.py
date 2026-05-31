"""配置模块 — 基于 pydantic-settings 的应用配置管理。

从 .env 文件和环境变量中读取 LLM API 密钥及模型参数，
提供统一的全局配置对象供其他模块使用。
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """应用配置类，自动从 .env 文件加载 LLM 相关配置项。

    支持的配置项包括 OpenAI / Anthropic 的 API 密钥与地址、
    Ollama 本地地址，以及模型名称、温度、最大 Token 数等参数。
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    openai_api_base: str = "https://api.openai.com/v1"
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    default_model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 4096


config = Config()
