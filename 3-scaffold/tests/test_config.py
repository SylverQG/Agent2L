from config import Config


class TestConfig:
    def test_default_openai_api_key_is_empty(self):
        config = Config()
        assert config.openai_api_key == ""

    def test_default_openai_api_base(self):
        config = Config()
        assert config.openai_api_base == "https://api.openai.com/v1"

    def test_default_anthropic_api_key_is_empty(self):
        config = Config()
        assert config.anthropic_api_key == ""

    def test_default_ollama_base_url(self):
        config = Config()
        assert config.ollama_base_url == "http://localhost:11434"

    def test_default_model(self):
        config = Config()
        assert config.default_model == "gpt-4o-mini"

    def test_default_temperature(self):
        config = Config()
        assert config.temperature == 0.7

    def test_default_max_tokens(self):
        config = Config()
        assert config.max_tokens == 4096
