import os

import pytest
from analyser import DataAnalysisAgent


@pytest.fixture(autouse=True)
def _set_fake_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")


class TestDataAnalysisAgentCreation:
    def test_create_instance_with_defaults(self):
        agent = DataAnalysisAgent()
        assert agent.llm is not None
        assert agent.df is None
        assert agent.filename == ""
        assert agent.plot_counter == 0

    def test_create_instance_with_custom_params(self):
        agent = DataAnalysisAgent(model="gpt-4", temperature=0.5)
        assert agent.llm.model_name == "gpt-4"
        assert agent.llm.temperature == 0.5


class TestExtractCode:
    @pytest.fixture
    def agent(self):
        return DataAnalysisAgent()

    def test_extract_python_code_block(self, agent):
        text = (
            'Some text\n```python\nimport pandas as pd\n'
            'print(df.head())\n```\nmore text'
        )
        code = agent._extract_code(text)
        assert code is not None
        assert "import pandas as pd" in code
        assert "print(df.head())" in code

    def test_extract_code_block_without_language(self, agent):
        text = '```\nprint("hello")\n```'
        code = agent._extract_code(text)
        assert code is not None
        assert 'print("hello")' in code

    def test_extract_code_import_and_print_fallback(self, agent):
        text = 'import pandas as pd\nprint(df.describe())'
        code = agent._extract_code(text)
        assert code is not None
        assert "import pandas as pd" in code

    def test_extract_code_returns_none_for_plain_text(self, agent):
        text = "This is just a plain text response without any code."
        code = agent._extract_code(text)
        assert code is None

    def test_extract_code_empty_string(self, agent):
        code = agent._extract_code("")
        assert code is None

    def test_extract_code_multiline_code_block(self, agent):
        text = (
            '```python\nimport matplotlib.pyplot as plt\nimport pandas as pd\n\n'
            'df = pd.DataFrame({"a": [1, 2, 3]})\nprint(df)\n'
            'plt.plot(df)\nplt.show()\n```'
        )
        code = agent._extract_code(text)
        assert code is not None
        assert "import matplotlib.pyplot" in code
        assert 'df = pd.DataFrame({"a": [1, 2, 3]})' in code


need_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set in environment",
)


@need_api_key
class TestDataAnalysisFull:
    def test_load_data(self):
        agent = DataAnalysisAgent()
        result = agent.load_data("nonexistent.csv")
        assert "Error" in result

    def test_describe_without_data(self):
        agent = DataAnalysisAgent()
        result = agent.describe_data()
        assert result == "No data loaded."

    def test_run_query_without_data(self):
        agent = DataAnalysisAgent()
        result = agent.run_query("Show me the data")
        assert "No data loaded" in result
