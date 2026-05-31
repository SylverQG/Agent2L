import pytest

from tools import ToolRegistry, calculator, current_datetime


class TestToolRegistry:
    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    def test_register_and_get(self, registry):
        def my_tool():
            return "hello"

        registry.register(my_tool)
        assert registry.get("my_tool") is my_tool

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("nonexistent") is None

    def test_unregister_removes_tool(self, registry):
        def my_tool():
            return "hello"

        registry.register(my_tool)
        registry.unregister("my_tool")
        assert registry.get("my_tool") is None

    def test_unregister_nonexistent_does_not_raise(self, registry):
        registry.unregister("nonexistent")

    def test_list_tools_returns_dicts_with_name_and_description(self, registry):
        def my_tool():
            """A test tool."""
            return "hello"

        registry.register(my_tool)
        tools = registry.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"
        assert tools[0]["description"] == "A test tool."

    def test_list_tools_empty_registry(self, registry):
        assert registry.list_tools() == []

    def test_register_returns_the_function(self, registry):
        def my_tool():
            return "hello"

        returned = registry.register(my_tool)
        assert returned is my_tool


class TestCalculatorTool:
    def test_addition(self):
        result = calculator.invoke({"expression": "2 + 3"})
        assert "5" in result

    def test_multiplication(self):
        result = calculator.invoke({"expression": "4 * 5"})
        assert "20" in result

    def test_division(self):
        result = calculator.invoke({"expression": "10 / 2"})
        assert "5.0" in result

    def test_complex_expression(self):
        result = calculator.invoke({"expression": "(2 + 3) * 4"})
        assert "20" in result

    def test_invalid_expression_returns_error(self):
        result = calculator.invoke({"expression": "1 / 0"})
        assert "Error" in result


class TestCurrentDatetimeTool:
    def test_returns_string(self):
        result = current_datetime.invoke({})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_string_with_custom_format(self):
        result = current_datetime.invoke({"format": "%Y-%m-%d"})
        assert isinstance(result, str)
        assert len(result) == 10
