import re
import pytest

from tools import search_knowledge_base, create_ticket, check_order_status


class TestSearchKnowledgeBase:
    def test_known_query_returns_policy_info(self):
        result = search_knowledge_base("return policy")
        assert "return" in result.lower()
        assert "Knowledge Base" in result
        assert "30 days" in result

    def test_unknown_query_returns_fallback(self):
        result = search_knowledge_base("nonexistent_topic_xyz")
        assert "couldn't find a specific answer" in result
        assert "support ticket" in result


class TestCreateTicket:
    def test_ticket_contains_uuid_like_id(self):
        result = create_ticket("Alice", "Login issue")
        assert "Ticket Created" in result
        assert "TKT-" in result
        match = re.search(r"TKT-([A-Z0-9]{8})", result)
        assert match is not None
        assert len(match.group(1)) == 8

    def test_ticket_includes_customer_info(self):
        result = create_ticket("Bob", "Payment failed")
        assert "Bob" in result
        assert "Payment failed" in result
        assert "Open" in result


class TestCheckOrderStatus:
    def test_known_order_returns_details(self):
        result = check_order_status("ORD-20240501")
        assert "Order Status" in result
        assert "ORD-20240501" in result
        assert "Wireless Headphones" in result
        assert "shipped" in result

    def test_unknown_order_returns_not_found(self):
        result = check_order_status("ORD-99999999")
        assert "not found" in result
        assert "ORD-99999999" in result

    def test_order_id_case_insensitive(self):
        result = check_order_status("ord-20240502")
        assert "ORD-20240502" in result
        assert "Laptop Stand" in result