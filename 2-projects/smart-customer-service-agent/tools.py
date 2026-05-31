"""
智能客服 Agent 工具模块。

提供知识库检索、创建支持工单和查询订单状态等工具函数，
供主程序中的 LLM Agent 调用。
"""
from typing import Any
import uuid


KNOWLEDGE_BASE: dict[str, str] = {
    "return policy": "Our return policy allows returns within 30 days of purchase. "
    "Items must be unused and in original packaging. "
    "Please contact support to initiate a return.",
    "shipping time": "Standard shipping takes 5-7 business days. "
    "Express shipping takes 2-3 business days. "
    "International shipping may take 10-15 business days.",
    "reset password": (
        "To reset your password, go to the login page and click 'Forgot Password'. "
        "Enter your registered email address "
        "and we will send you a password reset link."
    ),
    "payment methods": (
        "We accept Visa, MasterCard, American Express, PayPal, and Apple Pay."
    ),
    "refund": (
        "Refunds are processed within 5-10 business days after we receive your return. "
        "The refund will be issued to your original payment method."
    ),
    "warranty": (
        "All products come with a 1-year manufacturer warranty "
        "covering defects in materials and workmanship."
    ),
    "contact": "You can reach our support team via email at support@example.com "
    "or call us at 1-800-555-0199, Monday to Friday, 9 AM to 6 PM.",
}


ORDERS: dict[str, dict[str, Any]] = {
    "ORD-20240501": {
        "status": "shipped",
        "product": "Wireless Headphones",
        "estimated_delivery": "2026-06-05",
        "carrier": "FedEx",
        "tracking": "FX1234567890",
    },
    "ORD-20240502": {
        "status": "processing",
        "product": "Laptop Stand",
        "estimated_delivery": "2026-06-10",
        "carrier": "UPS",
        "tracking": "N/A",
    },
    "ORD-20240503": {
        "status": "delivered",
        "product": "USB-C Hub",
        "estimated_delivery": "2026-05-28",
        "carrier": "USPS",
        "tracking": "US9876543210",
    },
}


def search_knowledge_base(query: str) -> str:
    """在知识库中搜索与用户查询匹配的答案。

    将查询文本转为小写后在知识库字典的关键词中进行模糊匹配，
    返回第一个匹配到的答案。

    参数:
        query: 用户的查询文本

    返回:
        知识库匹配结果文本，未匹配时提示可创建工单
    """
    normalized_query = query.lower()
    for keyword, answer in KNOWLEDGE_BASE.items():
        if keyword in normalized_query:
            return f"[Knowledge Base] {answer}"
    return (
        "[Knowledge Base] I couldn't find a specific answer to your question. "
        "Would you like me to create a support ticket for further assistance?"
    )


def create_ticket(customer_name: str, issue: str) -> str:
    """为客户创建新的支持工单。

    生成唯一工单编号，记录客户名称和问题描述，
    返回包含工单详情的文本。

    参数:
        customer_name: 客户姓名
        issue: 问题描述

    返回:
        包含工单编号、客户信息、问题描述和状态的文本
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    ticket_info = (
        f"[Ticket Created] Ticket #{ticket_id}\n"
        f"  Customer: {customer_name}\n"
        f"  Issue: {issue}\n"
        f"  Status: Open\n"
        f"  A support representative will reach out within 24 hours."
    )
    return ticket_info


def check_order_status(order_id: str) -> str:
    """查询指定订单 ID 的当前状态和物流信息。

    在预定义的订单字典中查找订单，返回订单的详细状态信息。
    如果订单不存在，返回未找到提示。

    参数:
        order_id: 订单编号（不区分大小写）

    返回:
        订单状态详情文本，或未找到的错误提示
    """
    order_id_upper = order_id.upper()
    if order_id_upper in ORDERS:
        order = ORDERS[order_id_upper]
        status = order["status"]
        product = order["product"]
        estimated = order["estimated_delivery"]
        carrier = order["carrier"]
        tracking = order["tracking"]
        return (
            f"[Order Status] Order #{order_id_upper}\n"
            f"  Product: {product}\n"
            f"  Status: {status}\n"
            f"  Estimated Delivery: {estimated}\n"
            f"  Carrier: {carrier}\n"
            f"  Tracking: {tracking}"
        )
    return (
        f"[Order Status] Order #{order_id_upper} not found. "
        "Please verify your order ID and try again."
    )
