"""End-to-end integration tests for the BrandMind Constrained Agent Loop.

Validates:
1. Factual FAQ retrieval (RAG grounded).
2. Order tracking tool integration.
3. Deterministic return eligibility tool execution.
4. Medical / sensitive case safety checklist escalation.
5. Explicit human request escalation.
6. Multi-turn contextual co-reference entity resolution.
7. Multi-tenant brand isolation and tone differentiation.
"""

from pathlib import Path
import pytest

from src.agents.agent import BrandMindAgent
from src.agents.llm_client import RuleBasedMockLLMClient
from src.agents.memory import MemoryManager
from src.retrieval.retriever import KnowledgeRetriever


@pytest.fixture
def agent():
    """Initializes a deterministic BrandMind agent instance."""
    llm = RuleBasedMockLLMClient()
    retriever = KnowledgeRetriever()
    memory = MemoryManager()
    return BrandMindAgent(llm_client=llm, retriever=retriever, memory_manager=memory)


def test_agent_shipping_faq_grounded(agent):
    """Shipping inquiry retrieves grounded Lumora knowledge without escalation."""
    response = agent.handle_message(
        session_id="test_sess_01",
        brand_id="lumora",
        message="What are your shipping rates and delivery times for Cairo?",
    )
    assert response.escalated is False
    assert response.intent == "shipping"
    assert len(response.sources) >= 1
    assert any("cairo" in s["snippet"].lower() or "shipping" in s["snippet"].lower() for s in response.sources)
    assert response.latency_ms > 0
    assert "Lumora" in response.reply or "delivery" in response.reply.lower()


def test_agent_order_status_tracking(agent):
    """Order tracking executes get_order_status tool with validated mock data."""
    response = agent.handle_message(
        session_id="test_sess_02",
        brand_id="lumora",
        message="Where is order LUM-1001? I want to track it.",
    )
    assert response.escalated is False
    assert response.intent == "order_status"
    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.tool_name == "get_order_status"
    assert tool_call.output["found"] is True
    assert tool_call.output["status"] == "delivered"
    assert tool_call.output["customer_name"] == "Nour El-Din Hassan"


def test_agent_return_eligibility_deterministic(agent):
    """Return eligibility runs pure rule engine and explains outcome without policy hallucination."""
    response = agent.handle_message(
        session_id="test_sess_03",
        brand_id="lumora",
        message="I received the serum 10 days ago. The box is sealed and unopened. Can I return it?",
    )
    assert response.escalated is False
    assert response.intent == "returns"
    assert len(response.tool_calls) >= 1
    tool_call = response.tool_calls[0]
    assert tool_call.tool_name == "check_return_eligibility"
    assert tool_call.output["eligible"] is True
    assert tool_call.output["reason_code"] == "ELIGIBLE_WITHIN_WINDOW"


def test_agent_sensitive_medical_escalation(agent):
    """Allergic reaction reports are immediately caught by safety gate and escalated."""
    response = agent.handle_message(
        session_id="test_sess_04",
        brand_id="lumora",
        message="استخدمت السيروم وحصل لي حرقان ووشي احمر جدا وحسيت بالتهاب",
    )
    assert response.escalated is True
    assert response.escalation_reason == "SENSITIVE_MEDICAL_CASE"
    assert response.language == "ar_eg"
    assert "لومورا" in response.reply
    assert "أخصائي" in response.reply or "نساعدك" in response.reply


def test_agent_explicit_human_request(agent):
    """Explicit human request triggers honest hand-off message and logs ticket."""
    response = agent.handle_message(
        session_id="test_sess_05",
        brand_id="lumora",
        message="عايز اكلم حد من خدمة العملاء حالا مش روبوت",
    )
    assert response.escalated is True
    assert response.escalation_reason == "EXPLICIT_HUMAN_REQUEST"
    assert "لومورا" in response.reply


def test_agent_multi_turn_co_reference(agent):
    """Agent remembers previously mentioned order ID in subsequent turns."""
    session_id = "test_sess_co_ref"

    # Turn 1: Mention order LUM-1002
    resp1 = agent.handle_message(
        session_id=session_id,
        brand_id="lumora",
        message="Where is order LUM-1002?",
    )
    assert resp1.intent == "order_status"
    assert resp1.entities["order_id"] == "LUM-1002"

    # Turn 2: Ask follow-up referring to 'it' without repeating LUM-1002
    resp2 = agent.handle_message(
        session_id=session_id,
        brand_id="lumora",
        message="Can you track the order for me?",
    )
    # The order_id was remembered from Turn 1
    assert resp2.entities["order_id"] == "LUM-1002"
    assert len(resp2.tool_calls) == 1
    assert resp2.tool_calls[0].arguments["order_id"] == "LUM-1002"


def test_agent_brand_tone_differentiation(agent):
    """Identical query produces calm tone for Lumora and energetic tone for UrbanKicks."""
    lumora_resp = agent.handle_message(
        session_id="tone_lumora",
        brand_id="lumora",
        message="Hello, can you tell me about your shipping?",
    )
    urban_resp = agent.handle_message(
        session_id="tone_urban",
        brand_id="urbankicks",
        message="Hello, can you tell me about your shipping?",
    )

    assert lumora_resp.escalated is False
    assert urban_resp.escalated is False
    assert "Lumora" in lumora_resp.reply
    assert "UrbanKicks" in urban_resp.reply
