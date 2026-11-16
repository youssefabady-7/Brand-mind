"""Unit tests for the NLU intent classification and entity extraction pipeline.
"""

import pytest
from src.agents.llm_client import RuleBasedMockLLMClient
from src.agents.nlu import parse_customer_message
from src.agents.nlu_schemas import CustomerIntent


@pytest.fixture
def mock_llm():
    return RuleBasedMockLLMClient()


def test_nlu_shipping_intent(mock_llm):
    """Shipping inquiry correctly identified with language detection."""
    msg = "How much is shipping to Alexandria and how many days?"
    result = parse_customer_message(msg, brand_id="lumora", client=mock_llm)
    assert result.intent == CustomerIntent.SHIPPING
    assert result.detected_language == "en"
    assert result.confidence >= 0.90


def test_nlu_order_status_entity_extraction(mock_llm):
    """Order status message extracts order ID and flags get_order_status tool."""
    msg = "عايز اعرف اوردر رقم LUM-1002 وصل لفين؟"
    result = parse_customer_message(msg, brand_id="lumora", client=mock_llm)
    assert result.intent == CustomerIntent.ORDER_STATUS
    assert result.detected_language == "ar_eg"
    assert result.entities.order_id == "LUM-1002"
    assert result.tool_needed == "get_order_status"


def test_nlu_return_eligibility_extraction(mock_llm):
    """Return inquiry extracts days elapsed, sealed state, and return intent."""
    msg = "I received the serum 10 days ago. The box is sealed and unopened. Can I return it?"
    result = parse_customer_message(msg, brand_id="lumora", client=mock_llm)
    assert result.intent == CustomerIntent.RETURNS
    assert result.entities.days_since_delivery == 10
    assert result.entities.is_opened is False
    assert result.tool_needed == "check_return_eligibility"


def test_nlu_allergic_reaction_complaint(mock_llm):
    """Allergic reaction triggers complaint intent and immediate escalation tool."""
    msg = "استخدمت السيروم وحصل لي حرقان ووشي احمر جدا وحسيت بالتهاب"
    result = parse_customer_message(msg, brand_id="lumora", client=mock_llm)
    assert result.intent == CustomerIntent.COMPLAINT
    assert result.detected_language == "ar_eg"
    assert result.entities.is_allergic_reaction is True
    assert result.tool_needed == "escalate_to_human"


def test_nlu_human_support_request(mock_llm):
    """Explicit human request triggers human_support intent and escalation."""
    msg = "عايز اكلم حد من خدمة العملاء حالا مش روبوت"
    result = parse_customer_message(msg, brand_id="lumora", client=mock_llm)
    assert result.intent == CustomerIntent.HUMAN_SUPPORT
    assert result.tool_needed == "escalate_to_human"
