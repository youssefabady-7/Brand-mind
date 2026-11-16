"""Unit tests for the 4 core BrandMind tool functions.
"""

from datetime import date
import json
from pathlib import Path
import pytest

from src.tools.tools import (
    search_products,
    get_order_status,
    check_return_eligibility,
    escalate_to_human,
)


def test_search_products_lumora():
    """Searching for 'niacinamide' returns Lumora serum with high relevance."""
    results = search_products("niacinamide", brand_id="lumora")
    assert len(results) >= 1
    assert results[0]["product_id"] == "LUM-P01"
    assert "Niacinamide" in results[0]["name"]
    assert results[0]["price_egp"] == 480.0


def test_search_products_brand_isolation():
    """Sneakers query for Lumora returns empty; for UrbanKicks returns sneaker catalog."""
    lumora_sneakers = search_products("sneakers", brand_id="lumora")
    assert len(lumora_sneakers) == 0

    urban_sneakers = search_products("sneakers", brand_id="urbankicks")
    assert len(urban_sneakers) >= 1
    assert any("Sneaker" in p["name"] for p in urban_sneakers)


def test_get_order_status_found():
    """Retrieving existing Lumora order LUM-1001 returns delivered status and items."""
    order = get_order_status("LUM-1001", brand_id="lumora")
    assert order["found"] is True
    assert order["status"] == "delivered"
    assert order["customer_name"] == "Nour El-Din Hassan"
    assert len(order["items"]) == 1
    assert order["items"][0]["product_id"] == "LUM-P01"


def test_get_order_status_cross_brand_blocked():
    """Order LUM-1001 queried under urbankicks brand is rejected."""
    order = get_order_status("LUM-1001", brand_id="urbankicks")
    assert order["found"] is False


def test_get_order_status_not_found():
    """Non-existent order returns clean not found payload."""
    order = get_order_status("LUM-9999", brand_id="lumora")
    assert order["found"] is False


def test_check_return_eligibility_tool():
    """Tool invocation delegates to deterministic engine accurately."""
    facts = {
        "product_name": "Hydrating Niacinamide 10% Serum",
        "days_since_delivery": 4,
        "is_opened": False,
        "is_used": False,
    }
    decision = check_return_eligibility(facts, brand_id="lumora", today=date(2026, 11, 16))
    assert decision["eligible"] is True
    assert decision["reason_code"] == "ELIGIBLE_WITHIN_WINDOW"


def test_escalate_to_human_tool(tmp_path):
    """Escalation tool generates ticket and returns localized brand message."""
    log_file = tmp_path / "escalations.jsonl"
    context = {"user_message": "عندي احمرار شديد في وشي وحرقان", "intent": "complaint"}
    result = escalate_to_human(
        reason="allergic_reaction",
        context=context,
        brand_id="lumora",
        language="ar_eg",
        escalation_log_path=log_file,
    )
    assert result["escalated"] is True
    assert result["ticket_id"].startswith("ESC-")
    assert "لومورا" in result["hand_off_message"]

    # Verify log persistence
    assert log_file.exists()
    with open(log_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        assert len(records) == 1
        assert records[0]["reason"] == "allergic_reaction"
