"""Unit tests for FastAPI endpoints (POST /chat, GET /orders, POST /returns, etc.).
"""

from fastapi.testclient import TestClient
import pytest

from src.main import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "BrandMind" in data["service"]


def test_chat_endpoint_shipping():
    payload = {
        "session_id": "api_test_sess_1",
        "brand_id": "lumora",
        "message": "How much is shipping to Cairo and Giza?",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "shipping"
    assert data["escalated"] is False
    assert len(data["sources"]) >= 1
    assert data["latency_ms"] > 0


def test_chat_endpoint_sensitive_escalation():
    payload = {
        "session_id": "api_test_sess_2",
        "brand_id": "lumora",
        "message": "استخدمت السيروم وحصل لي حرقان ووشي احمر جدا وحسيت بالتهاب",
    }
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is True
    assert data["escalation_reason"] == "SENSITIVE_MEDICAL_CASE"


def test_get_order_found():
    response = client.get("/orders/LUM-1001?brand_id=lumora")
    assert response.status_code == 200
    data = response.json()
    assert data["order_id"] == "LUM-1001"
    assert data["status"] == "delivered"
    assert data["customer_name"] == "Nour El-Din Hassan"


def test_get_order_not_found():
    response = client.get("/orders/LUM-9999?brand_id=lumora")
    assert response.status_code == 404


def test_returns_check_endpoint():
    payload = {
        "facts": {
            "order_id": "LUM-1042",
            "product_name": "Hydrating Niacinamide Serum",
            "days_since_delivery": 8,
            "is_opened": False,
            "is_used": False,
        },
        "brand_id": "lumora",
        "decision_type": "return",
    }
    response = client.post("/returns/check", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["eligible"] is True
    assert data["reason_code"] == "ELIGIBLE_WITHIN_WINDOW"


def test_returns_request_endpoint():
    payload = {
        "order_id": "LUM-1001",
        "brand_id": "lumora",
        "product_name": "Hydrating Niacinamide 10% Serum",
        "reason": "changed_mind",
        "customer_phone": "+201091234001",
    }
    response = client.post("/returns/request", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["return_id"].startswith("RET-")
    assert data["status"] == "pending_courier_pickup"


def test_manual_escalate_endpoint():
    payload = {
        "reason": "customer_requested_manager",
        "brand_id": "lumora",
        "user_message": "Please connect me to management",
        "language": "en",
    }
    response = client.post("/escalate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["escalated"] is True
    assert data["ticket_id"].startswith("ESC-")


def test_get_product_by_id():
    response = client.get("/products/LUM-P01")
    assert response.status_code == 200
    data = response.json()
    assert data["product_id"] == "LUM-P01"
    assert data["price_egp"] == 480.0
    assert "Niacinamide" in data["name"]


def test_search_products_catalog():
    response = client.get("/products?query=sneakers&brand_id=urbankicks")
    assert response.status_code == 200
    data = response.json()
    assert len(data) >= 1
    assert any("Apex Retro" in p["name"] for p in data)
