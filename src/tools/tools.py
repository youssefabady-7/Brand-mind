"""Mocked tool functions for BrandMind agent execution.

Specified Tools:
1. search_products(query, filters, brand_id)
2. get_order_status(order_id, brand_id)
3. check_return_eligibility(facts, brand_id)
4. escalate_to_human(reason, context, brand_id)
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.policies.engine import decide
from src.policies.loader import load_brand_policy
from src.policies.schemas import PolicyDecisionType, ReturnFacts

logger = logging.getLogger("brandmind.tools")

PRODUCTS_CSV = Path("data/structured/products.csv")
ORDERS_CSV = Path("data/structured/orders.csv")
ESCALATION_LOG = Path("data/logs/escalations.jsonl")


def search_products(
    query: str,
    brand_id: str,
    category: Optional[str] = None,
    max_results: int = 5,
    products_csv_path: Path = PRODUCTS_CSV,
) -> List[Dict[str, Any]]:
    """Searches the brand product catalog for matching items.

    Args:
        query: Free text query (e.g. 'niacinamide', 'cleanser', 'sneakers', 'hoodie').
        brand_id: Brand filter ('lumora' or 'urbankicks').
        category: Optional specific category filter.
        max_results: Max items to return.
        products_csv_path: Path to products CSV file.

    Returns:
        List of matching product dicts.
    """
    if not products_csv_path.exists():
        logger.warning(f"Products CSV not found at {products_csv_path}")
        return []

    query_tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 1]
    matches = []

    with open(products_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("brand_id", "").lower() != brand_id.lower():
                continue

            if category and row.get("category", "").lower() != category.lower():
                continue

            searchable_text = " ".join([
                row.get("name", ""),
                row.get("category", ""),
                row.get("key_ingredients_or_material", ""),
                row.get("description", ""),
            ]).lower()

            # Rank by matched token count
            score = sum(1 for token in query_tokens if token in searchable_text)
            if score > 0 or not query_tokens:
                item = {
                    "product_id": row.get("product_id"),
                    "name": row.get("name"),
                    "category": row.get("category"),
                    "price_egp": float(row.get("price_egp", 0)),
                    "sizes_or_volume": row.get("sizes_or_volume"),
                    "key_ingredients_or_material": row.get("key_ingredients_or_material"),
                    "in_stock": row.get("in_stock", "True").lower() == "true",
                    "stock_quantity": int(row.get("stock_quantity", 0)),
                    "description": row.get("description"),
                    "relevance_score": score,
                }
                matches.append(item)

    matches.sort(key=lambda x: x["relevance_score"], reverse=True)
    return matches[:max_results]


def get_order_status(
    order_id: str,
    brand_id: Optional[str] = None,
    orders_csv_path: Path = ORDERS_CSV,
) -> Dict[str, Any]:
    """Retrieves validated delivery and shipment status for a mocked order.

    Args:
        order_id: Formatted order identifier (e.g. 'LUM-1001', 'UK-2001').
        brand_id: Optional brand identifier to ensure cross-brand isolation.
        orders_csv_path: Path to orders CSV file.

    Returns:
        Dictionary containing order status, dates, items, and tracking info.
    """
    if not orders_csv_path.exists():
        return {"found": False, "message": "Orders database unavailable"}

    clean_order_id = order_id.strip().upper()

    with open(orders_csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("order_id", "").strip().upper() == clean_order_id:
                if brand_id and row.get("brand_id", "").lower() != brand_id.lower():
                    return {
                        "found": False,
                        "message": f"Order {order_id} does not belong to brand {brand_id}",
                    }

                items = []
                try:
                    items = json.loads(row.get("items_json", "[]"))
                except json.JSONDecodeError:
                    pass

                return {
                    "found": True,
                    "order_id": row.get("order_id"),
                    "brand_id": row.get("brand_id"),
                    "customer_name": row.get("customer_name_synthetic"),
                    "governorate": row.get("governorate"),
                    "city_area": row.get("city_area"),
                    "status": row.get("status"),
                    "order_date": row.get("order_date"),
                    "delivery_date": row.get("delivery_date") or None,
                    "items": items,
                    "subtotal_egp": float(row.get("subtotal_egp", 0)),
                    "shipping_egp": float(row.get("shipping_egp", 0)),
                    "total_egp": float(row.get("total_egp", 0)),
                    "tracking_number": row.get("tracking_number_synthetic"),
                }

    return {"found": False, "order_id": order_id, "message": f"Order {order_id} not found in database"}


def check_return_eligibility(
    facts: Dict[str, Any],
    brand_id: str,
    decision_type: str = "return",
    today: Optional[date] = None,
) -> Dict[str, Any]:
    """Executes deterministic rule engine evaluation on extracted facts.

    Args:
        facts: Customer-provided claims dictionary.
        brand_id: Target brand identifier.
        decision_type: 'return' or 'exchange'.
        today: Reference date for testing.

    Returns:
        Structured outcome from PolicyDecision.
    """
    parsed_facts = ReturnFacts(**facts)
    policy = load_brand_policy(brand_id)
    dtype = PolicyDecisionType.EXCHANGE if decision_type.lower() == "exchange" else PolicyDecisionType.RETURN
    decision = decide(parsed_facts, policy, decision_type=dtype, today=today, log_audit=True)
    return decision.model_dump(mode="json")


def escalate_to_human(
    reason: str,
    context: Dict[str, Any],
    brand_id: str,
    language: str = "en",
    escalation_log_path: Path = ESCALATION_LOG,
) -> Dict[str, Any]:
    """Logs conversation context and triggers safe hand-off to human support.

    Args:
        reason: Operational trigger name (e.g. 'allergic_reaction', 'angry_customer', 'policy_silence').
        context: Conversation transcript snippet, intent, and customer details.
        brand_id: Brand identifier.
        language: Language code ('en', 'ar', 'ar_eg').
        escalation_log_path: Path to write the escalation event log.

    Returns:
        Hand-off payload containing ticket reference and localized honest message.
    """
    ticket_id = f"ESC-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now(timezone.utc).isoformat()

    escalation_record = {
        "ticket_id": ticket_id,
        "timestamp": timestamp,
        "brand_id": brand_id,
        "reason": reason,
        "language": language,
        "context": context,
    }

    try:
        escalation_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(escalation_log_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(escalation_record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.error(f"Failed to record escalation event: {exc}")

    # Load brand localized hand-off message from YAML
    config_file = Path("config") / f"{brand_id.lower()}.yaml"
    hand_off_message = "I am connecting you with a human representative. Please hold."
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            messages = cfg.get("escalation", {}).get("hand_off_messages", {})
            if language in ("ar", "ar_eg"):
                hand_off_message = messages.get(language, messages.get("ar", hand_off_message))
            else:
                hand_off_message = messages.get("en", hand_off_message)

    return {
        "escalated": True,
        "ticket_id": ticket_id,
        "reason": reason,
        "timestamp": timestamp,
        "hand_off_message": hand_off_message,
    }
