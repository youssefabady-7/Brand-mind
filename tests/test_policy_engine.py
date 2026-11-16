"""Comprehensive unit tests for the deterministic return/exchange policy engine.

Validates all boundary conditions, exception rules, skincare hygiene constraints,
and audit logging without any LLM dependencies.
"""

from datetime import date, timedelta
import json
import os
from pathlib import Path
import pytest

from src.policies.engine import decide, calculate_days_elapsed, AUDIT_LOG_FILE
from src.policies.loader import load_brand_policy
from src.policies.schemas import (
    BrandPolicyConfig,
    PolicyDecisionType,
    PolicyReasonCode,
    ReturnFacts,
)

TODAY = date(2026, 11, 16)


@pytest.fixture
def lumora_policy() -> BrandPolicyConfig:
    return BrandPolicyConfig(
        brand_id="lumora",
        allowed_return_days=14,
        allowed_exchange_days=14,
        non_returnable_categories=["opened_skincare", "used_products", "unsealed_serums", "sample_kits"],
        require_unopened_for_skincare=True,
        require_unworn_for_apparel=False,
    )


@pytest.fixture
def urbankicks_policy() -> BrandPolicyConfig:
    return BrandPolicyConfig(
        brand_id="urbankicks",
        allowed_return_days=14,
        allowed_exchange_days=14,
        non_returnable_categories=["worn_outdoor_shoes", "underwear_and_socks"],
        require_unopened_for_skincare=False,
        require_unworn_for_apparel=True,
    )


def test_worked_example_exchange_eligible(lumora_policy):
    """Worked example: 10 days ago, changed mind / unopened, 10 <= 14 -> Eligible."""
    facts = ReturnFacts(
        order_id="LUM-1042",
        product_name="Hydrating Niacinamide Serum",
        days_since_delivery=10,
        is_opened=False,
        is_used=False,
        reason="changed_mind",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.EXCHANGE, today=TODAY)
    assert decision.eligible is True
    assert decision.reason_code == PolicyReasonCode.ELIGIBLE_WITHIN_WINDOW
    assert decision.requires_escalation is False


def test_boundary_day_14_exact(lumora_policy):
    """Day 14 on 14-day window: boundary value is eligible."""
    facts = ReturnFacts(
        product_name="Gentle Gel Cleanser",
        delivery_date=TODAY - timedelta(days=14),
        is_opened=False,
        is_used=False,
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is True
    assert decision.reason_code == PolicyReasonCode.ELIGIBLE_WITHIN_WINDOW


def test_boundary_day_15_expired(lumora_policy):
    """Day 15 on 14-day window: boundary value is rejected (expired window)."""
    facts = ReturnFacts(
        product_name="Gentle Gel Cleanser",
        delivery_date=TODAY - timedelta(days=15),
        is_opened=False,
        is_used=False,
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.EXPIRED_TIME_WINDOW


def test_skincare_hygiene_opened_rejected(lumora_policy):
    """Skincare product opened: rejected under hygiene safety rule."""
    facts = ReturnFacts(
        product_name="Peptide Glow Elixir",
        days_since_delivery=3,
        is_opened=True,
        is_used=False,
        reason="changed_mind",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.OPENED_HYGIENE_RESTRICTION


def test_skincare_hygiene_used_rejected(lumora_policy):
    """Skincare product used: rejected under hygiene safety rule."""
    facts = ReturnFacts(
        product_name="Ceramide Barrier Cream",
        days_since_delivery=2,
        is_opened=True,
        is_used=True,
        reason="texture_feel",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.OPENED_HYGIENE_RESTRICTION


def test_sensitive_medical_allergic_reaction_escalated(lumora_policy):
    """Allergic reaction triggers immediate customer health escalation."""
    facts = ReturnFacts(
        product_name="AHA BHA Exfoliating Toner",
        days_since_delivery=4,
        is_opened=True,
        is_used=True,
        is_allergic_reaction=True,
        reason="allergic_reaction",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.SENSITIVE_MEDICAL_REACTION
    assert decision.requires_escalation is True


def test_exception_defective_item(lumora_policy):
    """Damaged or defective pump on arrival is accepted even past window or unsealed."""
    facts = ReturnFacts(
        product_name="Centella Calming Mist",
        days_since_delivery=20,
        is_opened=True,
        is_defective=True,
        reason="damaged_on_arrival",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.EXCHANGE, today=TODAY)
    assert decision.eligible is True
    assert decision.reason_code == PolicyReasonCode.EXCEPTION_DEFECTIVE_ACCEPTED


def test_exception_wrong_item_shipped(lumora_policy):
    """Brand delivered wrong item: accepted for exchange regardless of packaging state."""
    facts = ReturnFacts(
        product_name="Hydrating Niacinamide Serum",
        days_since_delivery=5,
        is_opened=True,
        is_wrong_item=True,
        reason="wrong_item",
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.EXCHANGE, today=TODAY)
    assert decision.eligible is True
    assert decision.reason_code == PolicyReasonCode.EXCEPTION_WRONG_ITEM_ACCEPTED


def test_non_returnable_category(lumora_policy):
    """Sample kits or gift cards are explicitly non-returnable."""
    facts = ReturnFacts(
        product_name="Lumora Discovery Sample Kit",
        category="sample_kits",
        days_since_delivery=2,
        is_opened=False,
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.NON_RETURNABLE_CATEGORY


def test_missing_critical_delivery_facts(lumora_policy):
    """Missing delivery date or days elapsed prompts for more information."""
    facts = ReturnFacts(
        product_name="Peptide Glow Elixir",
        days_since_delivery=None,
        delivery_date=None,
    )
    decision = decide(facts, lumora_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.MISSING_CRITICAL_FACTS


def test_urbankicks_unworn_sneakers_eligible(urbankicks_policy):
    """UrbanKicks sneakers unworn in box are eligible within 14 days."""
    facts = ReturnFacts(
        product_name="Retro Apex High",
        days_since_delivery=7,
        is_used=False,
        has_original_packaging=True,
    )
    decision = decide(facts, urbankicks_policy, PolicyDecisionType.EXCHANGE, today=TODAY)
    assert decision.eligible is True
    assert decision.reason_code == PolicyReasonCode.ELIGIBLE_WITHIN_WINDOW


def test_urbankicks_worn_sneakers_rejected(urbankicks_policy):
    """UrbanKicks sneakers worn outdoors are rejected."""
    facts = ReturnFacts(
        product_name="StreetRunner V2",
        days_since_delivery=5,
        is_used=True,
    )
    decision = decide(facts, urbankicks_policy, PolicyDecisionType.RETURN, today=TODAY)
    assert decision.eligible is False
    assert decision.reason_code == PolicyReasonCode.USED_CONDITION_REJECTED


def test_audit_logging_persists_record(lumora_policy, tmp_path):
    """Verifies that decisions are serialized to audit log."""
    facts = ReturnFacts(
        order_id="TEST-AUDIT-1",
        days_since_delivery=5,
        is_opened=False,
    )
    decision = decide(facts, lumora_policy, today=TODAY, log_audit=True)
    assert AUDIT_LOG_FILE.exists()
    
    with open(AUDIT_LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) >= 1
        last_entry = json.loads(lines[-1])
        assert last_entry["audit_id"] == decision.audit_id
        assert last_entry["brand_id"] == "lumora"


def test_yaml_policy_loader():
    """Verifies policies load correctly from config/lumora.yaml and config/urbankicks.yaml."""
    lumora_cfg = load_brand_policy("lumora", config_dir=Path("config"))
    assert lumora_cfg.brand_id == "lumora"
    assert lumora_cfg.allowed_return_days == 14
    assert lumora_cfg.require_unopened_for_skincare is True

    urban_cfg = load_brand_policy("urbankicks", config_dir=Path("config"))
    assert urban_cfg.brand_id == "urbankicks"
    assert urban_cfg.require_unworn_for_apparel is True
