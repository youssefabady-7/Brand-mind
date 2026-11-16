"""Data models and schemas for the deterministic policy engine.
Follows strict typing and immutable validation without any LLM dependencies.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PolicyDecisionType(str, Enum):
    RETURN = "return"
    EXCHANGE = "exchange"


class PolicyReasonCode(str, Enum):
    ELIGIBLE_WITHIN_WINDOW = "ELIGIBLE_WITHIN_WINDOW"
    EXPIRED_TIME_WINDOW = "EXPIRED_TIME_WINDOW"
    NON_RETURNABLE_CATEGORY = "NON_RETURNABLE_CATEGORY"
    OPENED_HYGIENE_RESTRICTION = "OPENED_HYGIENE_RESTRICTION"
    USED_CONDITION_REJECTED = "USED_CONDITION_REJECTED"
    MISSING_CRITICAL_FACTS = "MISSING_CRITICAL_FACTS"
    EXCEPTION_DEFECTIVE_ACCEPTED = "EXCEPTION_DEFECTIVE_ACCEPTED"
    EXCEPTION_WRONG_ITEM_ACCEPTED = "EXCEPTION_WRONG_ITEM_ACCEPTED"
    SENSITIVE_MEDICAL_REACTION = "SENSITIVE_MEDICAL_REACTION"


class ReturnFacts(BaseModel):
    """Customer-provided facts extracted from conversation turns."""
    order_id: Optional[str] = Field(default=None, description="Mocked or real order ID (e.g., LUM-1042)")
    product_name: Optional[str] = Field(default=None, description="Name of the product")
    category: Optional[str] = Field(default=None, description="Product category (e.g. serum, cleanser, sneakers)")
    delivery_date: Optional[date] = Field(default=None, description="Date the customer received the order")
    days_since_delivery: Optional[int] = Field(default=None, ge=0, description="Explicit days elapsed if known")
    is_opened: Optional[bool] = Field(default=None, description="Whether product packaging/seal was unsealed")
    is_used: Optional[bool] = Field(default=None, description="Whether product was used or applied")
    has_original_packaging: Optional[bool] = Field(default=True, description="Original packaging present")
    reason: Optional[str] = Field(
        default="changed_mind",
        description="Reason: changed_mind, damaged_on_arrival, wrong_item, allergic_reaction, wrong_size"
    )
    is_defective: bool = Field(default=False, description="Whether item arrived broken or defective")
    is_wrong_item: bool = Field(default=False, description="Whether brand delivered incorrect item")
    is_allergic_reaction: bool = Field(default=False, description="Whether customer experienced irritation or reaction")


class BrandPolicyConfig(BaseModel):
    """Declarative brand policy rules loaded from YAML or database."""
    brand_id: str
    allowed_return_days: int = 14
    allowed_exchange_days: int = 14
    non_returnable_categories: List[str] = Field(default_factory=list)
    require_unopened_for_skincare: bool = True
    require_unworn_for_apparel: bool = True
    allow_defective_exceptions: bool = True
    allow_wrong_item_exceptions: bool = True


class PolicyDecision(BaseModel):
    """Pure deterministic decision output with audit tracking."""
    audit_id: str
    brand_id: str
    decision_type: PolicyDecisionType
    eligible: bool
    reason_code: PolicyReasonCode
    reason_text_en: str
    reason_text_ar: str
    requires_escalation: bool = False
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    evaluated_facts: Dict[str, Any]
