"""NLU schemas for intent classification and structured JSON entity extraction.
Follows Section 6.4 [SPEC].
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CustomerIntent(str, Enum):
    """The 9 specified BrandMind intents."""
    SHIPPING = "shipping"
    RETURNS = "returns"
    EXCHANGE = "exchange"
    REFUND = "refund"
    PRODUCT_INFO = "product_info"
    AVAILABILITY = "availability"
    ORDER_STATUS = "order_status"
    COMPLAINT = "complaint"
    HUMAN_SUPPORT = "human_support"


class ExtractedEntities(BaseModel):
    """Structured entities extracted from conversation turns."""
    product: Optional[str] = Field(default=None, description="Mentioned product name or category")
    order_id: Optional[str] = Field(default=None, description="Formatted order number e.g. LUM-1001, UK-2001")
    date: Optional[str] = Field(default=None, description="Raw date string or timeline mentioned")
    days_since_delivery: Optional[int] = Field(default=None, ge=0, description="Explicit number of days elapsed")
    size: Optional[str] = Field(default=None, description="Mentioned shoe or apparel size (e.g. EU 43, L)")
    colour: Optional[str] = Field(default=None, description="Color variant")
    reason: Optional[str] = Field(default=None, description="Reason for return/exchange or inquiry")
    is_opened: Optional[bool] = Field(default=None, description="Whether product seal/box was opened")
    is_used: Optional[bool] = Field(default=None, description="Whether product was used or worn")
    is_defective: Optional[bool] = Field(default=None, description="Whether item arrived damaged or defective")
    is_allergic_reaction: Optional[bool] = Field(default=None, description="Whether customer reported skin irritation")


class NLUOutput(BaseModel):
    """Unified output from the NLU understanding pipeline."""
    intent: CustomerIntent
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    detected_language: str = "en"
    tool_needed: Optional[str] = Field(
        default=None,
        description="Suggested tool name if action required: search_products, get_order_status, check_return_eligibility, escalate_to_human"
    )
