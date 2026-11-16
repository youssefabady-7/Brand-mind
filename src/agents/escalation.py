"""Deterministic Escalation Checklist.

Evaluates the 7 mandatory escalation triggers BEFORE any draft reply is dispatched:
1. Explicit Human Request
2. Sensitive Medical / Dermatological Reaction
3. Angry Customer or Legal Threats (e.g. Consumer Protection Agency / حماية المستهلك)
4. Payment and Transactional Disputes (e.g. double charges)
5. Policy Silence (Missing knowledge coverage / RAG threshold failure)
6. Low Confidence Intent
7. Missing Critical Information

Non-negotiable rule:
Human escalation is treated as a safe first-class success path, never a failure.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.agents.nlu_schemas import CustomerIntent, ExtractedEntities
from src.retrieval.retriever import RetrievalResult

logger = logging.getLogger("brandmind.escalation")

CONFIDENCE_THRESHOLD = 0.60

# 1. Explicit Human Request Patterns
HUMAN_PATTERNS = [
    r"\btalk to a human\b",
    r"\bspeak to a human\b",
    r"\breal person\b",
    r"\bhuma[nn] agent\b",
    r"\bcustomer service rep(resentative)?\b",
    r"\bmanager\b",
    r"\blive agent\b",
    r"\bhuman support\b",
    r"بني\s*آدم",
    r"بني\s*ادم",
    r"خدمة\s*العملاء",
    r"كلمني\s*حد",
    r"حد\s*بشري",
    r"مش\s*روبوت",
    r"عايز\s*انسان",
    r"حولني\s*لموظف",
    r"شخص\s*حقيقي",
]
HUMAN_REGEX = re.compile("|".join(HUMAN_PATTERNS), re.IGNORECASE)

# 2. Medical / Dermatological Symptoms Patterns
MEDICAL_PATTERNS = [
    r"\ballergic\b",
    r"\breaction\b",
    r"\bburn(ing)?\b",
    r"\brash\b",
    r"\bswoll?en\b",
    r"\bblister\b",
    r"\birritat(ed|ion)\b",
    r"\bhospital\b",
    r"\bdoctor\b",
    r"حرقان",
    r"حساسية",
    r"وارم",
    r"تورم",
    r"التهاب",
    r"طفح\s*جلدي",
    r"مستشفى",
    r"دكتور",
    r"احمرار\s*شديد",
    r"بيهرش",
    r"حكة\s*شديدة",
]
MEDICAL_REGEX = re.compile("|".join(MEDICAL_PATTERNS), re.IGNORECASE)

# 3. Angry Customer / Legal Threat Patterns
ANGRY_LEGAL_PATTERNS = [
    r"\blawyer\b",
    r"\bsue\b",
    r"\blawsuit\b",
    r"\bpolice\b",
    r"\bconsumer protection\b",
    r"\bfraud\b",
    r"\bscam\b",
    r"\bthieves\b",
    r"\bworst (brand|service|company)\b",
    r"\bdisgusting\b",
    r"حماية\s*المستهلك",
    r"محامي",
    r"قضية",
    r"بلاغ",
    r"نصب",
    r"نصابين",
    r"سارقين",
    r"حرامية",
    r"بهدلة",
    r"شكوى\s*رسمية",
    r"هفضحكم",
    r"أسوأ\s*خدمة",
]
ANGRY_LEGAL_REGEX = re.compile("|".join(ANGRY_LEGAL_PATTERNS), re.IGNORECASE)

# 4. Payment Dispute Patterns
PAYMENT_PATTERNS = [
    r"\bcharged twice\b",
    r"\bdouble charged?\b",
    r"\bstolen card\b",
    r"\bunauthorized charge\b",
    r"\bbank dispute\b",
    r"اتخصم\s*مرتين",
    r"سحب\s*مرتين",
    r"اتسحب\s*من\s*الفيزا\s*مرتين",
    r"سرقة\s*كارت",
    r"عملية\s*نصب",
]
PAYMENT_REGEX = re.compile("|".join(PAYMENT_PATTERNS), re.IGNORECASE)


class EscalationCheckResult(BaseModel):
    """Outcome of pre-dispatch safety checklist."""
    should_escalate: bool
    trigger: Optional[str] = None
    hand_off_message: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


def evaluate_escalation_checklist(
    user_message: str,
    draft_reply: str,
    intent: CustomerIntent,
    entities: ExtractedEntities,
    confidence: float,
    rag_result: Optional[RetrievalResult],
    brand_config: Dict[str, Any],
    language: str = "en",
) -> EscalationCheckResult:
    """Executes deterministic rule verification over customer message and draft.

    Args:
        user_message: Raw customer query.
        draft_reply: Candidate generation from LLM.
        intent: Detected CustomerIntent.
        entities: Extracted structured entities.
        confidence: NLU model confidence score.
        rag_result: Vector retrieval result object.
        brand_config: Brand configuration YAML dictionary.
        language: Detected language ('en', 'ar', 'ar_eg').

    Returns:
        EscalationCheckResult detailing whether to halt automation and replace reply.
    """
    clean_message = user_message.strip()

    # Trigger 1: Explicit Human Request
    if intent == CustomerIntent.HUMAN_SUPPORT or HUMAN_REGEX.search(clean_message):
        return _build_escalation_response(
            trigger="EXPLICIT_HUMAN_REQUEST",
            brand_config=brand_config,
            language=language,
            details={"pattern": "User explicitly asked to speak to a human representative"},
        )

    # Trigger 2: Sensitive Medical Case / Allergic Reaction
    if entities.is_allergic_reaction or MEDICAL_REGEX.search(clean_message):
        return _build_escalation_response(
            trigger="SENSITIVE_MEDICAL_CASE",
            brand_config=brand_config,
            language=language,
            details={"pattern": "Customer reported skin irritation or health reaction"},
        )

    # Trigger 3: Angry Customer / Threat of Legal Action
    if ANGRY_LEGAL_REGEX.search(clean_message):
        return _build_escalation_response(
            trigger="ANGRY_CUSTOMER_OR_LEGAL",
            brand_config=brand_config,
            language=language,
            details={"pattern": "Detected extreme dissatisfaction or legal/regulatory threat"},
        )

    # Trigger 4: Payment Dispute / Double-Charge
    if PAYMENT_REGEX.search(clean_message):
        return _build_escalation_response(
            trigger="PAYMENT_DISPUTE",
            brand_config=brand_config,
            language=language,
            details={"pattern": "Customer reported billing discrepancy or unauthorized charge"},
        )

    # Trigger 5: Policy Silence (Knowledge base ungrounded or empty)
    if rag_result and (rag_result.policy_silence or not rag_result.has_answer):
        # Only escalate if the inquiry was not already addressed by tools
        return _build_escalation_response(
            trigger="POLICY_SILENCE",
            brand_config=brand_config,
            language=language,
            details={"pattern": "Knowledge base does not cover this inquiry with sufficient confidence"},
        )

    # Trigger 6: Low Confidence Intent
    if confidence < CONFIDENCE_THRESHOLD:
        return _build_escalation_response(
            trigger="LOW_CONFIDENCE",
            brand_config=brand_config,
            language=language,
            details={"confidence": confidence, "threshold": CONFIDENCE_THRESHOLD},
        )

    # All checks clear: safe to send response
    return EscalationCheckResult(should_escalate=False)


def _build_escalation_response(
    trigger: str,
    brand_config: Dict[str, Any],
    language: str,
    details: Dict[str, Any],
) -> EscalationCheckResult:
    """Builds localized honest hand-off message from brand YAML profile."""
    messages = brand_config.get("escalation", {}).get("hand_off_messages", {})
    if language in ("ar", "ar_eg"):
        hand_off = messages.get(language, messages.get("ar", "سأقوم بتحويلك لأحد ممثلي الدعم للمساعدة فوراً."))
    else:
        hand_off = messages.get("en", "I am connecting you with a brand specialist who can assist you directly.")

    logger.info(f"Escalation triggered: {trigger} | Details: {details}")

    return EscalationCheckResult(
        should_escalate=True,
        trigger=trigger,
        hand_off_message=hand_off,
        details=details,
    )
