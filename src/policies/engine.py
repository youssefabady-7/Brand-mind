"""Deterministic Return/Exchange Policy Engine.

Central Design Rule:
THE LLM NEVER INVENTS COMPANY POLICY.
All decisions, boundary evaluations, and reason codes are executed in pure,
deterministic Python with zero LLM dependence.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from src.policies.schemas import (
    BrandPolicyConfig,
    PolicyDecision,
    PolicyDecisionType,
    PolicyReasonCode,
    ReturnFacts,
)

logger = logging.getLogger("brandmind.policies")

AUDIT_LOG_DIR = Path("data/logs")
AUDIT_LOG_FILE = AUDIT_LOG_DIR / "audit_decisions.jsonl"


def log_decision_to_audit(decision: PolicyDecision) -> None:
    """Appends an auditable JSON record of the decision to disk.
    
    Ensures full reproducibility and governance compliance.
    """
    try:
        AUDIT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(decision.model_dump_json() + "\n")
    except Exception as exc:
        logger.error(f"Failed to write decision audit log: {exc}")


def calculate_days_elapsed(facts: ReturnFacts, today: date) -> Optional[int]:
    """Calculates days elapsed since delivery, either from explicit facts or dates."""
    if facts.days_since_delivery is not None:
        return facts.days_since_delivery
    if facts.delivery_date is not None:
        delta = (today - facts.delivery_date).days
        return max(0, delta)
    return None


def decide(
    facts: ReturnFacts,
    policy: BrandPolicyConfig,
    decision_type: PolicyDecisionType = PolicyDecisionType.RETURN,
    today: Optional[date] = None,
    log_audit: bool = True,
) -> PolicyDecision:
    """Evaluates customer return/exchange eligibility against brand policy rules.

    Args:
        facts: Customer-provided claims extracted from dialogue.
        policy: Configured brand policy thresholds and category exclusions.
        decision_type: Whether the request is a 'return' or 'exchange'.
        today: Injected reference date for deterministic testing.
        log_audit: Whether to persist the decision to the audit log.

    Returns:
        PolicyDecision containing strict boolean outcome, standardized reason code,
        and localized explanation strings.
    """
    if today is None:
        today = date.today()

    audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
    allowed_days = (
        policy.allowed_return_days
        if decision_type == PolicyDecisionType.RETURN
        else policy.allowed_exchange_days
    )

    days_elapsed = calculate_days_elapsed(facts, today)

    # 1. Critical Sensitive Case: Allergic Reaction / Irritation (Priority 1)
    if facts.is_allergic_reaction or facts.reason == "allergic_reaction":
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.SENSITIVE_MEDICAL_REACTION,
            reason_text_en=(
                "Your safety is our top priority. Because you experienced a skin reaction, "
                "this request has been flagged for dedicated review by our health & care support team."
            ),
            reason_text_ar=(
                "سلامتك هي أولويتنا القصوى. نظراً لحدوث حساسية أو تهيج للبشرة، "
                "تم تحويل طلبك مباشرة لفريق العناية المتخصص لمتابعة حالتك والاهتمام بك."
            ),
            requires_escalation=True,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 2. Defective / Damaged on Arrival Exception
    if (facts.is_defective or facts.reason == "damaged_on_arrival") and policy.allow_defective_exceptions:
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=True,
            reason_code=PolicyReasonCode.EXCEPTION_DEFECTIVE_ACCEPTED,
            reason_text_en=(
                "We sincerely apologize for the inconvenience. Items received defective or damaged "
                "are fully eligible for immediate replacement or full refund under our quality guarantee."
            ),
            reason_text_ar=(
                "نعتذر بشدة عن هذا الخطأ. المنتجات التي تصل بها عيوب تصنيع أو تلف أثناء الشحن "
                "مؤهلة بالكامل للاستبدال الفوري أو استرجاع كامل المبلغ طبقاً لضمان الجودة."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 3. Wrong Item Delivered Exception
    if (facts.is_wrong_item or facts.reason == "wrong_item") and policy.allow_wrong_item_exceptions:
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=True,
            reason_code=PolicyReasonCode.EXCEPTION_WRONG_ITEM_ACCEPTED,
            reason_text_en=(
                "Our sincere apologies for sending the wrong item. We will arrange a courier exchange "
                "at zero shipping cost to you to deliver your correct order."
            ),
            reason_text_ar=(
                "نعتذر بشدة عن استلام منتج خاطئ. سيتم إرسال مندوب الشحن لاستبدال المنتج مجاناً "
                "وتسليمك طلبك الصحيح دون أي مصاريف شحن."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 4. Check for Missing Critical Delivery Information
    if days_elapsed is None:
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.MISSING_CRITICAL_FACTS,
            reason_text_en=(
                f"To determine {decision_type.value} eligibility, we need your order delivery date "
                f"or order number to verify the {allowed_days}-day policy window."
            ),
            reason_text_ar=(
                f"للتحقق من إمكانية { 'الاسترجاع' if decision_type == PolicyDecisionType.RETURN else 'الاستبدال' }، "
                f"نحتاج لمعرفة تاريخ استلام الطلب أو رقم الطلب للتأكد من فترة السماح ({allowed_days} يوماً)."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 5. Non-Returnable Category Exclusion
    if facts.category and any(
        facts.category.lower() in non_ret.lower() for non_ret in policy.non_returnable_categories
    ):
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.NON_RETURNABLE_CATEGORY,
            reason_text_en=(
                f"Items under category '{facts.category}' are strictly non-returnable and non-exchangeable "
                "for health, safety, and promotional terms."
            ),
            reason_text_ar=(
                f"المنتجات ضمن فئة '{facts.category}' غير قابلة للاسترجاع أو الاستبدال "
                "طبقاً لشروط السلامة الصحية وسياسة المتجر."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 6. Skincare Hygiene Constraint: Opened or Used Products
    if policy.require_unopened_for_skincare and (facts.is_opened is True or facts.is_used is True):
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.OPENED_HYGIENE_RESTRICTION,
            reason_text_en=(
                "For hygiene and public health safety, skincare and cosmetic products cannot be returned "
                "or exchanged once the seal is broken or product is unsealed/used."
            ),
            reason_text_ar=(
                "حفاظاً على الصحة العامة والسلامة الصحية، لا يمكن استرجاع أو استبدال مستحضرات العناية بالبشرة "
                "إذا تم فتح الغلاف أو إزالة لاصق الأمان أو استخدام المنتج."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 7. Apparel / Streetwear Constraint: Worn Condition
    if policy.require_unworn_for_apparel and facts.is_used is True:
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.USED_CONDITION_REJECTED,
            reason_text_en=(
                "Footwear and apparel must be completely unworn, clean-soled, and in original packaging "
                "with all tags intact to be eligible for return or exchange."
            ),
            reason_text_ar=(
                "يجب أن تكون الأحذية والملابس غير مستعملة تماماً وبنعل نظيف ومع كافة بطاقات الأسعار "
                "والعبوة الأصلية لتكون مؤهلة للاسترجاع أو الاستبدال."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 8. Boundary Time-Window Check
    if days_elapsed > allowed_days:
        decision = PolicyDecision(
            audit_id=audit_id,
            brand_id=policy.brand_id,
            decision_type=decision_type,
            eligible=False,
            reason_code=PolicyReasonCode.EXPIRED_TIME_WINDOW,
            reason_text_en=(
                f"The request was submitted {days_elapsed} days after delivery, which exceeds "
                f"our standard {allowed_days}-day {decision_type.value} policy window."
            ),
            reason_text_ar=(
                f"تم تقديم الطلب بعد مرور {days_elapsed} يوماً من الاستلام، مما يتجاوز "
                f"فترة السماح المحددة بـ {allowed_days} يوماً لـ { 'الاسترجاع' if decision_type == PolicyDecisionType.RETURN else 'الاستبدال' }."
            ),
            requires_escalation=False,
            evaluated_facts=facts.model_dump(mode="json"),
        )
        if log_audit:
            log_decision_to_audit(decision)
        return decision

    # 9. All Constraints Passed -> Eligible
    decision = PolicyDecision(
        audit_id=audit_id,
        brand_id=policy.brand_id,
        decision_type=decision_type,
        eligible=True,
        reason_code=PolicyReasonCode.ELIGIBLE_WITHIN_WINDOW,
        reason_text_en=(
            f"Your item is eligible for {decision_type.value}. It is within the {allowed_days}-day "
            "window and meets all condition criteria. We will guide you through next steps."
        ),
        reason_text_ar=(
            f"طلب { 'الاسترجاع' if decision_type == PolicyDecisionType.RETURN else 'الاستبدال' } مؤهل بنجاح. "
            f"الطلب داخل فترة السماح ({allowed_days} يوماً) ومستوفٍ لكافة الشروط. سنوضح لك خطوات التسليم."
        ),
        requires_escalation=False,
        evaluated_facts=facts.model_dump(mode="json"),
    )
    if log_audit:
        log_decision_to_audit(decision)
    return decision
