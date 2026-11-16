"""Provider-Agnostic LLM Client Interface.

Supports:
1. OpenAI-compatible chat completions API (OpenAI, Ollama, vLLM, Azure).
2. Amazon Bedrock (via boto3).
3. RuleBasedMockLLMClient (Zero-cost offline deterministic client for unit tests and local runs).
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("brandmind.llm")


class BaseLLMClient(ABC):
    """Abstract interface for LLM provider operations."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        """Generates completion text from chat messages."""
        pass


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.model_name = model_name or os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock Foundation Models."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        self.model_id = model_id or os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
        self.region_name = region_name or os.getenv("AWS_REGION", "us-east-1")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region_name)
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        client = self._get_client()
        # Convert messages to Bedrock Converse API format
        converse_messages = []
        system_prompts = []
        for m in messages:
            if m["role"] == "system":
                system_prompts.append({"text": m["content"]})
            else:
                converse_messages.append({"role": m["role"], "content": [{"text": m["content"]}]})

        response = client.converse(
            modelId=self.model_id,
            messages=converse_messages,
            system=system_prompts if system_prompts else None,
            inferenceConfig={"temperature": temperature},
        )
        output_text = response["output"]["message"]["content"][0]["text"]
        return output_text


class RuleBasedMockLLMClient(BaseLLMClient):
    """High-fidelity deterministic offline LLM simulator.
    
    Permits comprehensive end-to-end testing and demonstrations without paid API keys.
    """

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> str:
        user_message = ""
        for m in reversed(messages):
            if m["role"] == "user":
                user_message = m["content"]
                break

        # Check if this is an NLU intent & entity extraction prompt
        if "TARGET INTENTS" in user_message or json_mode:
            return self._mock_nlu_extraction(user_message)

        # Otherwise, this is a response generation prompt
        return self._mock_response_generation(user_message, messages)

    def _mock_nlu_extraction(self, prompt: str) -> str:
        # Extract the user's actual query from the rendered template prompt
        match = re.search(r'Customer message:\s*"(.*?)"', prompt, re.DOTALL | re.IGNORECASE)
        if match:
            raw_text = match.group(1).lower().strip()
        else:
            raw_text = prompt.lower().strip()

        # Multilingual normalization for Egyptian Arabic
        norm_text = raw_text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
        norm_text = norm_text.replace("ة", "ه").replace("ى", "ي")

        # Extract order ID if present (e.g. LUM-1001, UK-2001)
        order_match = re.search(r"\b(lum-\d{4}|uk-\d{4})\b", raw_text, re.IGNORECASE)
        order_id = order_match.group(1).upper() if order_match else None

        # Extract days elapsed if present (digits or "today" / "النهاردة")
        days = None
        days_match = re.search(r"(\d+)\s*(days|يوم|ايام)", norm_text)
        if days_match:
            days = int(days_match.group(1))
        elif any(w in norm_text for w in ["today", "النهارده", "اليوم"]):
            days = 0

        # Extract product mentions
        product = None
        for p in ["niacinamide", "serum", "cleanser", "moisturizer", "toner", "sunscreen", "sneaker", "hoodie", "pants"]:
            if p in raw_text or p in norm_text:
                product = p
                break
        if not product and any(w in norm_text for w in ["سيروم", "مرطب", "تونر", "غسول", "واقي", "سنيكرز", "كوتش"]):
            product = "skincare_or_sneaker"

        # Condition and defect flags
        is_opened = None
        if any(w in norm_text for w in ["unopened", "sealed", "ماتفتحش", "سليم", "في العلبه"]):
            is_opened = False
        elif re.search(r"\bopened\b", raw_text) or any(w in norm_text for w in ["فتحته", "مفتوح", "unsealed"]):
            is_opened = True

        is_used = None
        if any(w in norm_text for w in ["unworn", "never worn", "clean soles", "ماتلبسش", "مش ملبوس"]):
            is_used = False
        elif any(w in norm_text for w in ["used", "outdoor", "scuff", "جربت", "نقطتين", "لبسته"]):
            is_used = True

        is_defective = any(w in norm_text for w in ["مكسور", "مسرب", "تالف", "broken", "leaking", "damaged", "defective"])
        is_allergic = any(w in norm_text for w in ["حرقان", "وارم", "التهاب", "احمرار", "allergic", "reaction", "rash", "burning", "irritation"])

        entities = {
            "product": product,
            "order_id": order_id,
            "date": None,
            "days_since_delivery": days,
            "size": None,
            "colour": None,
            "reason": "allergic_reaction" if is_allergic else ("defective" if is_defective else None),
            "is_opened": is_opened,
            "is_used": is_used,
            "is_defective": is_defective,
            "is_allergic_reaction": is_allergic,
        }

        # 1. Safety & Critical Escalations (Allergies, Medical, Legal Threats, Fraud)
        if is_allergic:
            intent = "complaint"
            tool_needed = "escalate_to_human"
            confidence = 0.99

        elif any(w in norm_text for w in ["fraud", "charged twice", "نصب", "سرقه", "rude", "mud", "بلاغ", "حمايه المستهلك", "server", "encryption", "franchise", "legal"]):
            intent = "complaint"
            tool_needed = "escalate_to_human"
            confidence = 0.98

        # 2. Human Support Inquiries
        elif any(w in norm_text for w in ["بني ادم", "خدمه عملاء", "خدمه العملاء", "العملاء", "مش روبوت", "روبوت", "موظف", "بشري", "انسان", "human", "representative", "agent", "manager", "مدير", "مشرف", "كلمني حد"]):
            intent = "human_support"
            tool_needed = "escalate_to_human"
            confidence = 0.99

        # 3. Exchanges
        elif any(w in norm_text for w in ["تبديل", "استبدال", "exchange", "swap", "مقاس"]):
            intent = "exchange"
            tool_needed = "check_return_eligibility"
            confidence = 0.96

        # 4. Returns & Defective Claims
        elif is_defective or any(w in norm_text for w in ["ارجع", "استرجاع", "return", "ترجيع"]):
            intent = "returns"
            tool_needed = "check_return_eligibility"
            confidence = 0.96

        # 5. Order Tracking
        elif order_id or any(w in norm_text for w in ["order status", "وصل لفين", "وصل فين", "track", "تتبع", "اوردر", "فين"]):
            intent = "order_status"
            tool_needed = "get_order_status"
            confidence = 0.97

        # 6. Shipping & Delivery FAQs
        elif any(w in norm_text for w in ["شحن", "shipping", "delivery", "توصيل", "كام يوم", "بيتكلف كام"]):
            intent = "shipping"
            tool_needed = None
            confidence = 0.97

        # 7. Refund Policy Inquiries
        elif any(w in norm_text for w in ["فلوس", "refund", "استرداد", "فلوسي"]):
            intent = "refund"
            tool_needed = None
            confidence = 0.94

        # 8. Product Search & Catalog Availability
        elif any(w in norm_text for w in ["do you have", "have a", "looking for", "recommend", "barrier repair", "available", "متوفر", "موجود", "stock", "عايز كوتش", "عندكم"]):
            intent = "product_info" if "serum" in raw_text or "cream" in raw_text else "availability"
            tool_needed = "search_products"
            confidence = 0.94

        # 9. General Product Info & Usage Guides
        elif any(w in norm_text for w in ["serum", "سيروم", "routine", "how to use", "طريقه", "ingredients", "مكونات"]):
            intent = "product_info"
            tool_needed = None
            confidence = 0.93

        else:
            intent = "complaint" if any(w in norm_text for w in ["angry", "bad", "سيء", "شكوى", "delay", "تاخير"]) else "product_info"
            tool_needed = None
            confidence = 0.85

        return json.dumps({
            "intent": intent,
            "confidence": confidence,
            "tool_needed": tool_needed,
            "entities": entities,
        }, ensure_ascii=False)

    def _mock_response_generation(self, prompt: str, messages: List[Dict[str, str]]) -> str:
        # Check system instructions for brand tone
        system_text = messages[0]["content"] if messages else ""
        is_lumora = "lumora" in system_text.lower()
        is_arabic = any("\u0600" <= c <= "\u06FF" for c in prompt)

        # 1. Tool Observation Grounded Replies
        if "tool_observation" in prompt:
            # Return / Exchange Eligibility
            if "ELIGIBLE_WITHIN_WINDOW" in prompt:
                if is_arabic:
                    return "يسعدنا إبلاغك بأن طلبك مؤهل للاسترجاع/الاستبدال طبقاً لفترة السماح (14 يوماً) والغلاف سليم. سيقوم مندوبنا بالتنسيق معك لاستلام المنتج."
                return "We are pleased to inform you that your request is eligible within our 14-day window with unopened seal. Instructions: our courier will collect the item."
            elif "OPENED_HYGIENE_RESTRICTION" in prompt:
                if is_arabic:
                    return "نعتذر منك بشدة، فالمنتج غير مؤهل للاسترجاع لأنه مفتوح وتم استخدامه، وذلك حفاظاً على الصحة والسلامة العامة لمستحضرات العناية."
                return "We sincerely apologize, but this item is ineligible for return because it has been opened, adhering to public health and hygiene safety standards."
            elif "EXCEEDED_RETURN_WINDOW" in prompt:
                if is_arabic:
                    return "نعتذر منك، الطلب غير مؤهل لأن فترة الاسترجاع المسموحة (14 يوماً) قد انتهت بالفعل."
                return "We apologize, but your request is ineligible because it exceeds our 14 days policy window."
            elif "DEFECTIVE_REPLACEMENT_ACCEPTED" in prompt:
                if is_arabic:
                    return "نعتذر بشدة عن وصول المنتج تالف أثناء الشحن! طلبك مؤهل لاستبدال فوري مجاني طبقاً لضمان الجودة الخاص بنا."
                return "We sincerely apologize that your item arrived damaged. You are eligible for an immediate free replacement under our quality guarantee."
            elif "WORN_FOOTWEAR_RESTRICTION" in prompt:
                if is_arabic:
                    return "نعتذر منك يا كوتش، السنيكرز غير مؤهل للاسترجاع لأنه ملبوس وعليه آثار استخدام، وسياسة الاسترجاع تشترط أن يكون غير ملبوس ونظيف تماماً."
                return "We apologize, but this pair is ineligible for return because outdoor wear was detected. Items must be in unworn condition with clean soles."
            elif "OUTDOOR_WEAR_DETECTED" in prompt:
                return "We are sorry, but the item is ineligible due to outdoor wear. Returns require unworn condition with clean soles."

            # Order Status Lookup
            if "out_for_delivery" in prompt.lower():
                if is_arabic:
                    return "طلبك حالياً out_for_delivery مع المندوب وفي طريقه إليك في Maadi. رقم التتبع BST-EG-91004."
                return "Your order is currently out_for_delivery with our courier on the way to Maadi (Tracking: BST-EG-91004)."
            elif "delivered" in prompt.lower():
                if is_arabic:
                    return "طلبك تم تسليمه بنجاح (delivered) للمستلم Nour El-Din عبر رقم التتبع BST-EG-91001. نتمنى لك تجربة رائعة!"
                return "Your order has been delivered to Nour El-Din under tracking BST-EG-91001."
            elif "in_transit" in prompt.lower():
                return "Your order is in_transit and on its way to your destination."

            # Product Search
            if "Niacinamide" in prompt or "Ceramide" in prompt:
                return "We recommend our 5-Ceramide Barrier Repair Cream (520 EGP) and Hydrating Niacinamide 10% Serum (480 EGP) specifically formulated for barrier repair."

        # 2. Knowledge-Grounded FAQ Replies
        if "cairo" in prompt.lower() or "القاهرة" in prompt:
            if is_lumora:
                return "Welcome to Lumora. Shipping and delivery to Cairo is 45 EGP within 1 to 2 business days. We offer free shipping on orders over 800 EGP."
            return "Yo! Welcome to UrbanKicks. Shipping and delivery to Cairo is 50 EGP within 1 to 2 business days, with free shipping over 1500 EGP. Step fresh!"

        if "alexandria" in prompt.lower() or "إسكندرية" in prompt or "اسكندرية" in prompt:
            if is_lumora:
                if is_arabic:
                    return "مصاريف الشحن للإسكندرية 60 جنيه والتوصيل يستغرق من 2 إلى 3 أيام عمل، مع شحن مجاني فوق 800 جنيه."
                return "Shipping to Alexandria is 60 EGP with delivery in 2 to 3 business days, and free shipping over 800 EGP."

        if "giza" in prompt.lower() or "الجيزة" in prompt:
            if not is_lumora:
                return "Shipping to Giza is 50 EGP within 1 to 2 business days, with free shipping on orders above 1500 EGP."
            return "Shipping to Giza is 45 EGP within 1 to 2 business days."

        if "vitamin c" in prompt.lower() or "فيتامين سي" in prompt:
            if is_arabic:
                return "طريقة استخدام سيروم فيتامين سي 15%: يُستخدم صباحاً على بشرة نظيفة وجافة، ثم يُتبع بمرطب وواقي شمس لحمايته من الأكسدة، ويُفضل حفظه في درجة حرارة باردة."
            return "How to use Vitamin C: Apply every morning on clean dry skin, follow with sunscreen to prevent oxidation, and store in a cool dry place."

        if is_lumora:
            if is_arabic:
                return "أهلاً بك في لومورا. نوفر شحن لجميع المحافظات خلال 1 إلى 3 أيام عمل، وشحن مجاني فوق 800 جنيه."
            return "Welcome to Lumora. We offer nationwide delivery across Egypt within 1-3 business days, with free shipping over 800 EGP."
        else:
            if is_arabic:
                return "يا هلا بيك في إيربان كيكس! الشحن بياخد من يوم ليومين ومجاني فوق 1500 جنيه. منور يا كوتش!"
            return "Yo! Welcome to UrbanKicks. We ship within 1-2 days across Cairo & Giza, with free shipping on orders over 1500 EGP."


def get_llm_client() -> BaseLLMClient:
    """Factory creating configured LLM client from environment settings."""
    provider = os.getenv("LLM_PROVIDER", "mock").lower()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key and not api_key.startswith("your-"):
            return OpenAIClient()
        logger.warning("OPENAI_API_KEY not configured. Falling back to RuleBasedMockLLMClient.")
        return RuleBasedMockLLMClient()

    if provider == "bedrock":
        if os.getenv("AWS_ACCESS_KEY_ID"):
            return BedrockClient()
        logger.warning("AWS credentials not configured. Falling back to RuleBasedMockLLMClient.")
        return RuleBasedMockLLMClient()

    return RuleBasedMockLLMClient()
