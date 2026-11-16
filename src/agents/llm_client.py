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
            text = match.group(1).lower().strip()
        else:
            text = prompt.lower().strip()

        # Extract order ID if present (e.g. LUM-1001, UK-2001)
        order_match = re.search(r"\b(lum-\d{4}|uk-\d{4})\b", text, re.IGNORECASE)
        order_id = order_match.group(1).upper() if order_match else None

        # Extract days elapsed if present
        days_match = re.search(r"(\d+)\s*(days|يوم|أيام)", text)
        days = int(days_match.group(1)) if days_match else None

        # Extract product mentions
        product = None
        for p in ["niacinamide", "serum", "cleanser", "moisturizer", "toner", "sunscreen", "sneaker", "hoodie", "pants"]:
            if p in text:
                product = p
                break

        # Default entities
        entities = {
            "product": product,
            "order_id": order_id,
            "date": None,
            "days_since_delivery": days,
            "size": None,
            "colour": None,
            "reason": None,
            "is_opened": None,
            "is_used": None,
            "is_defective": False,
            "is_allergic_reaction": False,
        }

        # Intent Detection Rules
        if any(w in text for w in ["حرقان", "وارم", "التهاب", "allergic", "reaction", "rash", "burning", "irritation"]):
            intent = "complaint"
            tool_needed = "escalate_to_human"
            entities["is_allergic_reaction"] = True
            entities["reason"] = "allergic_reaction"
            confidence = 0.98

        elif any(w in text for w in ["بني آدم", "خدمة العملاء", "human", "representative", "agent", "manager", "كلمني حد"]):
            intent = "human_support"
            tool_needed = "escalate_to_human"
            confidence = 0.99

        elif order_id or any(w in text for w in ["order status", "وصل لفين", "track", "تتبع", "أوردر", "فين"]):
            intent = "order_status"
            tool_needed = "get_order_status"
            confidence = 0.96

        elif any(w in text for w in ["تبديل", "استبدال", "exchange", "swap", "مقاس"]):
            intent = "exchange"
            tool_needed = "check_return_eligibility"
            entities["is_opened"] = False if "unopened" in text or "ماتفتحش" in text else None
            confidence = 0.95

        elif any(w in text for w in ["ارجع", "استرجاع", "return", "ترجيع"]):
            intent = "returns"
            tool_needed = "check_return_eligibility"
            if "unopened" in text or "sealed" in text or "ماتفتحش" in text or "سليم" in text:
                entities["is_opened"] = False
            elif re.search(r"\bopened\b", text) or "فتحته" in text or "مفتوح" in text or "unsealed" in text:
                entities["is_opened"] = True
            if "used" in text or "جربت" in text or "outdoor" in text or "لبسته" in text:
                entities["is_used"] = True
            confidence = 0.95

        elif any(w in text for w in ["فلوس", "refund", "استرداد", "فلوسي"]):
            intent = "refund"
            tool_needed = None
            confidence = 0.94

        elif any(w in text for w in ["شحن", "shipping", "delivery", "توصيل", "كام"]):
            intent = "shipping"
            tool_needed = None
            confidence = 0.97

        elif any(w in text for w in ["سيروم", "روتين", "routine", "how to use", "طريقة", "ingredients", "مكونات"]):
            intent = "product_info"
            tool_needed = "search_products" if "recommend" in text or "have" in text else None
            confidence = 0.93

        elif any(w in text for w in ["available", "متوفر", "موجود", "stock"]):
            intent = "availability"
            tool_needed = "search_products"
            confidence = 0.92

        else:
            intent = "complaint" if any(w in text for w in ["angry", "bad", "سيء", "شكوى"]) else "product_info"
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

        if "tool_observation" in prompt:
            if "ELIGIBLE_WITHIN_WINDOW" in prompt:
                if is_arabic:
                    return "يسعدنا إبلاغك بأن طلبك مؤهل بنجاح طبقاً لفترة السماح (14 يوماً). سيقوم مندوب الشحن بالتنسيق معك لإتمام العملية."
                return "We are pleased to inform you that your request is fully eligible within our 14-day window. Our courier will coordinate collection with you."
            elif "OPENED_HYGIENE_RESTRICTION" in prompt:
                if is_arabic:
                    return "نعتذر منك، ولكن طبقاً لمعايير الصحة العامة وسلامة المستحضرات، لا يمكن استرجاع أو استبدال المنتجات بعد فتح الغلاف أو إزالة لاصق الأمان."
                return "We sincerely apologize, but due to public health and cosmetics hygiene regulations, products cannot be returned or exchanged once the safety seal is broken."
            elif "delivered" in prompt.lower():
                if is_arabic:
                    return "طلبك تم تسليمه بنجاح عبر شركة الشحن. إذا كان لديك أي استفسار حول المنتجات فنحن دائماً في خدمتك."
                return "Your order has been successfully delivered by the courier. Please let us know if we can assist you with your ritual."

        if is_lumora:
            if is_arabic:
                return "أهلاً بيك في لومورا. نوفر شحن لجميع المحافظات خلال 1 إلى 3 أيام عمل، مع شحن مجاني للطلبات فوق 800 جنيه. يسعدنا دائماً العناية ببشرتك."
            return "Welcome to Lumora. We offer nationwide delivery across Egypt within 1-3 business days, with complimentary shipping on orders over 800 EGP."
        else:
            if is_arabic:
                return "يا هلا بيك في إيربان كيكس! الشحن بياخد من يوم ليومين في القاهرة والجيزة ومجاني فوق 1500 جنيه. منور يا كوتش!"
            return "Yo! Welcome to UrbanKicks. We ship within 1-2 days across Cairo & Giza, with free shipping on orders over 1500 EGP. Step fresh!"


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
