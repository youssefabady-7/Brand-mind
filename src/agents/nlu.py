"""NLU Engine: Intent Classification, Entity Extraction, and Language Detection.

Coordinates:
- Language & Egyptian Arabic dialect detection (Risk R3)
- 9-class intent recognition
- Structured JSON entity extraction
- Tool necessity identification
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from src.agents.language import detect_language
from src.agents.llm_client import BaseLLMClient, get_llm_client
from src.agents.nlu_schemas import CustomerIntent, ExtractedEntities, NLUOutput
from src.prompts.engine import render_nlu_prompt

logger = logging.getLogger("brandmind.nlu")

CONFIG_DIR = Path("config")


def load_brand_config(brand_id: str, config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Loads brand configuration YAML."""
    config_file = config_dir / f"{brand_id.lower()}.yaml"
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found for brand: {brand_id}")
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_customer_message(
    message: str,
    brand_id: str,
    client: Optional[BaseLLMClient] = None,
) -> NLUOutput:
    """Parses customer free text into structured intent, entities, and tool needs.

    Args:
        message: Raw customer inquiry text (English, Arabic, or Egyptian Arabic).
        brand_id: Target brand identifier ('lumora' or 'urbankicks').
        client: Optional LLM client (defaults to configured system client).

    Returns:
        Validated NLUOutput object.
    """
    clean_message = message.strip()
    detected_lang = detect_language(clean_message)

    if not clean_message:
        return NLUOutput(
            intent=CustomerIntent.PRODUCT_INFO,
            confidence=0.5,
            entities=ExtractedEntities(),
            detected_language=detected_lang,
            tool_needed=None,
        )

    llm = client or get_llm_client()
    brand_cfg = load_brand_config(brand_id)

    prompt = render_nlu_prompt(clean_message, brand_cfg)
    messages = [{"role": "user", "content": prompt}]

    try:
        raw_output = llm.generate(messages=messages, temperature=0.0, json_mode=True)
        # Parse and sanitize output
        cleaned_json = raw_output.strip()
        if cleaned_json.startswith("```json"):
            cleaned_json = cleaned_json[7:]
        if cleaned_json.startswith("```"):
            cleaned_json = cleaned_json[3:]
        if cleaned_json.endswith("```"):
            cleaned_json = cleaned_json[:-3]
        cleaned_json = cleaned_json.strip()

        data = json.loads(cleaned_json)

        # Validate intent exists in 9-intent enum
        intent_str = data.get("intent", "").lower()
        matched_intent = None
        for intent_enum in CustomerIntent:
            if intent_enum.value == intent_str:
                matched_intent = intent_enum
                break

        if matched_intent is None:
            matched_intent = CustomerIntent.PRODUCT_INFO

        entities_data = data.get("entities", {})
        extracted_entities = ExtractedEntities(**entities_data)

        return NLUOutput(
            intent=matched_intent,
            confidence=float(data.get("confidence", 0.9)),
            entities=extracted_entities,
            detected_language=detected_lang,
            tool_needed=data.get("tool_needed"),
        )

    except Exception as exc:
        logger.error(f"Error parsing NLU response: {exc}")
        # Deterministic safe fallback
        return NLUOutput(
            intent=CustomerIntent.PRODUCT_INFO,
            confidence=0.5,
            entities=ExtractedEntities(),
            detected_language=detected_lang,
            tool_needed=None,
        )
