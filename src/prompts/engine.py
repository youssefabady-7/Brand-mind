"""Jinja2 Layered Prompt Template Engine.

Loads and renders:
1. System prompt (role, persona, do/don't guardrails, injection protection)
2. NLU prompt (intent classification & JSON entity extraction with few-shot examples)
3. Response prompt (grounded generation, tool observation narration, tone alignment)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_system_prompt(brand_config: Dict[str, Any], language: str = "en") -> str:
    """Renders Layer 1: Core System & Brand Identity Prompt."""
    template = jinja_env.get_template("system.j2")
    return template.render(brand=brand_config, language=language)


def render_nlu_prompt(message: str, brand_config: Dict[str, Any]) -> str:
    """Renders Layer 2: Intent Classification and JSON Entity Extraction Prompt."""
    template = jinja_env.get_template("nlu.j2")
    return template.render(message=message, brand=brand_config)


def render_response_prompt(
    message: str,
    brand_config: Dict[str, Any],
    language: str = "en",
    context_text: str = "",
    tool_name: Optional[str] = None,
    tool_observation: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Renders Layer 3: Grounded Brand Response Generation Prompt."""
    template = jinja_env.get_template("response.j2")
    return template.render(
        message=message,
        brand=brand_config,
        language=language,
        context_text=context_text,
        tool_name=tool_name,
        tool_observation=tool_observation,
        history=history or [],
    )
