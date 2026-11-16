"""Unit tests for Jinja2 layered prompt templates.
"""

from pathlib import Path
import pytest
from src.agents.nlu import load_brand_config
from src.prompts.engine import (
    render_nlu_prompt,
    render_response_prompt,
    render_system_prompt,
)


def test_system_prompt_lumora():
    """Lumora system prompt renders calm persona and non-negotiable policy guardrails."""
    cfg = load_brand_config("lumora")
    prompt = render_system_prompt(cfg, language="ar_eg")
    assert "Lumora" in prompt
    assert "Radiance, simplified." in prompt
    assert "THE LLM NEVER INVENTS COMPANY POLICY" in prompt
    assert "calm" in prompt.lower()
    assert "لهجة مصرية" in prompt


def test_system_prompt_urbankicks():
    """UrbanKicks system prompt renders youthful streetwear tone."""
    cfg = load_brand_config("urbankicks")
    prompt = render_system_prompt(cfg, language="en")
    assert "UrbanKicks" in prompt
    assert "Step fresh, stay ahead." in prompt
    assert "youthful" in prompt.lower()
    assert "THE LLM NEVER INVENTS COMPANY POLICY" in prompt


def test_nlu_prompt_rendering():
    """NLU prompt embeds target intents, few-shot examples, and user query."""
    cfg = load_brand_config("lumora")
    prompt = render_nlu_prompt("Where is order LUM-1001?", cfg)
    assert "TARGET INTENTS" in prompt
    assert "order_status" in prompt
    assert "Where is order LUM-1001?" in prompt
    assert "FEW-SHOT EXAMPLES" in prompt


def test_response_prompt_rendering():
    """Response prompt layers context, tool observations, and history correctly."""
    cfg = load_brand_config("lumora")
    context = "<context>Shipping to Cairo takes 1-2 days.</context>"
    tool_obs = "Order LUM-1001: delivered on 2026-11-08"
    history = [{"user": "Hi", "assistant": "Welcome to Lumora"}]
    
    prompt = render_response_prompt(
        message="Did my package arrive?",
        brand_config=cfg,
        language="en",
        context_text=context,
        tool_name="get_order_status",
        tool_observation=tool_obs,
        history=history,
    )
    assert "<context>" in prompt
    assert "<tool_observation>" in prompt
    assert "get_order_status" in prompt
    assert "<conversation_history>" in prompt
    assert "Did my package arrive?" in prompt
