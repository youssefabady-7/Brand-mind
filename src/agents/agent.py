"""BrandMind Constrained Agent Loop.

Orchestration Architecture:
user message
  → load brand YAML, memory & language
  → NLU (intent + entity extraction)
  → co-reference entity resolution
  → vector RAG retrieval (top-4 ChromaDB + threshold fallback)
  → single decision point: "tool needed?"
  → layered prompt generation
  → draft reply
  → deterministic escalation checklist
  → honest hand-off or verified response
  → log audit & update session memory
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.agents.escalation import evaluate_escalation_checklist
from src.agents.language import detect_language
from src.agents.llm_client import BaseLLMClient, get_llm_client
from src.agents.memory import ConversationTurn, MemoryManager, SessionMemory
from src.agents.nlu import load_brand_config, parse_customer_message
from src.agents.nlu_schemas import CustomerIntent, ExtractedEntities, NLUOutput
from src.prompts.engine import render_response_prompt, render_system_prompt
from src.retrieval.retriever import KnowledgeRetriever, RetrievalResult, retrieve
from src.tools.tools import (
    check_return_eligibility,
    escalate_to_human,
    get_order_status,
    search_products,
)

logger = logging.getLogger("brandmind.agent")


class ToolCallRecord(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None


class AgentResponse(BaseModel):
    """The /chat response payload contract (Section 11 [REC])."""
    reply: str
    language: str
    intent: str
    entities: Dict[str, Any]
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCallRecord] = Field(default_factory=list)
    escalated: bool = False
    escalation_reason: Optional[str] = None
    latency_ms: float = 0.0


class BrandMindAgent:
    """Enterprise constrained agent loop coordinating tools, RAG, and memory."""

    def __init__(
        self,
        llm_client: Optional[BaseLLMClient] = None,
        retriever: Optional[KnowledgeRetriever] = None,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.llm = llm_client or get_llm_client()
        self.retriever = retriever or KnowledgeRetriever()
        self.memory = memory_manager or MemoryManager()

    def handle_message(
        self,
        session_id: str,
        brand_id: str,
        message: str,
    ) -> AgentResponse:
        """Processes a single conversational turn through the constrained loop.

        Args:
            session_id: Client session identifier for short-term memory.
            brand_id: Target multi-tenant brand ('lumora' or 'urbankicks').
            message: Customer free-text message.

        Returns:
            Structured AgentResponse.
        """
        start_time = time.perf_counter()
        clean_msg = message.strip()
        brand_id = brand_id.lower().strip()

        # 1. Load brand config and session state
        brand_cfg = load_brand_config(brand_id)
        session: SessionMemory = self.memory.get_or_create_session(session_id, brand_id)

        # 2. Language Detection
        language = detect_language(clean_msg)

        # 3. NLU Understanding (Intent & Entities)
        nlu_output: NLUOutput = parse_customer_message(clean_msg, brand_id, client=self.llm)

        # 4. Contextual Co-reference Entity Resolution with Session Memory
        resolved_order_id = session.resolve_entity("order_id", nlu_output.entities.order_id)
        resolved_product = session.resolve_entity("product", nlu_output.entities.product)

        entities_dict = nlu_output.entities.model_dump()
        entities_dict["order_id"] = resolved_order_id
        entities_dict["product"] = resolved_product

        # 5. ChromaDB Vector Retrieval
        rag_result: RetrievalResult = self.retriever.retrieve(
            query=clean_msg,
            brand_id=brand_id,
            top_k=4,
        )

        # 6. Single Decision Point: Tool Execution
        tool_calls: List[ToolCallRecord] = []
        tool_name: Optional[str] = None
        tool_observation_str: Optional[str] = None

        # Tool Route A: Order Status Lookup
        if (
            (nlu_output.intent == CustomerIntent.ORDER_STATUS or nlu_output.tool_needed == "get_order_status" or (resolved_order_id and "order" in clean_msg.lower()))
            and nlu_output.intent not in (CustomerIntent.RETURNS, CustomerIntent.EXCHANGE)
            and nlu_output.tool_needed != "check_return_eligibility"
        ):
            if resolved_order_id:
                tool_name = "get_order_status"
                order_data = get_order_status(resolved_order_id, brand_id=brand_id)
                tool_calls.append(
                    ToolCallRecord(
                        tool_name="get_order_status",
                        arguments={"order_id": resolved_order_id, "brand_id": brand_id},
                        output=order_data,
                    )
                )
                tool_observation_str = json.dumps(order_data, ensure_ascii=False)

        # Tool Route B: Deterministic Return/Exchange Policy Check
        elif (
            nlu_output.intent in (CustomerIntent.RETURNS, CustomerIntent.EXCHANGE)
            or nlu_output.tool_needed == "check_return_eligibility"
        ):
            facts_payload = {
                "order_id": resolved_order_id,
                "product_name": resolved_product,
                "days_since_delivery": nlu_output.entities.days_since_delivery,
                "is_opened": nlu_output.entities.is_opened,
                "is_used": nlu_output.entities.is_used,
                "is_defective": nlu_output.entities.is_defective or False,
                "is_allergic_reaction": nlu_output.entities.is_allergic_reaction or False,
                "reason": nlu_output.entities.reason or "changed_mind",
            }
            # Only execute rule engine if customer provided evaluation facts
            if any([
                facts_payload["days_since_delivery"] is not None,
                facts_payload["is_opened"] is not None,
                facts_payload["is_used"] is not None,
                facts_payload["is_allergic_reaction"],
                facts_payload["is_defective"],
            ]):
                tool_name = "check_return_eligibility"
                decision_type = "exchange" if nlu_output.intent == CustomerIntent.EXCHANGE else "return"
                decision_data = check_return_eligibility(
                    facts=facts_payload,
                    brand_id=brand_id,
                    decision_type=decision_type,
                )
                tool_calls.append(
                    ToolCallRecord(
                        tool_name="check_return_eligibility",
                        arguments={"facts": facts_payload, "brand_id": brand_id},
                        output=decision_data,
                    )
                )
                tool_observation_str = json.dumps(decision_data, ensure_ascii=False)

        # Tool Route C: Product Catalog Search
        elif (
            nlu_output.intent in (CustomerIntent.PRODUCT_INFO, CustomerIntent.AVAILABILITY)
            and nlu_output.tool_needed == "search_products"
        ):
            search_query = resolved_product or clean_msg
            products = search_products(search_query, brand_id=brand_id, max_results=3)
            tool_name = "search_products"
            tool_calls.append(
                ToolCallRecord(
                    tool_name="search_products",
                    arguments={"query": search_query, "brand_id": brand_id},
                    output=products,
                )
            )
            tool_observation_str = json.dumps(products, ensure_ascii=False)

        # 7. Layered Prompt Assembly
        system_prompt = render_system_prompt(brand_cfg, language=language)
        response_prompt = render_response_prompt(
            message=clean_msg,
            brand_config=brand_cfg,
            language=language,
            context_text=rag_result.context_text,
            tool_name=tool_name,
            tool_observation=tool_observation_str,
            history=session.get_recent_history(),
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": response_prompt},
        ]

        # 8. LLM Draft Generation
        draft_reply = self.llm.generate(messages=messages, temperature=0.2)

        # 9. Deterministic Escalation Checklist (Safety Gate)
        # Note: If tools already handled the inquiry (e.g. order lookup or eligibility),
        # do not trigger policy silence escalation!
        rag_check = rag_result if not tool_calls else None
        escalation_check = evaluate_escalation_checklist(
            user_message=clean_msg,
            draft_reply=draft_reply,
            intent=nlu_output.intent,
            entities=nlu_output.entities,
            confidence=nlu_output.confidence,
            rag_result=rag_check,
            brand_config=brand_cfg,
            language=language,
        )

        if escalation_check.should_escalate:
            final_reply = escalation_check.hand_off_message or draft_reply
            escalated = True
            escalation_reason = escalation_check.trigger
            # Audit the escalation ticket
            escalate_to_human(
                reason=escalation_reason or "SAFETY_CHECKLIST",
                context={"user_message": clean_msg, "intent": nlu_output.intent.value, "details": escalation_check.details},
                brand_id=brand_id,
                language=language,
            )
        else:
            final_reply = draft_reply
            escalated = False
            escalation_reason = None

        # 10. Update Session Memory
        turn_record = ConversationTurn(
            user_message=clean_msg,
            assistant_reply=final_reply,
            intent=nlu_output.intent.value,
            entities=entities_dict,
            tool_called=tool_name,
            escalated=escalated,
        )
        session.add_turn(turn_record)

        # 11. Format Output Sources for UI
        sources = [
            {
                "chunk_id": c.chunk_id,
                "doc_name": c.doc_name,
                "section": c.section,
                "similarity_score": c.similarity_score,
                "snippet": c.text[:280] + ("..." if len(c.text) > 280 else ""),
            }
            for c in rag_result.chunks
        ]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return AgentResponse(
            reply=final_reply,
            language=language,
            intent=nlu_output.intent.value,
            entities=entities_dict,
            sources=sources,
            tool_calls=tool_calls,
            escalated=escalated,
            escalation_reason=escalation_reason,
            latency_ms=latency_ms,
        )
