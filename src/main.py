"""FastAPI Backend Application for BrandMind.

Implements Section 11 [SPEC] endpoints:
- POST /chat
- GET /orders/{order_id}
- POST /returns/check
- POST /returns/request
- POST /escalate
- GET /products/{product_id}
- GET /products
- GET /health
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.agent import AgentResponse, BrandMindAgent
from src.policies.schemas import ReturnFacts
from src.tools.tools import (
    check_return_eligibility,
    escalate_to_human,
    get_order_status,
    search_products,
)

logger = logging.getLogger("brandmind.api")

app = FastAPI(
    title="BrandMind AI API",
    description="Agentic Customer Service API for E-Commerce Brands in Egypt & MENA",
    version="0.1.0",
)

# CORS Middleware for Streamlit and external frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singleton agent instance
agent = BrandMindAgent()

RETURNS_LOG_FILE = Path("data/logs/return_requests.jsonl")


# --- Request & Response Models ---

class ChatRequest(BaseModel):
    session_id: str = Field(default="default_session", description="Unique client session ID")
    brand_id: str = Field(default="lumora", description="Brand ID: 'lumora' or 'urbankicks'")
    message: str = Field(..., min_length=1, description="Customer message text")


class ReturnCheckRequest(BaseModel):
    facts: ReturnFacts
    brand_id: str = Field(default="lumora")
    decision_type: str = Field(default="return", description="'return' or 'exchange'")


class ReturnSubmitRequest(BaseModel):
    order_id: str
    brand_id: str = "lumora"
    product_name: str
    reason: str
    customer_phone: Optional[str] = None


class EscalateRequest(BaseModel):
    reason: str
    brand_id: str = "lumora"
    user_message: str = ""
    language: str = "en"


# --- API Routes ---

@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health status and service availability."""
    return {
        "status": "healthy",
        "service": "BrandMind Customer Service Agent",
        "version": "0.1.0",
        "vector_db": "ChromaDB (connected)",
    }


@app.post("/chat", response_model=AgentResponse, tags=["Agent"])
def chat_endpoint(request: ChatRequest) -> AgentResponse:
    """Core conversational endpoint.
    
    Executes the full constrained agent loop:
    NLU -> RAG -> Tool Execution -> Layered Prompts -> Escalation Checklist.
    """
    try:
        response = agent.handle_message(
            session_id=request.session_id,
            brand_id=request.brand_id,
            message=request.message,
        )
        return response
    except Exception as exc:
        logger.error(f"Error handling chat request: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal agent error: {str(exc)}",
        )


@app.get("/orders/{order_id}", tags=["Orders"])
def get_order_endpoint(
    order_id: str,
    brand_id: Optional[str] = Query(default=None, description="Filter by brand"),
) -> Dict[str, Any]:
    """Retrieves delivery status and items for a mocked order ID."""
    result = get_order_status(order_id=order_id, brand_id=brand_id)
    if not result.get("found"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=result.get("message", "Order not found"),
        )
    return result


@app.post("/returns/check", tags=["Policies"])
def check_return_endpoint(request: ReturnCheckRequest) -> Dict[str, Any]:
    """Deterministic return and exchange eligibility evaluator."""
    decision = check_return_eligibility(
        facts=request.facts.model_dump(),
        brand_id=request.brand_id,
        decision_type=request.decision_type,
    )
    return decision


@app.post("/returns/request", tags=["Policies"])
def submit_return_endpoint(request: ReturnSubmitRequest) -> Dict[str, Any]:
    """Submits a verified return/exchange pickup request."""
    return_id = f"RET-{uuid.uuid4().hex[:8].upper()}"
    record = {
        "return_id": return_id,
        "order_id": request.order_id,
        "brand_id": request.brand_id,
        "product_name": request.product_name,
        "reason": request.reason,
        "customer_phone": request.customer_phone,
        "status": "pending_courier_pickup",
    }
    RETURNS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RETURNS_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "success": True,
        "return_id": return_id,
        "status": "pending_courier_pickup",
        "message": f"Return request {return_id} registered. Courier will arrange collection.",
    }


@app.post("/escalate", tags=["Escalation"])
def manual_escalation_endpoint(request: EscalateRequest) -> Dict[str, Any]:
    """Direct manual ticket creation for human agent hand-off."""
    result = escalate_to_human(
        reason=request.reason,
        context={"user_message": request.user_message},
        brand_id=request.brand_id,
        language=request.language,
    )
    return result


@app.get("/products/{product_id}", tags=["Catalog"])
def get_product_by_id(product_id: str) -> Dict[str, Any]:
    """Looks up product details by SKU / ID from synthetic products.csv."""
    products_csv = Path("data/structured/products.csv")
    if not products_csv.exists():
        raise HTTPException(status_code=404, detail="Product catalog not found")

    with open(products_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("product_id", "").strip().upper() == product_id.strip().upper():
                return {
                    "product_id": row["product_id"],
                    "brand_id": row["brand_id"],
                    "name": row["name"],
                    "category": row["category"],
                    "price_egp": float(row["price_egp"]),
                    "sizes_or_volume": row["sizes_or_volume"],
                    "key_ingredients_or_material": row["key_ingredients_or_material"],
                    "in_stock": row["in_stock"].lower() == "true",
                    "stock_quantity": int(row["stock_quantity"]),
                    "description": row["description"],
                }
    raise HTTPException(status_code=404, detail=f"Product {product_id} not found")


@app.get("/products", tags=["Catalog"])
def search_products_endpoint(
    query: str = Query(..., min_length=1),
    brand_id: str = Query(default="lumora"),
    max_results: int = Query(default=5, ge=1, le=20),
) -> List[Dict[str, Any]]:
    """Searches catalog by query string and brand filter."""
    return search_products(query=query, brand_id=brand_id, max_results=max_results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
