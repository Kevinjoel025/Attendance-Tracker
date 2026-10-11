"""
web/routes/chat.py
──────────────────
Natural language assistant API endpoint powered by the pre-initialized ToolRouter.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from attendance_tracker.web.state import get_app_services

router = APIRouter(prefix="/api", tags=["Natural Language Assistant"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="Natural language user query")


class ToolDataResponse(BaseModel):
    success: bool
    intent: str
    tool_name: str
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    subject: Optional[str] = None


class NLUDataResponse(BaseModel):
    intent: str
    subject: Optional[str] = None
    target: Optional[float] = None
    count: Optional[int] = None
    confidence: float = 0.0


class ChatResponse(BaseModel):
    query: str
    response: str
    response_text: str
    intent: str
    tool: str
    subject: Optional[str] = None
    target: Optional[float] = None
    count: Optional[int] = None
    latency_ms: float
    total_time_ms: float
    nlu: NLUDataResponse
    tool_data: ToolDataResponse


@router.post("/ask", response_model=ChatResponse)
def ask_assistant(req: ChatRequest) -> ChatResponse:
    """
    Process a natural language student query through the frozen SLM and ToolRouter pipeline.
    """
    query_text = req.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    services = get_app_services()
    resp = services.tool_router.process(query_text)

    tool_subj = resp.tool.subject.short_name if resp.tool.subject else resp.nlu.subject

    return ChatResponse(
        query=resp.query,
        response=resp.response_text,
        response_text=resp.response_text,
        intent=resp.nlu.intent,
        tool=resp.tool.tool_name,
        subject=tool_subj,
        target=resp.nlu.target,
        count=resp.nlu.count,
        latency_ms=round(resp.total_time_ms, 2),
        total_time_ms=round(resp.total_time_ms, 2),
        nlu=NLUDataResponse(
            intent=resp.nlu.intent,
            subject=resp.nlu.subject,
            target=resp.nlu.target,
            count=resp.nlu.count,
            confidence=round(resp.nlu.confidence, 4),
        ),
        tool_data=ToolDataResponse(
            success=resp.tool.success,
            intent=resp.tool.intent,
            tool_name=resp.tool.tool_name,
            error_message=resp.tool.error_message,
            error_type=resp.tool.error_type,
            subject=tool_subj,
        ),
    )
