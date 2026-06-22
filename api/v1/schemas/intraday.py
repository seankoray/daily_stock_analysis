"""Intraday bars and advisory-only T-trade schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class IntradayBarItem(BaseModel):
    timestamp: str
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    amount: Optional[float] = None
    source: Optional[str] = None
    is_complete: bool = True


class IntradayBarsResponse(BaseModel):
    stock_code: str
    interval: str
    data: List[IntradayBarItem] = Field(default_factory=list)


class IntradaySignalResponse(BaseModel):
    code: str
    status: str
    direction: str
    suggested_quantity: float = 0
    signal_key: Optional[str] = None
    reason: Optional[str] = None
    reasons: List[str] = Field(default_factory=list)
    candidate_buy_range: Optional[List[Optional[float]]] = None
    candidate_sell_range: Optional[List[Optional[float]]] = None
    trigger_price: Optional[float] = None
    invalidation_price: Optional[float] = None
    data_fresh: bool = False
    data_age_seconds: Optional[int] = None
    signal_changed: bool = False
    notification_requested: bool = False
    advisory_only: bool = True
    technical: Dict[str, Any] = Field(default_factory=dict)
    portfolio_context: Dict[str, Any] = Field(default_factory=dict)
