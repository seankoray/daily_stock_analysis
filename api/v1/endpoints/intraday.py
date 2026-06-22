"""A-share intraday observation endpoints. No order execution is exposed."""

from fastapi import APIRouter, Body, HTTPException, Query

from api.v1.schemas.intraday import (
    IntradayBarItem,
    IntradayBarsResponse,
    IntradaySignalResponse,
)
from data_provider.base import normalize_stock_code
from src.services.intraday_service import IntradayAnalysisService


router = APIRouter()


@router.post("/{stock_code}/refresh", response_model=IntradaySignalResponse)
def refresh_intraday(
    stock_code: str,
    account_id: int | None = Query(None, ge=1),
    notify: bool = Query(False),
) -> IntradaySignalResponse:
    try:
        payload = IntradayAnalysisService().refresh(
            stock_code,
            account_id=account_id,
            notify=notify,
        )
        return IntradaySignalResponse(**payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_request", "message": str(exc)})
    except Exception as exc:
        raise HTTPException(status_code=503, detail={"error": "intraday_unavailable", "message": str(exc)})


@router.get("/{stock_code}/signal", response_model=IntradaySignalResponse)
def get_intraday_signal(
    stock_code: str,
    account_id: int | None = Query(None, ge=1),
) -> IntradaySignalResponse:
    payload = IntradayAnalysisService().evaluate(stock_code, account_id=account_id)
    return IntradaySignalResponse(**payload)


@router.get("/{stock_code}/bars", response_model=IntradayBarsResponse)
def get_intraday_bars(
    stock_code: str,
    interval: str = Query("5m", pattern="^(5m|15m)$"),
    limit: int = Query(120, ge=20, le=800),
) -> IntradayBarsResponse:
    service = IntradayAnalysisService()
    frame = service.load_bars(normalize_stock_code(stock_code), interval="5m", limit=max(limit, 320))
    if interval == "15m":
        frame = service.aggregate_15m(frame)
    frame = frame.tail(limit)
    items = []
    for _, row in frame.iterrows():
        timestamp = row.get("timestamp")
        items.append(
            IntradayBarItem(
                timestamp=timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp),
                open=row.get("open"),
                high=row.get("high"),
                low=row.get("low"),
                close=row.get("close"),
                volume=row.get("volume"),
                amount=row.get("amount"),
                source=row.get("source"),
                is_complete=bool(row.get("is_complete", True)),
            )
        )
    return IntradayBarsResponse(
        stock_code=normalize_stock_code(stock_code),
        interval=interval,
        data=items,
    )


@router.post("/{stock_code}/cycle")
def update_intraday_cycle(
    stock_code: str,
    action: str = Body(..., embed=True, pattern="^(complete|invalidate|reset)$"),
    account_id: int | None = Query(None, ge=1),
) -> dict:
    try:
        return IntradayAnalysisService().update_cycle(
            stock_code,
            account_id=account_id,
            action=action,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": "invalid_request", "message": str(exc)})
