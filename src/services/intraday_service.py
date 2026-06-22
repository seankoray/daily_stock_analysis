"""A-share intraday bar persistence and conservative T-trade observations."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, select

from data_provider.base import DataFetcherManager, normalize_stock_code
from src.indicators.bollinger import MultiTimeframeBollAnalyzer
from src.services.portfolio_context_service import PortfolioContextService
from src.storage import IntradayTradeState, StockIntradayBar, get_db


class IntradayAnalysisService:
    """Fetch 5m bars, derive 15m bars, and maintain one daily guidance cycle."""

    STALE_SECONDS = 8 * 60
    MAX_POSITION_FRACTION = 0.30

    def __init__(
        self,
        *,
        fetcher: Optional[DataFetcherManager] = None,
        portfolio: Optional[PortfolioContextService] = None,
    ):
        self.fetcher = fetcher or DataFetcherManager()
        self.portfolio = portfolio or PortfolioContextService()
        self.db = get_db()
        self.boll = MultiTimeframeBollAnalyzer()

    def refresh(
        self,
        code: str,
        *,
        account_id: Optional[int] = None,
        notify: bool = False,
    ) -> Dict[str, Any]:
        normalized = normalize_stock_code(code)
        if not normalized or not normalized.isdigit() or len(normalized) != 6:
            raise ValueError("盘中做T首期仅支持沪深A股代码")
        frame, source = self.fetcher.get_intraday_bars(
            normalized,
            interval="5m",
            count=320,
        )
        self._save_bars(normalized, frame, source=source)
        return self.evaluate(normalized, account_id=account_id, notify=notify)

    def evaluate(
        self,
        code: str,
        *,
        account_id: Optional[int] = None,
        notify: bool = False,
    ) -> Dict[str, Any]:
        normalized = normalize_stock_code(code)
        bars5 = self.load_bars(normalized, interval="5m", limit=320)
        portfolio = self.portfolio.get_stock_context(normalized, account_id=account_id)
        if bars5.empty:
            return self._watch_payload(normalized, portfolio, "分钟K数据缺失")

        bars15 = self.aggregate_15m(bars5)
        latest_ts = pd.Timestamp(bars5["timestamp"].iloc[-1]).to_pydatetime()
        age_seconds = max(0, int((datetime.now() - latest_ts).total_seconds()))
        fresh = age_seconds <= self.STALE_SECONDS
        technical = self._technical_packet(bars5, bars15)
        state = self._load_state(normalized, account_id=account_id)
        decision = self._decide(
            normalized,
            portfolio=portfolio,
            technical=technical,
            fresh=fresh,
            age_seconds=age_seconds,
            state=state,
        )
        changed = decision["signal_key"] != (state.last_signal_key if state else None)
        self._save_state(normalized, account_id, decision)
        decision["signal_changed"] = changed
        decision["notification_requested"] = bool(notify and changed)
        return decision

    def update_cycle(
        self,
        code: str,
        *,
        account_id: Optional[int] = None,
        action: str,
    ) -> Dict[str, Any]:
        """Acknowledge advisory lifecycle without placing an order."""
        if action not in {"complete", "invalidate", "reset"}:
            raise ValueError("action must be complete, invalidate, or reset")
        normalized = normalize_stock_code(code)
        with self.db.get_session() as session:
            row = session.execute(
                select(IntradayTradeState).where(
                    and_(
                        IntradayTradeState.code == normalized,
                        IntradayTradeState.trade_date == date.today(),
                        IntradayTradeState.account_id == account_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = IntradayTradeState(
                    code=normalized,
                    trade_date=date.today(),
                    account_id=account_id,
                )
                session.add(row)
            if action == "complete":
                row.phase = "completed"
                row.cycle_count = 1
            elif action == "invalidate":
                row.phase = "invalidated"
            else:
                row.phase = "watch"
                row.direction = "none"
                row.cycle_count = 0
                row.suggested_quantity = 0
            row.last_signal_key = f"{row.phase}:{datetime.now().isoformat()}"
            row.updated_at = datetime.now()
            session.commit()
            return {
                "code": normalized,
                "action": action,
                "phase": row.phase,
                "cycle_count": row.cycle_count,
                "advisory_only": True,
            }

    def load_bars(self, code: str, *, interval: str = "5m", limit: int = 320) -> pd.DataFrame:
        with self.db.get_session() as session:
            rows = session.execute(
                select(StockIntradayBar)
                .where(
                    and_(
                        StockIntradayBar.code == normalize_stock_code(code),
                        StockIntradayBar.interval == interval,
                    )
                )
                .order_by(desc(StockIntradayBar.timestamp))
                .limit(limit)
            ).scalars().all()
        records = [
            {
                "timestamp": row.timestamp,
                "open": row.open,
                "high": row.high,
                "low": row.low,
                "close": row.close,
                "volume": row.volume,
                "amount": row.amount,
                "source": row.data_source,
                "is_complete": row.is_complete,
                "fetched_at": row.fetched_at,
            }
            for row in reversed(rows)
        ]
        return pd.DataFrame(records)

    @staticmethod
    def aggregate_15m(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame()
        indexed = frame.copy()
        indexed["timestamp"] = pd.to_datetime(indexed["timestamp"])
        indexed = indexed.set_index("timestamp")
        result = indexed.resample("15min", label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        return result.dropna(subset=["close"]).reset_index()

    def _save_bars(self, code: str, frame: pd.DataFrame, *, source: str) -> None:
        now = datetime.now()
        with self.db.get_session() as session:
            for _, item in frame.iterrows():
                timestamp = pd.Timestamp(item["timestamp"]).to_pydatetime().replace(tzinfo=None)
                existing = session.execute(
                    select(StockIntradayBar).where(
                        and_(
                            StockIntradayBar.code == code,
                            StockIntradayBar.interval == "5m",
                            StockIntradayBar.timestamp == timestamp,
                        )
                    )
                ).scalar_one_or_none()
                values = {
                    "open": self._float(item.get("open")),
                    "high": self._float(item.get("high")),
                    "low": self._float(item.get("low")),
                    "close": self._float(item.get("close")),
                    "volume": self._float(item.get("volume")),
                    "amount": self._float(item.get("amount")),
                    "data_source": source,
                    "is_complete": timestamp + timedelta(minutes=5) <= now,
                    "fetched_at": now,
                }
                if existing is None:
                    session.add(
                        StockIntradayBar(
                            code=code,
                            interval="5m",
                            timestamp=timestamp,
                            **values,
                        )
                    )
                else:
                    for key, value in values.items():
                        setattr(existing, key, value)
            session.commit()

    def _technical_packet(self, bars5: pd.DataFrame, bars15: pd.DataFrame) -> Dict[str, Any]:
        five = self._intraday_boll(bars5)
        fifteen = self._intraday_boll(bars15)
        latest = bars5.iloc[-1]
        volume = pd.to_numeric(bars5["volume"], errors="coerce").fillna(0)
        amount = pd.to_numeric(bars5["amount"], errors="coerce").fillna(0)
        cumulative_volume = float(volume.sum())
        vwap = (
            float(amount.sum() / cumulative_volume)
            if cumulative_volume > 0 and float(amount.sum()) > 0
            else None
        )
        today = pd.Timestamp(latest["timestamp"]).date()
        today_rows = bars5[pd.to_datetime(bars5["timestamp"]).dt.date == today]
        return {
            "five_minute": five,
            "fifteen_minute": fifteen,
            "last_price": self._float(latest.get("close")),
            "vwap": vwap,
            "session_high": self._float(today_rows["high"].max()) if not today_rows.empty else None,
            "session_low": self._float(today_rows["low"].min()) if not today_rows.empty else None,
            "latest_timestamp": pd.Timestamp(latest["timestamp"]).isoformat(),
        }

    @staticmethod
    def _intraday_boll(frame: pd.DataFrame) -> Dict[str, Any]:
        if frame is None or len(frame) < 20:
            return {"available": False, "missing_reason": "requires_20_bars"}
        close = pd.to_numeric(frame["close"], errors="coerce")
        middle = close.rolling(20).mean()
        std = close.rolling(20).std(ddof=0)
        upper = middle + 2 * std
        lower = middle - 2 * std
        width = upper - lower
        latest = float(close.iloc[-1])
        slope = (
            (float(middle.iloc[-1]) - float(middle.iloc[-4])) / float(middle.iloc[-4]) * 100
            if len(frame) >= 23 and middle.iloc[-4]
            else 0.0
        )
        position = (
            "below_lower" if latest < lower.iloc[-1]
            else "above_upper" if latest > upper.iloc[-1]
            else "lower_half" if latest < middle.iloc[-1]
            else "upper_half"
        )
        return {
            "available": True,
            "lower": float(lower.iloc[-1]),
            "middle": float(middle.iloc[-1]),
            "upper": float(upper.iloc[-1]),
            "percent_b": float((latest - lower.iloc[-1]) / width.iloc[-1]) if width.iloc[-1] else None,
            "middle_slope_pct": slope,
            "position": position,
        }

    def _decide(
        self,
        code: str,
        *,
        portfolio: Dict[str, Any],
        technical: Dict[str, Any],
        fresh: bool,
        age_seconds: int,
        state: Optional[IntradayTradeState],
    ) -> Dict[str, Any]:
        reasons = []
        if not portfolio.get("is_held"):
            return self._watch_payload(code, portfolio, "未识别到持仓底仓", technical, age_seconds)
        if not fresh:
            return self._watch_payload(code, portfolio, "分钟K已过期", technical, age_seconds)
        if state and state.cycle_count >= 1:
            return self._watch_payload(code, portfolio, "今日已完成一轮做T", technical, age_seconds)

        five = technical.get("five_minute") or {}
        fifteen = technical.get("fifteen_minute") or {}
        if not five.get("available") or not fifteen.get("available"):
            return self._watch_payload(code, portfolio, "分钟K样本不足", technical, age_seconds)

        last_price = float(technical["last_price"])
        sell_conditions = 0
        buy_conditions = 0
        if five.get("position") in {"above_upper", "upper_half"}:
            sell_conditions += 1
            reasons.append("5分钟价格位于BOLL上半区")
        if fifteen.get("position") in {"above_upper", "upper_half"} and fifteen.get("middle_slope_pct", 0) <= 0:
            sell_conditions += 1
            reasons.append("15分钟上方承压")
        if technical.get("vwap") and last_price > technical["vwap"]:
            sell_conditions += 1
        if five.get("position") in {"below_lower", "lower_half"}:
            buy_conditions += 1
            reasons.append("5分钟价格位于BOLL下半区")
        if fifteen.get("position") in {"below_lower", "lower_half"} and fifteen.get("middle_slope_pct", 0) >= 0:
            buy_conditions += 1
            reasons.append("15分钟下方出现企稳")
        if technical.get("vwap") and last_price < technical["vwap"]:
            buy_conditions += 1

        sellable = float(portfolio.get("sellable_quantity") or 0)
        available_cash = float(portfolio.get("available_cash") or 0)
        quantity_cap = int(sellable * self.MAX_POSITION_FRACTION // 100 * 100)
        direction = "watch"
        quantity = 0
        candidate_low = five.get("lower")
        candidate_high = five.get("upper")
        invalidation = fifteen.get("lower")
        if sell_conditions >= 2 and quantity_cap >= 100:
            direction = "sell_then_buy"
            quantity = quantity_cap
        elif buy_conditions >= 2 and quantity_cap >= 100 and available_cash >= last_price * 100:
            direction = "buy_then_sell"
            cash_cap = int((available_cash / last_price) // 100 * 100)
            quantity = min(quantity_cap, cash_cap)

        signal_key = f"{direction}:{round(last_price, 3)}:{quantity}"
        return {
            "code": code,
            "status": "triggered" if direction != "watch" else "watch",
            "direction": direction,
            "signal_key": signal_key,
            "suggested_quantity": quantity,
            "max_position_fraction": self.MAX_POSITION_FRACTION,
            "candidate_buy_range": [candidate_low, five.get("middle")],
            "candidate_sell_range": [five.get("middle"), candidate_high],
            "trigger_price": last_price,
            "invalidation_price": invalidation,
            "reasons": reasons,
            "technical": technical,
            "portfolio_context": portfolio,
            "data_fresh": fresh,
            "data_age_seconds": age_seconds,
            "daily_cycle_limit": 1,
            "advisory_only": True,
        }

    @staticmethod
    def _watch_payload(
        code: str,
        portfolio: Dict[str, Any],
        reason: str,
        technical: Optional[Dict[str, Any]] = None,
        age_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        return {
            "code": code,
            "status": "watch",
            "direction": "watch",
            "signal_key": f"watch:{reason}",
            "suggested_quantity": 0,
            "reason": reason,
            "technical": technical or {},
            "portfolio_context": portfolio,
            "data_fresh": age_seconds is None or age_seconds <= IntradayAnalysisService.STALE_SECONDS,
            "data_age_seconds": age_seconds,
            "advisory_only": True,
        }

    def _load_state(self, code: str, *, account_id: Optional[int]) -> Optional[IntradayTradeState]:
        with self.db.get_session() as session:
            return session.execute(
                select(IntradayTradeState).where(
                    and_(
                        IntradayTradeState.code == code,
                        IntradayTradeState.trade_date == date.today(),
                        IntradayTradeState.account_id == account_id,
                    )
                )
            ).scalar_one_or_none()

    def _save_state(self, code: str, account_id: Optional[int], payload: Dict[str, Any]) -> None:
        with self.db.get_session() as session:
            row = session.execute(
                select(IntradayTradeState).where(
                    and_(
                        IntradayTradeState.code == code,
                        IntradayTradeState.trade_date == date.today(),
                        IntradayTradeState.account_id == account_id,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                row = IntradayTradeState(
                    code=code,
                    trade_date=date.today(),
                    account_id=account_id,
                )
                session.add(row)
            row.phase = payload.get("status", "watch")
            row.direction = payload.get("direction", "watch")
            row.suggested_quantity = float(payload.get("suggested_quantity") or 0)
            row.trigger_price = self._float(payload.get("trigger_price"))
            row.invalidation_price = self._float(payload.get("invalidation_price"))
            row.last_signal_key = payload.get("signal_key")
            row.payload = json.dumps(payload, ensure_ascii=False, default=str)
            row.updated_at = datetime.now()
            session.commit()

    @staticmethod
    def _float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None
