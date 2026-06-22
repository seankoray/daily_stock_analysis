"""FastAPI-lifespan monitor for held A-share intraday guidance."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Set

from data_provider.base import normalize_stock_code
from src.core.trading_calendar import build_market_phase_context
from src.notification import NotificationService
from src.services.intraday_service import IntradayAnalysisService
from src.services.portfolio_service import PortfolioService

logger = logging.getLogger(__name__)


class IntradayMonitor:
    def __init__(self, config: Any):
        self.config = config
        self.service = IntradayAnalysisService()
        self.portfolio = PortfolioService()
        self.notifier = NotificationService()

    async def run_forever(self) -> None:
        interval = int(getattr(self.config, "intraday_monitor_interval_seconds", 60))
        while True:
            try:
                phase = build_market_phase_context(market="cn").phase.value
                if phase in {"intraday", "lunch_break", "closing_auction"}:
                    await asyncio.to_thread(self.run_once)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[intraday-monitor] iteration failed: %s", exc)
            await asyncio.sleep(interval)

    def run_once(self) -> Dict[str, Any]:
        snapshot = self.portfolio.get_portfolio_snapshot(cost_method="fifo")
        codes: Set[str] = set()
        for account in snapshot.get("accounts", []):
            for position in account.get("positions", []):
                code = normalize_stock_code(position.get("symbol", ""))
                if code.isdigit() and len(code) == 6 and float(position.get("quantity") or 0) > 0:
                    codes.add(code)
        results = []
        for code in sorted(codes):
            try:
                signal = self.service.refresh(code, notify=False)
                results.append(signal)
                if signal.get("signal_changed") and signal.get("status") == "triggered":
                    self.notifier.send(
                        self._format_signal(signal),
                        route_type="intraday_t_observation",
                        severity="warning",
                        dedup_key=signal.get("signal_key"),
                    )
            except Exception as exc:
                logger.warning("[intraday-monitor] %s refresh failed: %s", code, exc)
        return {"count": len(results), "results": results}

    @staticmethod
    def _format_signal(signal: Dict[str, Any]) -> str:
        direction = {
            "sell_then_buy": "先卖后买回观察",
            "buy_then_sell": "先买后卖底仓观察",
        }.get(signal.get("direction"), "观察")
        return (
            f"## 盘中做T观察｜{signal.get('code')}\n"
            f"- 方向：{direction}\n"
            f"- 建议上限：{signal.get('suggested_quantity', 0)} 股\n"
            f"- 买入候选：{signal.get('candidate_buy_range')}\n"
            f"- 卖出候选：{signal.get('candidate_sell_range')}\n"
            f"- 失效价：{signal.get('invalidation_price')}\n"
            f"- 说明：仅为辅助观察，不自动下单。"
        )
