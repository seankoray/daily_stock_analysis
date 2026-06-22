"""Read-only portfolio context used by stock analysis and intraday guidance."""

from __future__ import annotations

from datetime import date
from typing import Any, Dict, Optional

from data_provider.base import normalize_stock_code
from src.services.portfolio_service import PortfolioService


class PortfolioContextService:
    def __init__(self, service: Optional[PortfolioService] = None):
        self.service = service or PortfolioService()

    def get_stock_context(
        self,
        stock_code: str,
        *,
        account_id: Optional[int] = None,
        as_of: Optional[date] = None,
    ) -> Dict[str, Any]:
        as_of_date = as_of or date.today()
        target = normalize_stock_code(stock_code)
        try:
            snapshot = self.service.get_portfolio_snapshot(
                account_id=account_id,
                as_of=as_of_date,
                cost_method="fifo",
            )
        except Exception as exc:
            return {
                "status": "unavailable",
                "is_held": False,
                "stock_code": target,
                "missing_reason": str(exc),
            }

        total_quantity = 0.0
        total_cost = 0.0
        market_value = 0.0
        unrealized_pnl = 0.0
        total_cash = 0.0
        matched_accounts = []
        for account in snapshot.get("accounts", []):
            total_cash += float(account.get("total_cash") or 0)
            matched_positions = []
            for position in account.get("positions", []):
                if normalize_stock_code(position.get("symbol", "")) != target:
                    continue
                quantity = float(position.get("quantity") or 0)
                total_quantity += quantity
                total_cost += float(position.get("total_cost") or 0)
                market_value += float(position.get("market_value_base") or 0)
                unrealized_pnl += float(position.get("unrealized_pnl_base") or 0)
                matched_positions.append(position)
            if matched_positions:
                matched_accounts.append(
                    {
                        "account_id": account.get("account_id"),
                        "account_name": account.get("account_name"),
                        "positions": matched_positions,
                    }
                )

        today_bought_quantity = self._today_bought_quantity(
            target,
            account_id=account_id,
            trade_date=as_of_date,
        )
        sellable_quantity = max(0.0, total_quantity - today_bought_quantity)
        avg_cost = total_cost / total_quantity if total_quantity > 0 else 0.0
        return {
            "status": "available",
            "stock_code": target,
            "is_held": total_quantity > 0,
            "account_id": account_id,
            "total_quantity": total_quantity,
            "sellable_quantity": sellable_quantity,
            "today_bought_quantity": today_bought_quantity,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "market_value": market_value,
            "unrealized_pnl": unrealized_pnl,
            "available_cash": total_cash,
            "matched_accounts": matched_accounts,
            "as_of": as_of_date.isoformat(),
        }

    def _today_bought_quantity(
        self,
        normalized_code: str,
        *,
        account_id: Optional[int],
        trade_date: date,
    ) -> float:
        try:
            trades = self.service.list_trade_events(
                account_id=account_id,
                date_from=trade_date,
                date_to=trade_date,
                symbol=normalized_code,
                side="buy",
                page=1,
                page_size=200,
            )
        except Exception:
            return 0.0
        return sum(float(item.get("quantity") or 0) for item in trades.get("items", []))
