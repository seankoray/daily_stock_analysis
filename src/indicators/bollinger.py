"""Daily/weekly Bollinger Band analysis.

The analyzer is intentionally independent from trading-score policy.  It
normalizes daily bars, aggregates weekly bars, calculates standard 20-period
bands with population standard deviation, and emits compact structural labels
that downstream scorers and prompts can consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class BollingerSnapshot:
    available: bool = False
    timeframe: str = "daily"
    period: int = 20
    stddev_multiplier: float = 2.0
    sample_count: int = 0
    as_of: Optional[str] = None
    is_partial: bool = False
    current_price: Optional[float] = None
    middle: Optional[float] = None
    upper: Optional[float] = None
    lower: Optional[float] = None
    bandwidth_pct: Optional[float] = None
    percent_b: Optional[float] = None
    middle_slope_pct: Optional[float] = None
    width_state: str = "unavailable"
    position: str = "unavailable"
    event: str = "none"
    missing_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MultiTimeframeBollResult:
    daily: BollingerSnapshot
    weekly: BollingerSnapshot
    confluence: str = "unavailable"
    summary: str = "BOLL 数据不足"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "daily": self.daily.to_dict(),
            "weekly": self.weekly.to_dict(),
            "confluence": self.confluence,
            "summary": self.summary,
        }


class MultiTimeframeBollAnalyzer:
    PERIOD = 20
    MULTIPLIER = 2.0

    def analyze(
        self,
        daily_df: pd.DataFrame,
        *,
        include_partial_week: bool = True,
    ) -> MultiTimeframeBollResult:
        normalized = self._normalize_daily(daily_df)
        daily = self._snapshot(normalized, timeframe="daily", is_partial=False)
        weekly_df, weekly_partial = self._aggregate_weekly(
            normalized,
            include_partial_week=include_partial_week,
        )
        weekly = self._snapshot(
            weekly_df,
            timeframe="weekly",
            is_partial=weekly_partial,
        )
        confluence, summary = self._classify_confluence(daily, weekly)
        return MultiTimeframeBollResult(
            daily=daily,
            weekly=weekly,
            confluence=confluence,
            summary=summary,
        )

    @staticmethod
    def _normalize_daily(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "date" not in df or "close" not in df:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "amount"])
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        for column in ("open", "high", "low", "close", "volume", "amount"):
            if column not in out:
                out[column] = np.nan
            out[column] = pd.to_numeric(out[column], errors="coerce")
        out = (
            out.dropna(subset=["date", "close"])
            .sort_values("date")
            .drop_duplicates(subset=["date"], keep="last")
            .reset_index(drop=True)
        )
        return out

    def _aggregate_weekly(
        self,
        df: pd.DataFrame,
        *,
        include_partial_week: bool,
    ) -> tuple[pd.DataFrame, bool]:
        if df.empty:
            return df.copy(), False
        indexed = df.set_index("date")
        weekly = indexed.resample("W-FRI", label="right", closed="right").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "amount": "sum",
            }
        )
        weekly = weekly.dropna(subset=["close"]).reset_index()
        if weekly.empty:
            return weekly, False

        last_daily = pd.Timestamp(df["date"].iloc[-1]).normalize()
        final_week_end = pd.Timestamp(weekly["date"].iloc[-1]).normalize()
        is_partial = last_daily < final_week_end
        if is_partial and not include_partial_week:
            weekly = weekly.iloc[:-1].reset_index(drop=True)
            is_partial = False
        return weekly, is_partial

    def _snapshot(
        self,
        df: pd.DataFrame,
        *,
        timeframe: str,
        is_partial: bool,
    ) -> BollingerSnapshot:
        snapshot = BollingerSnapshot(
            timeframe=timeframe,
            sample_count=0 if df is None else len(df),
            is_partial=is_partial,
        )
        if df is None or df.empty:
            snapshot.missing_reason = "no_bars"
            return snapshot
        snapshot.as_of = pd.Timestamp(df["date"].iloc[-1]).date().isoformat()
        snapshot.current_price = float(df["close"].iloc[-1])
        if len(df) < self.PERIOD:
            snapshot.missing_reason = f"requires_{self.PERIOD}_{timeframe}_bars"
            return snapshot

        close = pd.to_numeric(df["close"], errors="coerce")
        middle = close.rolling(self.PERIOD).mean()
        std = close.rolling(self.PERIOD).std(ddof=0)
        upper = middle + self.MULTIPLIER * std
        lower = middle - self.MULTIPLIER * std
        width = upper - lower
        bandwidth = width / middle.replace(0, np.nan) * 100
        percent_b = (close - lower) / width.replace(0, np.nan)

        current = float(close.iloc[-1])
        mid = float(middle.iloc[-1])
        up = float(upper.iloc[-1])
        low = float(lower.iloc[-1])
        snapshot.available = all(np.isfinite(v) for v in (current, mid, up, low))
        if not snapshot.available:
            snapshot.missing_reason = "non_finite_band"
            return snapshot

        snapshot.middle = mid
        snapshot.upper = up
        snapshot.lower = low
        snapshot.bandwidth_pct = self._finite_float(bandwidth.iloc[-1])
        snapshot.percent_b = self._finite_float(percent_b.iloc[-1])
        if len(middle.dropna()) >= 4:
            prior_mid = float(middle.iloc[-4])
            if prior_mid:
                snapshot.middle_slope_pct = (mid - prior_mid) / prior_mid * 100

        snapshot.width_state = self._width_state(bandwidth)
        snapshot.position = self._position(current, low, mid, up)
        snapshot.event = self._event(close, lower, middle, upper)
        return snapshot

    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    @staticmethod
    def _position(price: float, lower: float, middle: float, upper: float) -> str:
        if price < lower:
            return "below_lower"
        if price > upper:
            return "above_upper"
        if price < middle:
            return "lower_half"
        return "upper_half"

    @staticmethod
    def _width_state(bandwidth: pd.Series) -> str:
        clean = bandwidth.dropna()
        if len(clean) < 6:
            return "stable"
        recent = float(clean.iloc[-1])
        prior = float(clean.iloc[-6])
        if prior <= 0:
            return "stable"
        change = (recent - prior) / prior
        if change >= 0.12:
            return "expanding"
        if change <= -0.12:
            return "contracting"
        return "stable"

    @staticmethod
    def _event(
        close: pd.Series,
        lower: pd.Series,
        middle: pd.Series,
        upper: pd.Series,
    ) -> str:
        if len(close) < 2:
            return "none"
        previous = float(close.iloc[-2])
        current = float(close.iloc[-1])
        pairs = (
            ("break_above_upper", upper),
            ("break_above_middle", middle),
            ("break_below_middle", middle),
            ("break_below_lower", lower),
        )
        for event, line in pairs:
            previous_line = line.iloc[-2]
            current_line = line.iloc[-1]
            if not np.isfinite(previous_line) or not np.isfinite(current_line):
                continue
            if event.startswith("break_above") and previous <= previous_line < current:
                return event
            if event.startswith("break_below") and previous >= previous_line > current:
                return event
        if previous < lower.iloc[-2] and current >= lower.iloc[-1]:
            return "reclaim_lower"
        if previous > upper.iloc[-2] and current <= upper.iloc[-1]:
            return "reject_upper"
        return "none"

    @staticmethod
    def _classify_confluence(
        daily: BollingerSnapshot,
        weekly: BollingerSnapshot,
    ) -> tuple[str, str]:
        if not daily.available or not weekly.available:
            return "unavailable", "日K或周K BOLL 数据不足"

        lower_positions = {"below_lower", "lower_half"}
        upper_positions = {"above_upper", "upper_half"}
        if daily.position == "below_lower" and weekly.position == "below_lower":
            return "double_oversold", "日周双周期跌破下轨，均值回归压力增强，但需等待日线企稳"
        if daily.position in lower_positions and weekly.position in upper_positions:
            return "daily_weak_weekly_strong", "日线回调但周线结构仍强，关注短线企稳"
        if daily.position in upper_positions and weekly.position in lower_positions:
            return "daily_strong_weekly_weak", "日线反弹但周线位置偏弱，谨防反弹受阻"
        if daily.position in upper_positions and weekly.position in upper_positions:
            return "double_bullish", "日周BOLL位置共振偏强"
        if daily.position in lower_positions and weekly.position in lower_positions:
            return "double_bearish", "日周BOLL位置共振偏弱"
        return "mixed", "日周BOLL信号分化，需结合趋势与支撑确认"
