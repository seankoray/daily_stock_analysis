# -*- coding: utf-8 -*-
"""
===================================
Report Engine - Jinja2 Report Renderer
===================================

Renders reports from Jinja2 templates. Falls back to caller's logic on template
missing or render error. Template path is relative to project root.
Any expensive data preparation should be injected by the caller via extra_context.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analyzer import AnalysisResult
from src.config import get_config
from src.report_language import (
    get_localized_stock_name,
    get_report_labels,
    get_signal_level,
    get_chip_unavailable_reason,
    is_chip_structure_unavailable,
    localize_chip_health,
    localize_operation_advice,
    localize_trend_prediction,
    normalize_report_language,
)
from src.utils.data_processing import normalize_model_used

logger = logging.getLogger(__name__)


def _escape_md(text: str) -> str:
    """Escape markdown special chars (*ST etc)."""
    if not text:
        return ""
    return text.replace("*", "\\*").replace("_", "\\_")


def _clean_sniper_value(val: Any) -> str:
    """Format sniper point value for display (strip label prefixes)."""
    if val is None:
        return "N/A"
    if isinstance(val, (int, float)):
        return str(val)
    s = str(val).strip() if val else ""
    if not s or s == "N/A":
        return s or "N/A"
    prefixes = [
        "理想买入点：", "次优买入点：", "止损位：", "目标位：",
        "理想买入点:", "次优买入点:", "止损位:", "目标位:",
        "Ideal Entry:", "Secondary Entry:", "Stop Loss:", "Target:",
    ]
    for prefix in prefixes:
        if s.startswith(prefix):
            return s[len(prefix):]
    return s


def _format_number(value: Any, digits: int = 2) -> str:
    """Format report numbers without exposing floating-point implementation noise."""
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _format_percent_b(value: Any) -> str:
    """Render BOLL %B as an intuitive percentage rather than a 0-1 decimal."""
    if value is None or value == "":
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return str(value)


def _localize_boll_position(value: Any) -> str:
    return {
        "above_upper": "上轨上方（偏热）",
        "upper_half": "中轨上方",
        "lower_half": "中轨下方",
        "below_lower": "下轨下方（偏弱）",
        "unavailable": "数据不足",
    }.get(str(value or "").strip(), str(value or "N/A"))


def _localize_boll_width(value: Any) -> str:
    return {
        "expanding": "带宽扩大",
        "contracting": "带宽收窄",
        "stable": "带宽平稳",
        "unavailable": "数据不足",
    }.get(str(value or "").strip(), str(value or "N/A"))


def _score_level(value: Any) -> str:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "数据不足"
    if score >= 80:
        return "较强"
    if score >= 70:
        return "偏积极"
    if score >= 60:
        return "中性偏积极"
    if score >= 50:
        return "一般"
    if score >= 35:
        return "偏弱"
    return "较弱"


def _boll_score_interpretation(boll_data: Dict[str, Any]) -> str:
    """Explain the difference between stock quality and current entry timing."""
    try:
        medium = float(boll_data.get("medium_term_score"))
        entry = float(boll_data.get("entry_timing_score"))
    except (TypeError, ValueError):
        return "评分数据不足，暂不能判断中期结构与当前买点的匹配程度。"

    if medium >= 70 and entry < 60:
        return "中期结构较好，但当前买点一般；“值得关注”不等于“现在适合追买”，宜等待回踩或突破确认。"
    if medium >= 70 and entry >= 70:
        return "中期结构与当前时机同时偏强，但仍需结合量能、支撑和风险收益空间确认。"
    if medium < 50 and entry >= 65:
        return "短线位置有所改善，但中期结构仍弱，更接近反弹观察而非趋势性机会。"
    if medium < 50 and entry < 50:
        return "中期结构和当前时机均偏弱，优先观察企稳，不宜仅因接近下轨而机械抄底。"
    return "中期结构与当前时机没有形成强共振，建议等待趋势或价格位置进一步确认。"


def _boll_watch_condition(boll_data: Dict[str, Any]) -> str:
    daily = boll_data.get("daily") or {}
    weekly = boll_data.get("weekly") or {}
    daily_position = daily.get("position")
    weekly_position = weekly.get("position")
    if daily_position in {"below_lower", "lower_half"}:
        return "关注日K重新站上中轨或在确认支撑附近止跌；若跌破下轨后继续放量走弱，信号失效。"
    if daily_position in {"above_upper", "upper_half"} and weekly_position in {"above_upper", "upper_half"}:
        return "关注回踩日K中轨能否获得支撑，或放量突破近期压力；当前位置不宜仅凭趋势偏强追高。"
    if weekly_position in {"below_lower", "lower_half"}:
        return "短线反弹需先确认日K企稳，同时观察周K能否收复中轨，否则仍按弱势反弹处理。"
    return "等待日K方向确认，并与均线、近期高低点、量能和风险收益空间交叉验证。"


def _resolve_templates_dir() -> Path:
    """Resolve template directory relative to project root."""
    config = get_config()
    base = Path(__file__).resolve().parent.parent.parent
    templates_dir = Path(config.report_templates_dir)
    if not templates_dir.is_absolute():
        return base / templates_dir
    return templates_dir


def render(
    platform: str,
    results: List[AnalysisResult],
    report_date: Optional[str] = None,
    summary_only: bool = False,
    extra_context: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Render report using Jinja2 template.

    Args:
        platform: One of: markdown, wechat, brief
        results: List of AnalysisResult
        report_date: Report date string (default: today)
        summary_only: Whether to output summary only
        extra_context: Additional template context

    Returns:
        Rendered string, or None on error (caller should fallback).
    """
    from datetime import datetime

    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        logger.warning("jinja2 not installed, report renderer disabled")
        return None

    if report_date is None:
        report_date = datetime.now().strftime("%Y-%m-%d")

    templates_dir = _resolve_templates_dir()
    template_name = f"report_{platform}.j2"
    template_path = templates_dir / template_name
    if not template_path.exists():
        logger.debug("Report template not found: %s", template_path)
        return None

    report_language = normalize_report_language(
        (extra_context or {}).get("report_language")
        or next(
            (getattr(result, "report_language", None) for result in results if getattr(result, "report_language", None)),
            None,
        )
        or getattr(get_config(), "report_language", "zh")
    )
    labels = get_report_labels(report_language)

    # Build template context with pre-computed signal levels (sorted by score)
    sorted_results = sorted(results, key=lambda x: x.sentiment_score, reverse=True)
    sorted_enriched = []
    for r in sorted_results:
        st, se, _ = get_signal_level(r.operation_advice, r.sentiment_score, report_language)
        rn = get_localized_stock_name(r.name, r.code, report_language)
        sorted_enriched.append({
            "result": r,
            "signal_text": st,
            "signal_emoji": se,
            "stock_name": _escape_md(rn),
            "localized_operation_advice": localize_operation_advice(r.operation_advice, report_language),
            "localized_trend_prediction": localize_trend_prediction(r.trend_prediction, report_language),
        })

    buy_count = sum(1 for r in results if getattr(r, "decision_type", "") == "buy")
    sell_count = sum(1 for r in results if getattr(r, "decision_type", "") == "sell")
    hold_count = sum(1 for r in results if getattr(r, "decision_type", "") in ("hold", ""))
    show_llm_model = bool(getattr(get_config(), "report_show_llm_model", True))
    models_used: List[str] = []
    if show_llm_model:
        for result in results:
            model = normalize_model_used(getattr(result, "model_used", None))
            if model:
                models_used.append(model)
        models_used = list(dict.fromkeys(models_used))

    report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def failed_checks(checklist: List[str]) -> List[str]:
        return [c for c in (checklist or []) if c.startswith("❌") or c.startswith("⚠️")]

    context: Dict[str, Any] = {
        "report_date": report_date,
        "report_timestamp": report_timestamp,
        "results": sorted_results,
        "enriched": sorted_enriched,  # Sorted by sentiment_score desc
        "summary_only": summary_only,
        "buy_count": buy_count,
        "sell_count": sell_count,
        "hold_count": hold_count,
        "labels": labels,
        "report_language": report_language,
        "models_used": models_used,
        "show_llm_model": show_llm_model,
        "escape_md": _escape_md,
        "clean_sniper": _clean_sniper_value,
        "format_number": _format_number,
        "format_percent_b": _format_percent_b,
        "localize_boll_position": _localize_boll_position,
        "localize_boll_width": _localize_boll_width,
        "score_level": _score_level,
        "boll_score_interpretation": _boll_score_interpretation,
        "boll_watch_condition": _boll_watch_condition,
        "failed_checks": failed_checks,
        "history_by_code": {},
        "get_chip_unavailable_reason": get_chip_unavailable_reason,
        "is_chip_structure_unavailable": is_chip_structure_unavailable,
        "localize_operation_advice": localize_operation_advice,
        "localize_trend_prediction": localize_trend_prediction,
        "localize_chip_health": localize_chip_health,
    }
    if extra_context:
        safe_extra_context = dict(extra_context)
        safe_extra_context.pop("labels", None)
        safe_extra_context.pop("report_language", None)
        context.update(safe_extra_context)

    try:
        env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(default=False),
        )
        template = env.get_template(template_name)
        return template.render(**context)
    except Exception as e:
        logger.warning("Report render failed for %s: %s", template_name, e)
        return None
