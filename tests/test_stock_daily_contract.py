from datetime import date

from src.storage import StockDaily


def test_stock_daily_to_dict_remains_available_after_intraday_models():
    row = StockDaily(code="002050", date=date(2026, 6, 20), close=25.5)

    payload = row.to_dict()

    assert payload["code"] == "002050"
    assert payload["date"] == date(2026, 6, 20)
    assert payload["close"] == 25.5
