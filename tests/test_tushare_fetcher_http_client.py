# -*- coding: utf-8 -*-
"""Regression tests for TushareFetcher HTTP client initialization."""

import importlib.util
import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.litellm_stub import ensure_litellm_stub

ensure_litellm_stub()

try:
    json_repair_available = importlib.util.find_spec("json_repair") is not None
except ValueError:
    json_repair_available = "json_repair" in sys.modules

if not json_repair_available and "json_repair" not in sys.modules:
    sys.modules["json_repair"] = MagicMock()

if "fake_useragent" not in sys.modules:
    sys.modules["fake_useragent"] = MagicMock()

from data_provider.tushare_fetcher import TushareFetcher, _ThrottledTushareSdkClient, _TushareHttpClient


class TestTushareHttpClient(unittest.TestCase):
    """Ensure the lightweight HTTP client preserves Tushare Pro request semantics."""

    def test_query_posts_to_official_pro_endpoint(self) -> None:
        client = _TushareHttpClient(token="demo-token", timeout=15)
        response = MagicMock(
            status_code=200,
            text=json.dumps(
                {
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "close"],
                        "items": [["600519.SH", 1688.0]],
                    },
                }
            ),
        )

        with patch("data_provider.tushare_fetcher.requests.post", return_value=response) as post_mock:
            df = client.daily(ts_code="600519.SH", start_date="20260320", end_date="20260325")

        post_mock.assert_called_once_with(
            "http://api.tushare.pro",
            json={
                "api_name": "daily",
                "token": "demo-token",
                "params": {
                    "ts_code": "600519.SH",
                    "start_date": "20260320",
                    "end_date": "20260325",
                },
                "fields": "",
            },
            timeout=15,
        )
        self.assertEqual(df.to_dict(orient="records"), [{"ts_code": "600519.SH", "close": 1688.0}])

    def test_query_posts_to_custom_pro_endpoint(self) -> None:
        client = _TushareHttpClient(
            token="demo-token",
            timeout=15,
            api_url="https://tu.brze.top/",
        )
        response = MagicMock(
            status_code=200,
            text=json.dumps(
                {
                    "code": 0,
                    "data": {
                        "fields": ["ts_code", "close"],
                        "items": [["600519.SH", 1688.0]],
                    },
                }
            ),
        )

        with patch("data_provider.tushare_fetcher.requests.post", return_value=response) as post_mock:
            client.daily(ts_code="600519.SH", start_date="20260320", end_date="20260325")

        post_mock.assert_called_once()
        self.assertEqual(post_mock.call_args.args[0], "https://tu.brze.top")

    def test_query_respects_minimum_request_interval(self) -> None:
        client = _TushareHttpClient(
            token="demo-token",
            api_url="https://tu.brze.top",
            request_interval=0.6,
        )
        response = MagicMock(
            status_code=200,
            text=json.dumps({"code": 0, "data": {"fields": ["ts_code"], "items": [["600519.SH"]]}}),
        )

        with patch("data_provider.tushare_fetcher.requests.post", return_value=response), patch(
            "data_provider.tushare_fetcher.time.monotonic",
            side_effect=[100.0, 100.1, 100.6],
        ), patch("data_provider.tushare_fetcher.time.sleep") as sleep_mock:
            client.daily(ts_code="600519.SH")
            client.daily(ts_code="600519.SH")

        sleep_mock.assert_called_once()
        self.assertAlmostEqual(sleep_mock.call_args.args[0], 0.5)


class TestTushareFetcherInit(unittest.TestCase):
    """Ensure fetcher initialization no longer depends on the tushare SDK package."""

    def test_init_builds_http_client_when_token_present_without_proxy_url(self) -> None:
        config = SimpleNamespace(
            tushare_token="demo-token",
            tushare_api_url=None,
            tushare_request_interval=0.6,
        )

        with patch("data_provider.tushare_fetcher.get_config", return_value=config):
            fetcher = TushareFetcher()

        self.assertIsInstance(fetcher._api, _TushareHttpClient)
        self.assertEqual(fetcher._api._api_url, "http://api.tushare.pro")
        self.assertEqual(fetcher._api._request_interval, 0.6)
        self.assertTrue(fetcher.is_available())
        self.assertEqual(fetcher.priority, -1)

    def test_init_builds_official_sdk_client_when_proxy_url_present(self) -> None:
        config = SimpleNamespace(
            tushare_token="demo-token",
            tushare_api_url="https://tu.brze.top/",
            tushare_request_interval=0.6,
        )
        sdk_client = MagicMock()
        tushare_module = types.ModuleType("tushare")
        tushare_module.pro_api = MagicMock(return_value=sdk_client)
        pro_module = types.ModuleType("tushare.pro")
        client_module = types.ModuleType("tushare.pro.client")

        class DataApi:
            _DataApi__http_url = "http://api.tushare.pro"

        client_module.DataApi = DataApi
        pro_module.client = client_module

        with patch("data_provider.tushare_fetcher.get_config", return_value=config), patch.dict(
            sys.modules,
            {
                "tushare": tushare_module,
                "tushare.pro": pro_module,
                "tushare.pro.client": client_module,
            },
        ):
            fetcher = TushareFetcher()

        self.assertIsInstance(fetcher._api, _ThrottledTushareSdkClient)
        self.assertEqual(client_module.DataApi._DataApi__http_url, "https://tu.brze.top")
        tushare_module.pro_api.assert_called_once_with("demo-token")
        self.assertEqual(fetcher._api._client, sdk_client)
        self.assertEqual(fetcher._api._request_interval, 0.6)
        self.assertTrue(fetcher.is_available())
        self.assertEqual(fetcher.priority, -1)

    def test_sdk_client_wrapper_respects_minimum_request_interval(self) -> None:
        sdk_client = MagicMock()
        sdk_client.daily.return_value = "ok"
        client = _ThrottledTushareSdkClient(sdk_client, request_interval=0.6)

        with patch(
            "data_provider.tushare_fetcher.time.monotonic",
            side_effect=[100.0, 100.1, 100.6],
        ), patch("data_provider.tushare_fetcher.time.sleep") as sleep_mock:
            self.assertEqual(client.daily(ts_code="600519.SH"), "ok")
            self.assertEqual(client.daily(ts_code="600519.SH"), "ok")

        self.assertEqual(sdk_client.daily.call_count, 2)
        sleep_mock.assert_called_once()
        self.assertAlmostEqual(sleep_mock.call_args.args[0], 0.5)


if __name__ == "__main__":
    unittest.main()
