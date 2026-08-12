from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from strangeloop.kimi_cli import (KIMI_CODE_MANAGED_USAGE_URL,
                                  KimiCliManagedUsageAdapter, UsageHttpResponse,
                                  default_oauth_credential_reader,
                                  normalize_managed_usage,
                                  normalize_managed_usage_windows)
from strangeloop.quota import QuotaController, QuotaSource


NOW = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def managed_payload():
    return {
        "usage": {"limit": 100, "used": 20, "resetAt": "2026-08-19T12:00:00Z"},
        "limits": [
            {"window": {"duration": 5, "timeUnit": "HOUR"},
             "detail": {"limit": 10, "remaining": 1, "reset_at": "2026-08-12T14:00:00Z"}},
            {"window": {"duration": 1, "timeUnit": "MONTH"},
             "detail": {"limit": 1000, "remaining": 900, "resetAt": "2026-09-01T00:00:00Z"}},
        ],
    }


class KimiCliManagedUsageTests(unittest.TestCase):
    def test_normalizes_weekly_and_five_hour_to_tightest_window(self):
        snapshot = normalize_managed_usage(managed_payload(), NOW)
        self.assertEqual((10, 1, NOW), (snapshot.total, snapshot.remaining, snapshot.observed_at))
        self.assertEqual("provider_units", snapshot.primary_unit)
        self.assertEqual("rolling_5h", snapshot.primary_window_kind)
        self.assertEqual(datetime(2026, 8, 12, 14, 0, tzinfo=timezone.utc), snapshot.reset_at)
        windows = normalize_managed_usage_windows(managed_payload(), NOW)
        self.assertEqual(("rolling_5h", "weekly"), tuple(window.kind for window in windows))
        self.assertEqual((1, 10), (windows[0].remaining, windows[0].total))
        self.assertEqual((20, 100), (windows[1].total - windows[1].remaining,
                                    windows[1].total))

    def test_window_identity_allows_provider_switch_from_weekly_to_rolling(self):
        first_payload = {
            "usage": {"limit": 100, "used": 4, "resetAt": "2026-08-19T12:00:00Z"},
            "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                        "detail": {"limit": 100, "remaining": 97,
                                   "resetAt": "2026-08-12T17:00:00Z"}}],
        }
        second_payload = {
            "usage": {"limit": 100, "used": 4, "resetAt": "2026-08-19T12:00:00Z"},
            "limits": [{"window": {"duration": 5, "timeUnit": "HOUR"},
                        "detail": {"limit": 100, "remaining": 96,
                                   "resetAt": "2026-08-12T17:01:00Z"}}],
        }
        first = normalize_managed_usage(first_payload, NOW)
        later = NOW + timedelta(minutes=1)
        second = normalize_managed_usage(second_payload, later)
        self.assertEqual((96, "weekly"), (first.remaining, first.primary_window_kind))
        self.assertEqual((96, "rolling_5h"), (second.remaining, second.primary_window_kind))
        controller = QuotaController()
        self.assertTrue(controller.ingest_snapshot(first, QuotaSource.PROVIDER_USAGE, "system", now=NOW))
        self.assertTrue(controller.ingest_snapshot(second, QuotaSource.PROVIDER_USAGE, "system", now=later))

    def test_reads_only_oauth_access_token_from_injected_credential_and_uses_fixed_get(self):
        captured = []
        token = "oauth-token-that-must-not-escape"

        def transport(url, headers, timeout):
            captured.append((url, dict(headers), timeout))
            return UsageHttpResponse(200, managed_payload())

        adapter = KimiCliManagedUsageAdapter(
            credential_reader=lambda: {"access_token": token, "refresh_token": "never-used"},
            transport=transport, timeout_seconds=3)
        result = adapter.fetch(NOW)
        self.assertTrue(result.known)
        self.assertEqual(("rolling_5h", "weekly"), tuple(window.kind for window in result.windows))
        self.assertEqual(KIMI_CODE_MANAGED_USAGE_URL, captured[0][0])
        self.assertEqual("Bearer " + token, captured[0][1]["Authorization"])
        self.assertEqual("application/json", captured[0][1]["Accept"])
        self.assertEqual(3.0, captured[0][2])
        self.assertNotIn(token, repr(result))
        self.assertNotIn("refresh", repr(result).lower())

    def test_regular_api_key_cannot_be_used_for_usage(self):
        called = []
        adapter = KimiCliManagedUsageAdapter(
            credential_reader=lambda: {"api_key": "sk-not-oauth"},
            transport=lambda *args: called.append(args))
        result = adapter.fetch(NOW)
        self.assertEqual("oauth_access_token_missing", result.error_category)
        self.assertEqual([], called)

    def test_http_and_transport_errors_are_redacted_and_not_retried(self):
        calls = []
        adapter = KimiCliManagedUsageAdapter(
            credential_reader=lambda: {"access_token": "oauth-test"},
            transport=lambda *args: (calls.append(args) or UsageHttpResponse(401, {"detail": "secret body"})))
        result = adapter.fetch(NOW)
        self.assertEqual("oauth_unauthorized", result.error_category)
        self.assertEqual(1, len(calls))
        self.assertNotIn("secret", repr(result).lower())

    def test_unparseable_usage_stays_explicitly_unknown(self):
        adapter = KimiCliManagedUsageAdapter(
            credential_reader=lambda: {"access_token": "oauth-test"},
            transport=lambda *args: UsageHttpResponse(200, {"usage": {"limit": 10, "used": 2}}))
        result = adapter.fetch(NOW)
        self.assertEqual("unrecognized_usage_payload", result.error_category)

    def test_refresh_only_ingests_verified_snapshot(self):
        controller = QuotaController()
        adapter = KimiCliManagedUsageAdapter(
            credential_reader=lambda: {"access_token": "oauth-test"},
            transport=lambda *args: UsageHttpResponse(200, managed_payload()))
        result = adapter.refresh_controller(controller, NOW)
        self.assertTrue(result.known)
        telemetry = controller.export_telemetry(NOW)
        self.assertEqual("provider_usage", telemetry["source"])
        self.assertEqual(10, telemetry["snapshot"]["total"])

    def test_default_reader_accepts_cli_json_format_when_injected_home(self):
        # Exercise the file format using a temporary fake home, never the real CLI credential.
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / ".kimi-code" / "credentials"
            path.mkdir(parents=True)
            (path / "kimi-code.json").write_text(json.dumps({"access_token": "oauth-temp"}), encoding="utf-8")
            import strangeloop.kimi_cli as module
            previous = module.Path.home
            module.Path.home = classmethod(lambda cls: Path(temporary))
            try:
                self.assertEqual({"access_token": "oauth-temp"}, default_oauth_credential_reader())
            finally:
                module.Path.home = previous

    def test_usage_url_is_not_configurable_to_another_host_or_path(self):
        with self.assertRaises(ValueError):
            KimiCliManagedUsageAdapter(usage_url="https://example.com/coding/v1/usages")
        with self.assertRaises(ValueError):
            KimiCliManagedUsageAdapter(usage_url=KIMI_CODE_MANAGED_USAGE_URL + "?x=1")


if __name__ == "__main__":
    unittest.main()
