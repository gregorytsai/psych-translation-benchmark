from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import litellm_safe_discovery as discovery


class SafeDiscoveryTests(unittest.TestCase):
    def test_allowlist_drops_raw_secret_fields_and_keeps_capabilities(self) -> None:
        model_payload = {
            "data": [
                {
                    "id": "claude-test",
                    "object": "model",
                    "owned_by": "anthropic",
                    "model_info": {
                        "provider": "anthropic",
                        "family": "claude",
                        "supports_tools": True,
                        "context_window": 128000,
                        "api_key": "raw-secret-must-not-persist",
                    },
                    "litellm_params": {"api_key": "raw-secret-must-not-persist"},
                }
            ]
        }

        def fake_request(url: str, token: str, timeout: float) -> tuple[int, str, object]:
            self.assertEqual(token, "sk-test-only")
            self.assertEqual(timeout, 2)
            if url.endswith("/health"):
                return 200, "application/json", {"status": "healthy", "secret": "must-not-persist"}
            return 200, "application/json", model_payload

        with mock.patch.dict(os.environ, {"TEST_LITELLM_TOKEN": "sk-test-only"}, clear=False), mock.patch.object(discovery, "request_json", side_effect=fake_request):
            result = discovery.discover("http://127.0.0.1:4000", "TEST_LITELLM_TOKEN", 2)
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn("raw-secret-must-not-persist", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertEqual(result["candidate_count"], 1)
        row = result["endpoints"]["models"]["models"][0]
        self.assertEqual(row["id"], "claude-test")
        self.assertEqual(row["model_info_capability"]["context_window"], 128000)
        self.assertTrue(row["model_info_capability"]["supports_tools"])
        self.assertEqual(row["lineage_hint"]["hint"], "claude")

    def test_base_url_rejects_userinfo_and_query(self) -> None:
        with self.assertRaises(ValueError):
            discovery.safe_base_url("http://user:pass@example.invalid:4000")
        with self.assertRaises(ValueError):
            discovery.safe_base_url("http://example.invalid:4000/?token=secret")


if __name__ == "__main__":
    unittest.main()
