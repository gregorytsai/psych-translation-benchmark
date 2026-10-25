from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import run_live_judge as runner


class _FakeHTTPResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b'{"choices":[{"message":{"content":"{\\"ok\\":true}"}}]}'


class RunLiveJudgeOfflineTests(unittest.TestCase):
    def test_extra_body_merges_effective_thinking_kwargs_and_preserves_track_budget(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeHTTPResponse()

        client = runner.Client(
            "http://127.0.0.1:9/v1",
            "qwen-test",
            12,
            None,
            chat_template_kwargs={"enable_thinking": False, "preserve_thinking": True},
            extra_body={
                "reasoning_effort": "xhigh",
                "reasoning_format": "deepseek",
                "reasoning_budget_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
            },
        )
        with mock.patch.object(runner.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = client.call([{"role": "user", "content": "offline"}], 20261022, "request-1", max_tokens=8192)

        self.assertTrue(result["ok"])
        payload = json.loads(captured["request"].data.decode("utf-8"))
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": True, "preserve_thinking": False})
        self.assertEqual(payload["reasoning_effort"], "xhigh")
        self.assertEqual(payload["reasoning_format"], "deepseek")
        self.assertEqual(payload["reasoning_budget_tokens"], 4096)
        self.assertEqual(payload["max_tokens"], 8192)

    def test_extra_body_cannot_override_protected_request_keys(self):
        for key in ("model", "messages", "stream"):
            with self.subTest(key=key):
                client = runner.Client("http://127.0.0.1:9/v1", "model", 1, None, extra_body={key: "override"})
                with self.assertRaisesRegex(ValueError, "protected request key"):
                    client.call([], 1, "request", max_tokens=7)

    def test_strict_json_rejects_fences_and_footer_but_legacy_parser_remains_permissive(self):
        clean = '{"ok":true}'
        self.assertEqual(runner.extract_json_strict(clean), {"ok": True})
        with self.assertRaises(ValueError):
            runner.extract_json_strict("```json\n" + clean + "\n```")
        with self.assertRaises(ValueError):
            runner.extract_json_strict(clean + "\nfooter")
        self.assertEqual(runner.extract_json("```json\n" + clean + "\n```"), {"ok": True})
        self.assertEqual(runner.extract_json(clean + "\nfooter"), {"ok": True})

    def test_redaction_keeps_nonsecret_token_accounting_and_hides_auth_secret(self):
        value = {
            "request": {
                "max_tokens": 8192,
                "auth_env": "JUDGE_API_TOKEN",
                "Authorization": "Bearer super-secret-value",
                "api_key": "sk-test-secret-value",
                "usage": {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168},
            },
            "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
            "error": "password=hidden token: hidden-too",
        }
        clean = runner.redact(value)
        self.assertEqual(clean["request"]["max_tokens"], 8192)
        self.assertEqual(clean["request"]["auth_env"], "JUDGE_API_TOKEN")
        self.assertEqual(clean["request"]["usage"], {"prompt_tokens": 123, "completion_tokens": 45, "total_tokens": 168})
        self.assertEqual(clean["response"]["usage"]["total_tokens"], 15)
        self.assertEqual(clean["request"]["Authorization"], "<REDACTED>")
        self.assertEqual(clean["request"]["api_key"], "<REDACTED>")
        self.assertNotIn("hidden", clean["error"])

    def test_auth_env_secret_is_used_for_header_but_not_persisted_in_event(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["authorization"] = request.get_header("Authorization")
            return _FakeHTTPResponse()

        secret = "offline-auth-secret-20261022"
        client = runner.Client("http://127.0.0.1:9/v1", "model", 1, "OFFLINE_AUTH_ENV")
        with mock.patch.dict(os.environ, {"OFFLINE_AUTH_ENV": secret}, clear=False), mock.patch.object(runner.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = client.call([], 1, "auth-request", max_tokens=12)

        self.assertEqual(captured["authorization"], f"Bearer {secret}")
        self.assertTrue(result["event"]["auth_present"])
        self.assertNotIn(secret, json.dumps(result["event"], ensure_ascii=False))

    def test_main_applies_each_track_max_tokens_without_network_and_records_strict_mode(self):
        packet = {
            "packet_id": "offline-runner-test",
            "qualification": {
                "items": [{
                    "qualification_id": "q1",
                    "source_en": "source",
                    "variants": [
                        {"blind_id": "cand-a", "target_zh_tw": "甲"},
                        {"blind_id": "cand-b", "target_zh_tw": "乙"},
                    ],
                }],
            },
            "pointwise": {
                "items": [{
                    "item_id": "p1",
                    "source_en": "source",
                    "candidates": [{"blind_id": "cand-a", "target_zh_tw": "甲"}],
                }],
            },
            "pairwise": {
                "tasks": [{
                    "pair_id": "pair-1",
                    "orientation": "AB",
                    "item_id": "p1",
                    "source_en": "source",
                    "candidate_a": {"blind_id": "cand-a", "target_zh_tw": "甲"},
                    "candidate_b": {"blind_id": "cand-b", "target_zh_tw": "乙"},
                }],
            },
        }
        calls = []

        def fake_call(self, messages, seed, request_id, max_tokens=1400):
            calls.append((request_id, max_tokens))
            if request_id.startswith("capability") or request_id.startswith("request-format"):
                content = '{"ok":true}'
            elif request_id.startswith("qualification"):
                content = '{"judgments":[]}'
            elif request_id.startswith("pointwise"):
                content = json.dumps({
                    "blind_id": "cand-a",
                    "score_0_100": 100,
                    "subscores": {"terminology_locale": 30, "semantic_safety": 30, "fluency": 20, "completeness_format": 20},
                    "critical": False,
                    "errors": [],
                })
            else:
                content = '{"winner":"A","confidence":1}'
            response = {"choices": [{"message": {"content": content}}]}
            return {"ok": True, "response": response, "event": {"request_id": request_id, "status": "ok"}, "elapsed_s": 0}

        with tempfile.TemporaryDirectory(prefix="runner-offline-test-") as directory:
            root = Path(directory)
            packet_path = root / "packet.json"
            instructions_path = root / "instructions.txt"
            output_path = root / "bundle.json"
            packet_path.write_text(json.dumps(packet), encoding="utf-8")
            instructions_path.write_text("Return JSON only.", encoding="utf-8")
            argv = [
                "run_live_judge.py",
                "--packet", str(packet_path),
                "--instructions", str(instructions_path),
                "--base-url", "http://127.0.0.1:9/v1",
                "--model", "offline-model",
                "--judge-id", "offline-judge",
                "--family", "test",
                "--provider", "offline",
                "--output", str(output_path),
                "--strict-json-output",
                "--max-tokens-capability", "11",
                "--max-tokens-qualification", "22",
                "--max-tokens-pointwise", "33",
                "--max-tokens-pairwise", "44",
            ]
            with mock.patch.object(runner.Client, "call", new=fake_call), mock.patch.object(sys, "argv", argv):
                self.assertEqual(runner.main(), 0)
            bundle = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual([tokens for _, tokens in calls], [11, 11, 22, 33, 44])
        self.assertEqual(bundle["settings"]["max_tokens_by_track"], {"capability": 11, "qualification": 22, "pointwise": 33, "pairwise": 44})
        self.assertTrue(bundle["settings"]["strict_json_output"])
        self.assertEqual(bundle["counts"]["failed"], 0)


if __name__ == "__main__":
    unittest.main()
