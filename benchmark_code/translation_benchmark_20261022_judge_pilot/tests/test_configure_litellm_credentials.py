from __future__ import annotations

import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import configure_litellm_credentials as helper


class ConfigureLiteLLMCredentialsTests(unittest.TestCase):
    def test_store_status_and_mode_without_printing_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace.secrets" / "litellm_judge.key"
            token = "sk-test-local-only-20261022"
            output = io.StringIO()
            with mock.patch.object(helper.getpass, "getpass", return_value=token), contextlib.redirect_stdout(output):
                self.assertEqual(helper.main(["--secret-path", str(path)]), 0)
            self.assertNotIn(token, output.getvalue())
            self.assertEqual(path.read_text(encoding="utf-8"), token)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(helper.main(["--secret-path", str(path), "--status"]), 0)
            status = json.loads(output.getvalue())
            self.assertEqual(status["status"], "ready")
            self.assertNotIn(token, output.getvalue())

    def test_refuses_existing_file_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret.key"
            path.write_text("old-token", encoding="utf-8")
            os.chmod(path, 0o600)
            with mock.patch.object(helper.getpass, "getpass") as prompt:
                self.assertEqual(helper.main(["--secret-path", str(path)]), 2)
                prompt.assert_not_called()
            self.assertEqual(path.read_text(encoding="utf-8"), "old-token")

    def test_exec_env_passes_secret_without_printing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret.key"
            path.write_text("sk-child-only", encoding="utf-8")
            os.chmod(path, 0o600)
            child = [
                sys.executable,
                "-c",
                "import os,sys; sys.exit(0 if os.environ.get('TEST_LITELLM_TOKEN') == 'sk-child-only' else 1)",
            ]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(helper.main(["--secret-path", str(path), "--exec-env", "TEST_LITELLM_TOKEN", "--", *child]), 0)
            self.assertEqual(output.getvalue(), "")
            self.assertNotIn("sk-child-only", output.getvalue())

    def test_rejects_multiline_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secret.key"
            with mock.patch.object(helper.getpass, "getpass", return_value="line1\nline2"):
                self.assertEqual(helper.main(["--secret-path", str(path)]), 2)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
