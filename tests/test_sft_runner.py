"""Exercise paid-run mode selection and argument quoting without calling an API."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project with spaces"
        (self.root / "scripts").mkdir(parents=True)
        original = Path(__file__).resolve().parents[1] / "scripts/run_sft_teacher.sh"
        self.runner = self.root / "scripts/run_sft_teacher.sh"
        shutil.copyfile(original, self.runner)
        self.capture = self.root / "arguments.json"
        self.python = self.root / "mock python"
        self.python.write_text(
            "#!" + sys.executable + "\n"
            "import json,os,sys\n"
            "from pathlib import Path\n"
            "Path(os.environ['TEST_SFT_ARGS_FILE']).write_text(json.dumps(sys.argv[1:]))\n")
        self.python.chmod(0o755)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(("SFT_", "DEEPSEEK_", "STUDENT_", "SEARCH_R1_"))}
        self.env.update({"SFT_PYTHON": str(self.python), "TEST_SFT_ARGS_FILE": str(self.capture),
                         "DEEPSEEK_API_KEY": "test-api-key",
                         "STUDENT_TOKENIZER": "/models/tokenizer with spaces"})

    def tearDown(self):
        self.temp.cleanup()

    def run_mode(self, *arguments, status=0):
        if self.capture.exists():
            self.capture.unlink()
        result = subprocess.run(["bash", str(self.runner), *arguments], env=self.env,
                                cwd=self.temp.name, capture_output=True, text=True)
        self.assertEqual(result.returncode, status, result.stdout + result.stderr)
        self.assertNotIn(self.env["DEEPSEEK_API_KEY"], result.stdout + result.stderr)
        args = json.loads(self.capture.read_text()) if self.capture.exists() else None
        return args

    def test_check_and_default_pilot_use_safe_modes_and_quote_paths(self):
        check = self.run_mode("check")
        self.assertIn("--preflight-only", check)
        self.assertEqual(check[check.index("--generation-mode") + 1], "query-first")
        self.assertIn("--quality-review", check)
        self.assertEqual(check[check.index("--student-tokenizer") + 1], "/models/tokenizer with spaces")
        pilot = self.run_mode()
        self.assertEqual(pilot[pilot.index("--limit") + 1], "20")
        self.assertNotIn("--preflight-only", pilot)
        self.assertEqual(pilot[pilot.index("--output-dir") + 1], check[check.index("--output-dir") + 1])

    def test_full_resumes_same_directory_and_export_never_calls_generate(self):
        pilot = self.run_mode("pilot")
        full = self.run_mode("full")
        self.assertEqual(full[full.index("--limit") + 1], "0")
        self.assertEqual(full[full.index("--output-dir") + 1], pilot[pilot.index("--output-dir") + 1])
        preview = self.run_mode("preview")
        self.assertEqual(preview[1], "export")
        self.assertIn("--allow-partial", preview)
        export = self.run_mode("export")
        self.assertEqual(export[1], "export")
        self.assertNotIn("--allow-partial", export)
        self.assertNotEqual(export[export.index("--output-dir") + 1], preview[preview.index("--output-dir") + 1])

    def test_pilot_limit_override_is_bounded_and_zero_is_rejected(self):
        self.env["SFT_PILOT_LIMIT"] = "2"
        arguments = self.run_mode("pilot")
        self.assertEqual(arguments[arguments.index("--limit") + 1], "2")
        for invalid in ("0", "", "-1", "unlimited"):
            with self.subTest(value=invalid):
                self.env["SFT_PILOT_LIMIT"] = invalid
                self.assertIsNone(self.run_mode("pilot", status=2))

    def test_nonempty_export_is_protected_before_python_runs(self):
        target = self.root / "existing export"
        target.mkdir()
        (target / "keep.txt").write_text("existing output")
        self.env["SFT_EXPORT_DIR"] = str(target)
        self.assertIsNone(self.run_mode("export", status=1))
        self.assertEqual((target / "keep.txt").read_text(), "existing output")

    def test_help_and_invalid_mode_never_invoke_python(self):
        self.assertIsNone(self.run_mode("--help"))
        self.assertIsNone(self.run_mode("typo", status=2))
        self.assertIsNone(self.run_mode("full", "unexpected", status=2))


if __name__ == "__main__":
    unittest.main()
