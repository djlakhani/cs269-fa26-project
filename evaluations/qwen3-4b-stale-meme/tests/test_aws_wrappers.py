from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class AwsWrapperTests(unittest.TestCase):
    def _run(self, script: str, arguments: list[str]) -> list[str]:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            log = root / "uv.log"
            uv = fake_bin / "uv"
            uv.write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> \"$FAKE_UV_LOG\"\n",
                encoding="utf-8",
            )
            aws = fake_bin / "aws"
            aws.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
            uv.chmod(0o755)
            aws.chmod(0o755)
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "FAKE_UV_LOG": str(log),
                "QWEN4B_ARTIFACT_BASE": str(root / "artifacts"),
                "QWEN4B_CACHE_DIR": str(root / "cache"),
                "QWEN4B_S3_BASE_URI": "s3://test-bucket/test-prefix",
                "QWEN4B_SYNC_INTERVAL_SECONDS": "3600",
            }
            subprocess.run(
                ["bash", str(PACKAGE_ROOT / "scripts" / script), *arguments],
                cwd=PACKAGE_ROOT,
                env=environment,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return log.read_text(encoding="utf-8").splitlines()

    def test_stale_wrapper_constructs_all_four_view_commands(self) -> None:
        calls = self._run(
            "run_aws_stale.sh",
            ["base-run", "key-run", "smoke", "description"],
        )
        joined = "\n".join(calls)
        self.assertIn("qwen3-memory-eval prepare ", joined)
        self.assertIn("qwen3-memory-eval prepare-key-pairs ", joined)
        self.assertEqual(joined.count("qwen3-memory-eval run "), 2)
        self.assertEqual(joined.count("--max-scenarios 1"), 2)
        self.assertNotIn(" + ", joined)

    def test_meme_wrapper_constructs_both_view_commands(self) -> None:
        calls = self._run(
            "run_aws_meme.sh",
            ["meme-run", "smoke", "description"],
        )
        joined = "\n".join(calls)
        self.assertIn("qwen3-memory-eval meme-prepare ", joined)
        self.assertIn("qwen3-memory-eval meme-run ", joined)
        self.assertIn("--max-episodes 1", joined)
        self.assertNotIn(" + ", joined)


if __name__ == "__main__":
    unittest.main()
