"""Launcher contract tests for the pretrain auto-DDP entrypoint."""

from __future__ import annotations

from pathlib import Path

import unittest

from ..scripts.run_pretrain import _build_torchrun_cmd


class RunPretrainLauncherTest(unittest.TestCase):
    """Validate the torchrun command built for local DDP auto-spawn."""

    def test_auto_spawn_uses_standalone_when_port_is_omitted(self) -> None:
        """Single-node DDP should default to a free rendezvous port."""
        cmd = _build_torchrun_cmd(
            script_path=Path("/tmp/train.py"),
            nproc_per_node=8,
            master_port=None,
        )

        self.assertIn("--standalone", cmd)
        self.assertNotIn("--master_port", cmd)
        self.assertEqual(cmd[-1], "/tmp/train.py")

    def test_explicit_master_port_overrides_standalone_mode(self) -> None:
        """User-provided master_port should be forwarded unchanged."""
        cmd = _build_torchrun_cmd(
            script_path=Path("/tmp/train.py"),
            nproc_per_node=4,
            master_port=29517,
        )

        self.assertNotIn("--standalone", cmd)
        self.assertIn("--master_port", cmd)
        port_index = cmd.index("--master_port")
        self.assertEqual(cmd[port_index + 1], "29517")
        self.assertEqual(cmd[-1], "/tmp/train.py")


if __name__ == "__main__":
    unittest.main()
