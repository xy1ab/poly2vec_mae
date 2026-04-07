"""Runtime logging helpers for AE pretraining scripts.

This module provides a simple tee-style logger that mirrors `stdout` into a
persistent log file under the current experiment directory.
"""

from __future__ import annotations

import sys
from pathlib import Path


class TeeStdout:
    """Mirror terminal output into a text log file.

    This class is intended to replace `sys.stdout` during long training runs
    where users need both live console logs and saved logs.
    """

    def __init__(self, filename: str | Path):
        """Initialize a tee logger.

        Args:
            filename: Target log file path.
        """
        self.terminal = sys.stdout
        log_path = Path(filename)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = log_path.open("a", encoding="utf-8")

    def write(self, message: str) -> None:
        """Write text to both terminal and log file.

        Args:
            message: Output message.
        """
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        """Flush both terminal and log streams."""
        self.terminal.flush()
        self.log.flush()


def attach_tee_stdout(log_path: str | Path) -> TeeStdout:
    """Attach a tee logger to global stdout.

    Args:
        log_path: Log file path.

    Returns:
        The tee logger instance.
    """
    tee = TeeStdout(log_path)
    sys.stdout = tee
    return tee
