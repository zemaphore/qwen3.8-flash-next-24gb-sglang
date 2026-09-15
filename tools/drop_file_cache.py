#!/usr/bin/env python3
"""Advise the kernel to discard clean cached pages for one explicit file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.resolve(strict=True)
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    print(f"POSIX_FADV_DONTNEED: {path} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
