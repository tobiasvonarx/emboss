"""Shared local application startup; each application owns its environment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_environment() -> None:
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    prefix = os.environ.get("NATIVE_PREFIX")
    if prefix:
        lib = str(Path(prefix).expanduser().resolve() / "lib")
        libraries = os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if lib not in libraries:
            os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(
                [lib, *filter(None, libraries)]
            )
            os.execv(sys.executable, [sys.executable, *sys.argv])


def launch(factory: str, default_port: int) -> None:
    configure_environment()
    import uvicorn

    uvicorn.run(
        factory,
        factory=True,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", str(default_port))),
    )


def data_directory() -> Path:
    return Path(os.getenv("DATA_DIR", "./data")).expanduser().resolve()


def worker_count() -> int:
    value = int(os.getenv("WORKERS", "2"))
    if value < 1:
        raise ValueError("WORKERS must be positive")
    return value
