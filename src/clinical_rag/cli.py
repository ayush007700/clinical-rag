"""CLI entrypoints."""

from __future__ import annotations

import argparse

import uvicorn

from clinical_rag.config import get_settings
from clinical_rag.ml import train_and_persist


def main() -> None:
    parser = argparse.ArgumentParser(prog="clinical-rag")
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run API server")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    sub.add_parser("train-risk", help="Train patient risk model")

    args = parser.parse_args()
    settings = get_settings()

    if args.cmd == "serve":
        uvicorn.run(
            "clinical_rag.api.app:app",
            host=args.host or settings.api_host,
            port=args.port or settings.api_port,
            reload=settings.app_env == "development",
        )
    elif args.cmd == "train-risk":
        print(train_and_persist())


if __name__ == "__main__":
    main()
