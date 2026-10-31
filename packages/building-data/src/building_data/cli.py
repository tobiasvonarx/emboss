"""Local source installation, shared by Emboss and its consumers."""

import argparse
from pathlib import Path

from .runtime import configure_environment


def main():
    configure_environment()
    from .cambridge import install, source_root

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cambridge = commands.add_parser(
        "import-cambridge", help="Install the supplied Cambridge data once"
    )
    cambridge.add_argument("--imagery", type=Path, required=True)
    cambridge.add_argument("--topography", type=Path, required=True)
    cambridge.add_argument("--lidar", type=Path, required=True)
    cambridge.add_argument("--output", type=Path, default=source_root())
    args = parser.parse_args()
    install(args.imagery, args.topography, args.lidar, args.output)
