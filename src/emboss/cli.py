"""CLI and local browser application entry point."""

from __future__ import annotations

import argparse
import json


def main():
    from building_data.runtime import configure_environment

    configure_environment()
    parser = argparse.ArgumentParser(description="Emboss roof reconstruction")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="Open the local application (default)")
    acquire = sub.add_parser("acquire", help="Prepare a house or bounding box")
    group = acquire.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--location", nargs=2, type=float, metavar=("LONGITUDE", "LATITUDE")
    )
    group.add_argument(
        "--bbox", nargs=4, type=float, metavar=("WEST", "SOUTH", "EAST", "NORTH")
    )
    run = sub.add_parser("reconstruct", help="Reconstruct an acquired house")
    run.add_argument("house_id")
    run.add_argument("--force", action="store_true")
    sub.add_parser("houses", help="List prepared houses")
    args = parser.parse_args()
    if args.command in (None, "serve"):
        from building_data.runtime import launch

        launch("emboss.web:create_app", 5001)
        return
    import os

    from building_data.runtime import data_directory, worker_count

    from .api import Client

    client = Client(
        data_directory(), workers=worker_count(), device=os.getenv("DEVICE", "auto")
    )
    if args.command == "houses":
        output = client.store.list_houses()
    elif args.command == "acquire":
        selection = (
            {
                "mode": "house",
                "longitude": args.location[0],
                "latitude": args.location[1],
            }
            if args.location
            else {"mode": "area", "bbox": args.bbox}
        )
        output = client.store.acquire(selection, print)
    else:
        output = client.reconstruct(
            args.house_id, force=args.force, progress=print
        ).as_dict()
    print(json.dumps(output, indent=2))
