"""Command-line summary of the Point 18 data infrastructure layer."""
from __future__ import annotations

import json

from ore_data import build_pipeline


def main() -> None:
    """Print the deployment capabilities and registry summary."""
    pipeline = build_pipeline()
    payload = {
        "pipeline": pipeline.describe(),
        "registry": pipeline.registry_summary(),
        "interface_checks": pipeline.interface_checks(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
