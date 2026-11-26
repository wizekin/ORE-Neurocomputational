"""Command-line summary of the Point 17 distributed coordination layer."""
from __future__ import annotations

import json

from ore_distributed import build_pipeline


def main() -> None:
    """Print the deployment classification and registry summary."""
    pipeline = build_pipeline(node_id="cli-node", cluster_id="ore-cli", process_count=1)
    payload = {
        "deployment": pipeline.describe(),
        "registry": pipeline.registry_summary(),
        "interface_checks": pipeline.interface_checks(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
