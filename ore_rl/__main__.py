"""CLI smoke run: build the environment, run the deterministic baseline, and report.

Usage (from the repository root):
    python -m ore_rl [episodes]
"""
from __future__ import annotations

import sys

from ore_contracts import ContractCatalog, ModuleRegistry

from .environment import OREEnvironment, TaskSampler
from .modules import register_rl_example_modules
from .policies.base import DeterministicBaselinePolicy


def main(episodes: int = 8) -> int:
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    env = OREEnvironment(
        registry=registry,
        catalog=catalog,
        sampler=TaskSampler(seed=7, n_tasks=episodes),
        seed=7,
    )
    policy = DeterministicBaselinePolicy()
    for _ in range(episodes):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        while True:
            decision = policy.propose_actions(state, eligible)
            action = decision.chosen_action_id
            state, reward, terminated, truncated, info = env.step(action)
            eligible = info["feasible_action_ids"]
            if info.get("rejected"):
                state, reward, terminated, truncated, info = env.step(
                    "meta.stop_insufficient_evidence"
                )
                break
            if terminated or truncated:
                break
    summary = env.episode_summary()
    print("ORE-RL deterministic baseline smoke run")
    print(f"  episodes          : {summary['episodes']}")
    print(f"  success rate      : {summary['success_rate']:.3f}")
    print(f"  mean quality      : {summary['mean_quality']:.3f}")
    print(f"  mean evidence     : {summary['mean_evidence']:.3f}")
    print(f"  mean latency (ms) : {summary['mean_latency_ms']:.1f}")
    print(f"  mean module calls : {summary['mean_module_calls']:.1f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 8))
