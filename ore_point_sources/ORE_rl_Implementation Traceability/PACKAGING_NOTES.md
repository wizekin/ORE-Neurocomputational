# Point 10 deliverable — ore_rl reinforcement learning layer

Contents of this archive (paths are relative to the repository root of
`ORE-Neurocomputational`, so the archive can be unzipped over a checkout):

| Path | What it is |
| --- | --- |
| `ore_rl/` | The complete `ore_rl` package (22 modules): contracts, state encoder, action catalog, safety gates, reward, trajectory store, Gymnasium-shaped environment, example modules, bandits, DQN, PPO, options, offline RL (BC/CQL/IQL), constrained optimization, world model, evaluation, policy registry, controller adapter (shadow mode, preference bridge) |
| `tests/test_rl_*.py` | 10 acceptance test files, 94 tests, all passing |
| `docs/ORE_RL_IMPLEMENTATION.md` | Step-by-step traceability document mapping all 52 implementation steps and the 8 stages of the Point 10 roadmap to modules, classes, and tests |
| `examples/ore_rl_demo.py` | End-to-end demonstration of every roadmap stage (Stages 0-10) |
| `ore_logs/ore_rl_demo_report.json` | Machine-readable report produced by the last full demo run |
| `README.md` | Repository README updated with the `ore_rl/` entry |
| `pyproject.toml` | Packaging config updated to include `ore_rl*` in the installed packages |

## Running after extraction

From the repository root (requires the existing `ore_contracts` package and
its dependencies, which are unchanged):

```bash
python -m pytest tests/test_rl_*.py -q   # 94 acceptance tests
python -m ore_rl                         # deterministic baseline smoke run
python examples/ore_rl_demo.py           # full roadmap demonstration
python examples/ore_rl_demo.py --quick   # fast smoke version
```

The demo writes its report to `ore_logs/ore_rl_demo_report.json` and its
trajectory database to a timestamped file under `ore_logs/`, so it can be
re-run without collision.

## Verified results (full demo run)

- Deterministic baseline: success 0.70
- Contextual bandits (UCB / Thompson / LinUCB): 1.00 / 1.00 / 0.85
- DQN (Double + prioritized replay): 1.00
- PPO (clipped surrogate + GAE): 1.00
- Hierarchical options: 1.00
- Offline RL — CQL 1.00, IQL 0.90, BC 0.70
- Governance: shadow-mode records, promotion gates, rollback, and drift
  triggers all exercised; trajectory datasets checksum-verified.
