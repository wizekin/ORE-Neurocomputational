# Batch 02 — Point 02 installation

1. Copy the `ore_cognitive_control/` directory into the repository root.
2. Copy `tests/test_cognitive_control_point02.py` into `tests/`.
3. Copy `docs/ORE_POINT02_COGNITIVE_CONTROL.md` into `docs/`.
4. Add `ore_cognitive_control*` to the existing `[tool.setuptools.packages.find].include` list in `pyproject.toml`.
5. Run:

```bash
python -m pytest tests/test_cognitive_control_point02.py -q
```

This is a standalone foundation package. It does not change existing ORE runtime behavior until a separately reviewed adapter wires it to workspace, retrieval, goals, logging and scheduling APIs.
