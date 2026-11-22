# ORE-Neurocomputational — AI Agent Working Guide

Use this as the compact operating contract when another AI model is asked to modify this repository. The full system description is in `ORE-Neurocomputational_AI_Model_Specification.md`.

## Before coding

1. Confirm repository, branch, and HEAD commit.
2. Read `README.md`, the relevant `docs/*.md` architecture specification, the target package's `__init__.py`, contracts, config, pipeline, functional/biology modules and registry.
3. Read the focused tests and all integration tests that consume the target package.
4. Search for public API callers and adapters before altering signatures or serialized structures.
5. Keep legacy `ore_n1`–`ore_n9` identifiers separate from canonical `F-A`–`F-J` and `B-N1`–`B-N9` identifiers.

## Non-negotiable design rules

- Preserve backward compatibility unless the task explicitly calls for a breaking change.
- Prefer additive adapters and exact-version contract changes over cross-package rewrites.
- Preserve provenance, source IDs, hashes, parent artifacts, model fingerprints, dataset splits and transformation lineage.
- Do not treat activations or uncalibrated confidence as calibrated probabilities.
- Keep observation, prediction, inference, learning and external reference evidence distinct.
- Preserve alternatives, unknown mass, unresolved data and abstention; do not force unsupported decisions.
- A missing backend remains `unavailable` / `not_configured`; never invent a result.
- Ordinary inference is read-only with respect to learned model state. Learning requires explicit authorization, verification/provenance gates and a detached child model where specified.
- Functional output and biological-reference simulation are separate. Simulated LIF, EEG/MEG/BOLD, lesion or TMS output is not empirical validation.
- Bound recurrence, retries, timeouts, scheduler rounds and resource budgets.
- Preserve stable operation IDs and module IDs; never recycle IDs for new meanings.

## Test and report

Run a focused test first, then related integration suites, then `python -m pytest -q`. Report exact commands, interpreter/dependency context and results. A test file's existence is not evidence that it passed. Update tests, operation registries, docs, examples and package exports together.

## Shared infrastructure boundaries

- `ore_contracts`: typed schemas, manifests, artifacts, compatibility, adapters, task planning and canonical controller interfaces.
- `ore_workspace`: validated shared state, revisions, permissions, conflicts, subscriptions and goal-aware module scheduling.
- `ore_logger`: SQLite source annotations and run/event observability.
- `experiment_workbench`: local Streamlit execution/inspection harness; run trusted code only and do not expose it publicly without security controls.
- `ore_usg`: read-only import of caller-supplied USG-KNOWLEDGE bundles; never write back to the source database or invent unsupported lexical/semantic links.

## Scientific and operational honesty

Describe implementations as prototypes, approximations or computational hypotheses unless independent empirical evidence supports stronger wording. Distinguish code implemented, software-tested, task-validated and biologically supported statuses. Never infer one status from another.
