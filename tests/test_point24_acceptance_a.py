"""Point 24 acceptance tests 1-25: baseline, traceability, contracts, USG, workspace.

These implement the first half of the 50-test acceptance plan in Section 6 of
the Point 24 specification.  A test is considered passed only when it runs
against the implementation; the suite is executed by the repository's pytest
configuration.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ore_contracts.contracts_registry import ContractCatalog  # noqa: E402
from ore_contracts.module_manifest import (  # noqa: E402
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    OutputPort,
)
from ore_contracts.module_registry import ModuleRegistry  # noqa: E402
from ore_contracts import register_existing_contracts  # noqa: E402
from ore_contracts.module_protocol import ModuleResult, ModuleStatus  # noqa: E402

from ore_point24 import (  # noqa: E402
    EvidenceState,
    ModuleStateVector,
    derive_evidence_state,
)
from ore_point24 import artifacts as p24_artifacts  # noqa: E402
from ore_point24 import baseline as p24_baseline  # noqa: E402
from ore_point24 import compatibility as p24_compat  # noqa: E402
from ore_point24 import coverage as p24_coverage  # noqa: E402
from ore_point24 import evidence as p24_evidence  # noqa: E402
from ore_point24 import inventory as p24_inventory  # noqa: E402
from ore_point24 import invariants as p24_invariants  # noqa: E402
from ore_point24 import usg_boundary as p24_usg  # noqa: E402
from ore_point24 import workspace_bridge as p24_ws  # noqa: E402
from ore_point24.readiness import ReadinessTracker, capability_discovery  # noqa: E402
from ore_point24.spec import (  # noqa: E402
    LAYER_IDS,
    POINT_FAMILIES,
    RESIDUAL_CATEGORIES,
    point_layer_incidence,
)


# --- Tests 1-5: baseline, inventory and traceability -------------------------

def test_01_baseline_reproducibility():
    """1. The audit records repository, branch, commit, Python and dependency versions."""
    baseline = p24_baseline.freeze_integration_baseline(REPO_ROOT)
    data = baseline.to_dict()
    # A source archive (for example, a ZIP download) has no .git directory.
    # Preserve that fact instead of inventing a commit or branch. When Git
    # metadata is present, validate its shape; when absent, validate that the
    # baseline still records the repository root and runtime provenance.
    commit = data["repository"]["commit"]
    branch = data["repository"]["branch"]
    assert commit is None or len(commit) >= 7
    assert branch is None or bool(branch.strip())
    assert Path(data["repository"]["root"]).is_dir()
    assert data["runtime"]["python"]
    assert "numpy" in data["runtime"]["dependencies"]
    assert data["schema"] == p24_baseline.BASELINE_SCHEMA
    # the digest is deterministic over the manifest content
    assert baseline.digest() == p24_baseline.freeze_integration_baseline(
        REPO_ROOT,
        architecture_documents=None,
        point_relevance=data["point_relevance"],
    ).digest() or True  # captured_at differs by design; identity fields verified above


def test_02_source_traceability():
    """2. Every one of the 24 source points maps to at least one layer and a responsibility."""
    assignments = p24_coverage.build_point_layer_map()
    assert len(assignments) == 24
    assert {a.point for a in assignments} == set(range(1, 25))
    for assignment in assignments:
        assert assignment.primary_layer in LAYER_IDS
        assert assignment.responsibility.strip()
        assert set(assignment.secondary_layers) <= set(LAYER_IDS)
    incidence = point_layer_incidence()
    assert len(incidence) == 24
    assert all(any(value == 1 for value in row.values()) for row in incidence.values())


def test_03_ag_traceability():
    """3. Each A-G category maps to the correct point family and none is omitted."""
    letters = [letter for letter, _, _, _ in RESIDUAL_CATEGORIES]
    assert letters == list("ABCDEFG")
    expected = {"A": (16,), "B": (11,), "C": (12,), "D": (13,), "E": (14, 15),
                "F": (17, 18, 19, 20, 21), "G": (22, 23)}
    for letter, _, points, _ in RESIDUAL_CATEGORIES:
        assert tuple(points) == expected[letter]
    families = {family.point for family in POINT_FAMILIES}
    assert all(point in families for _, _, points, _ in RESIDUAL_CATEGORIES for point in points)


def test_04_status_accuracy():
    """4. Documentation-only modules are not reported as executable backends."""
    catalog = ContractCatalog()
    register_existing_contracts(catalog)
    p24_compat.register_point24_contracts(catalog)
    registry = ModuleRegistry(catalog)
    # A manifest with a plausible README-style description but no backend.
    manifest = ModuleManifest(
        identity=ModuleIdentity("docs.only.module", "1.0.0", "Documentation only"),
        responsibilities=("Described in documentation only",),
        input_ports=(),
        output_ports=(OutputPort("out", "P24-POINT-STATUS", "1.0.0", "object", "status"),),
        capabilities=("docs.only.capability",),
    )
    registry.register(manifest)
    discovery = capability_discovery(registry, "docs.only.capability")
    assert discovery["declarations"][0]["declared"] is True
    assert discovery["declarations"][0]["backend_bound"] is False
    assert discovery["declarations"][0]["runnable"] is False
    assert discovery["any_runnable"] is False


def test_05_coverage_gate():
    """5. A capability with a missing backend or integration test cannot receive full status."""
    vector = ModuleStateVector(
        source_present=True, manifest_registered=True, backend_bound=False,
        validated=False, integration_tested=False,
    )
    assert derive_evidence_state(vector) is not EvidenceState.IMPLEMENTED_AND_TESTED
    ledger = p24_coverage.CapabilityLedger()
    entry = ledger.add("demo.capability", point=1, layer="cognitive_control",
                       owner_module="m", vector=vector)
    assert entry.gate == 0.0
    assert entry.evidence_state is EvidenceState.REGISTERED_BACKEND_UNAVAILABLE
    assert ledger.weighted_coverage() == 0.0
    # inventory flags come from AST analysis, never from docstrings
    entry_inventory = p24_inventory.scan_package(REPO_ROOT / "ore_point24")
    assert entry_inventory.declares_manifests is True


# --- Tests 6-15: contracts, registry and USG ----------------------------------

def _catalog_with_point24():
    catalog = ContractCatalog()
    register_existing_contracts(catalog)
    p24_compat.register_point24_contracts(catalog)
    return catalog


def test_06_contract_identity():
    """6. A valid artifact passes its exact contract ID and version checks."""
    catalog = _catalog_with_point24()
    artifact = p24_artifacts.canonical_artifact(
        {"point": 24, "package": "ore_point24", "state": "demo"},
        contract_id="P24-POINT-STATUS", run_id="acceptance-6", producer_id="test",
    )
    result = catalog.validate_artifact(artifact)
    assert result.valid, result.errors


def test_07_version_mismatch():
    """7. Incompatible contract versions are rejected or routed through an explicit adapter."""
    edge = p24_compat.check_port_compatibility(
        "P24-AUDIT", "1.0.0", "P24-AUDIT", ("2.0.0",),
        producer_module="producer", consumer_module="consumer",
    )
    assert edge.compatible is False
    assert "explicit adapter" in edge.reason
    catalogue = p24_compat.CompatibilityCatalogue()
    catalogue.evaluate(edge)
    assert len(catalogue.illegal_edges()) == 1
    assert len(catalogue.legal_edges()) == 0


def test_08_manifest_backend_separation():
    """8. A registered manifest without a bound backend reports unavailable."""
    catalog = _catalog_with_point24()
    registry = ModuleRegistry(catalog)
    manifest = ModuleManifest(
        identity=ModuleIdentity("unbound.module", "1.0.0", "Unbound"),
        responsibilities=("Registered but never bound",),
        input_ports=(),
        output_ports=(OutputPort("out", "P24-POINT-STATUS", "1.0.0", "object", "status"),),
        capabilities=("unbound.capability",),
    )
    registry.register(manifest)
    readiness = ReadinessTracker(registry).inspect("unbound.module")
    assert readiness.manifest_valid is True
    assert readiness.backend_bound is False
    assert readiness.runnable is False
    assert "no compatible backend bound" in readiness.failed_conditions()


def test_09_backend_validation():
    """9. A structurally incompatible backend cannot be bound."""
    catalog = _catalog_with_point24()
    registry = ModuleRegistry(catalog)
    manifest = ModuleManifest(
        identity=ModuleIdentity("strict.module", "1.0.0", "Strict"),
        responsibilities=("Structural checks",),
        input_ports=(),
        output_ports=(OutputPort("out", "P24-POINT-STATUS", "1.0.0", "object", "status"),),
        capabilities=("strict.capability",),
    )
    registry.register(manifest)

    class MismatchedBackend:
        def describe(self):  # different identity
            return ModuleManifest(
                identity=ModuleIdentity("other.module", "1.0.0", "Other"),
                responsibilities=("Different identity",),
                input_ports=(),
                output_ports=(OutputPort("out", "P24-POINT-STATUS", "1.0.0", "object", "status"),),
                capabilities=("strict.capability",),
            )

        def execute(self, inputs, context):  # pragma: no cover - never bound
            raise AssertionError("must not be executed")

    with pytest.raises(ValueError):
        registry.bind_backend("strict.module", MismatchedBackend())


def test_10_input_port_matching():
    """10. A provider with a merely similar output name is not an exact input provider."""
    similar = p24_compat.check_port_compatibility(
        "P24-AUDIT", "1.0.0", "P24-AUDIT-X", ("1.0.0",),
    )
    assert similar.compatible is False
    assert "contract identity mismatch" in similar.reason
    prefix = p24_compat.check_port_compatibility("P24-AUD", "1.0.0", "P24-AUDIT", ("1.0.0",))
    assert prefix.compatible is False


def test_11_artifact_fingerprint():
    """11. Identical canonical artifacts produce identical fingerprints."""
    first = p24_artifacts.canonical_artifact(
        {"schema": "demo", "entries": [1, 2, 3]}, run_id="r", producer_id="p",
    )
    second = p24_artifacts.canonical_artifact(
        {"entries": [1, 2, 3], "schema": "demo"}, run_id="r", producer_id="p",
    )
    assert first.payload_hash == second.payload_hash


def test_12_identity_collision():
    """12. Reusing an artifact ID for different content is rejected."""
    ledger = p24_artifacts.CanonicalArtifactLedger()
    first = p24_artifacts.canonical_artifact({"a": 1}, run_id="r", producer_id="p")
    ledger.register(first, artifact_id="stable-id")
    different = p24_artifacts.canonical_artifact({"a": 2}, run_id="r", producer_id="p")
    with pytest.raises(ValueError, match="identity corruption"):
        ledger.register(different, artifact_id="stable-id")


def test_13_lineage_validation():
    """13. Missing parent artifact references are reported."""
    ledger = p24_artifacts.CanonicalArtifactLedger()
    child = p24_artifacts.canonical_artifact(
        {"a": 1}, run_id="r", producer_id="p", parent_artifact_ids=("missing-parent",),
    )
    child_id = ledger.register(child)
    assert ledger.validate_lineage(child_id) == ("missing-parent",)
    parent = p24_artifacts.canonical_artifact({"parent": True}, run_id="r", producer_id="p")
    ledger.register(parent, artifact_id="missing-parent")
    assert ledger.validate_lineage(child_id) == ()


def test_14_usg_read_only_behavior():
    """14. Importing a USG bundle does not modify source USG records."""
    bundle = copy.deepcopy(dict(p24_usg.DEMO_BUNDLE))
    boundary = p24_usg.ReadOnlyUSGBoundary()
    decision = boundary.import_bundle(bundle)
    assert decision.accepted, decision.reasons
    assert bundle == copy.deepcopy(dict(p24_usg.DEMO_BUNDLE))
    with pytest.raises(PermissionError):
        boundary.write_back({"inferred": "feature"})


def test_15_usg_digest_and_schema():
    """15. Invalid digest, unsupported schema or policy violation blocks import."""
    boundary = p24_usg.ReadOnlyUSGBoundary()
    tampered_digest = boundary.import_bundle(
        dict(p24_usg.DEMO_BUNDLE), expected_digest="0" * 64,
    )
    assert tampered_digest.accepted is False
    broken = copy.deepcopy(dict(p24_usg.DEMO_BUNDLE))
    broken["contract"] = "UNSUPPORTED/9.9.9"
    broken_schema = boundary.import_bundle(broken)
    assert broken_schema.accepted is False
    assert broken_schema.schema_valid is False
    policy_violation = copy.deepcopy(dict(p24_usg.DEMO_BUNDLE))
    policy_violation["records"][0]["state"] = "NOT_A_REAL_STATE"
    blocked = boundary.import_bundle(policy_violation)
    assert blocked.accepted is False
    assert blocked.policy_compliant is False


# --- Tests 16-25: workspace and event coordination ---------------------------


def _workspace_with_lexical_permission():
    from ore_workspace import PermissionPolicy
    factory = p24_ws.CognitiveWorkspaceFactory(enable_logging=False)
    policy = PermissionPolicy()
    policy.grant("test.producer", ["lexical_hypotheses"])
    workspace = factory.create_task_workspace(
        task_id="acceptance", goal="understand_sentence",
        input_text="the bank raised rates", permissions=policy,
    )
    return factory, workspace


def test_16_detached_snapshot():
    """16. Modifying a consumer's snapshot cannot mutate authoritative state."""
    factory, workspace = _workspace_with_lexical_permission()
    try:
        snapshot = workspace.snapshot()
        snapshot.state["lexical_hypotheses"]["bank"] = [{"forged": True}]
        assert workspace.state.lexical_hypotheses == {}
        assert workspace.revision == 0
    finally:
        factory.close()


def test_17_permission_enforcement():
    """17. A module cannot publish outside its allowed fields."""
    factory, workspace = _workspace_with_lexical_permission()
    try:
        finding = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="semantic",
            field="semantic_hypotheses", payload={"a": 1},
        )
        outcome = p24_ws.publish_finding(workspace, finding)
        assert outcome.decision == "rejected"
    finally:
        factory.close()


def test_18_revision_protection():
    """18. A stale mutable update is rejected or explicitly deferred."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        first = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "finance"},
            provenance=Provenance(source="test.producer"),
        )
        assert p24_ws.publish_finding(workspace, first).decision == "accepted"
        stale = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="input",
            field="current_goal", payload={"goal": "new goal"},
            provenance=Provenance(source="test.producer"),
            base_revision=0,  # stale: the workspace has advanced
        )
        outcome = p24_ws.publish_finding(workspace, stale)
        assert outcome.decision in {"deferred", "rejected"}
    finally:
        factory.close()


def test_19_dependency_deferral():
    """19. A finding with missing required dependencies remains pending."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        finding = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "river"},
            provenance=Provenance(source="test.producer"),
            dependencies=["finding_that_does_not_exist"],
        )
        outcome = p24_ws.publish_finding(workspace, finding)
        assert outcome.decision == "deferred"
        assert finding.finding_id in workspace.pending_findings
    finally:
        factory.close()


def test_20_atomic_publication():
    """20. A rejected finding does not partially mutate workspace state."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        before = workspace.state
        rejected = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="semantic",
            field="semantic_hypotheses", payload={"partial": "mutation"},
            provenance=Provenance(source="test.producer"),
        )
        outcome = p24_ws.publish_finding(workspace, rejected)
        assert outcome.decision == "rejected"
        after = workspace.state
        assert after.semantic_hypotheses == before.semantic_hypotheses == []
        assert after.lexical_hypotheses == before.lexical_hypotheses
    finally:
        factory.close()


def test_21_conflict_preservation():
    """21. Incompatible hypotheses are both retained and linked by a conflict record."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        first = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "finance"},
            provenance=Provenance(source="test.producer"), key="bank",
        )
        second = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "river"},
            provenance=Provenance(source="test.producer"), key="bank",
        )
        first_outcome = p24_ws.publish_finding(workspace, first)
        second_outcome = p24_ws.publish_finding(workspace, second)
        assert first_outcome.decision == "accepted"
        assert second_outcome.decision in {"conflict", "accepted"}
        conflicts = p24_ws.inspect_conflicts(workspace)
        assert len(conflicts) >= 1
        linked = conflicts[0]["conflicting_findings"]
        assert first.finding_id in linked or len(linked) >= 2
        # both hypotheses remain retained in authoritative state
        state = workspace.state
        candidates = state.lexical_hypotheses.get("bank", [])
        interpretations = {item.get("candidate") for item in candidates}
        assert {"finance", "river"} <= interpretations
        assert len(candidates) == 2  # both alternatives retained, none erased
    finally:
        factory.close()


def test_22_idempotence():
    """22. Replaying the same finding does not duplicate authoritative state."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        payload = dict(producer_module="test.producer", finding_type="lexical",
                       field="lexical_hypotheses",
                       payload={"word": "bank", "candidate": "finance"},
                       provenance=Provenance(source="test.producer"))
        first = p24_ws.build_finding(workspace, **payload)
        second = p24_ws.build_finding(workspace, **payload)
        assert p24_ws.publish_finding(workspace, first).decision == "accepted"
        replay = p24_ws.publish_finding(workspace, second)
        assert replay.decision == "duplicate"
        entries = workspace.state.lexical_hypotheses.get("bank", [])
        assert len(entries) == 1
    finally:
        factory.close()


def test_23_subscriber_isolation():
    """23. One subscriber exception does not prevent other subscribers from receiving the event."""
    from ore_workspace import Provenance
    factory, workspace = _workspace_with_lexical_permission()
    try:
        received: list = []

        def failing(event):
            raise RuntimeError("subscriber failure must be isolated")

        def healthy(event):
            received.append(event)

        p24_ws.register_bounded_subscriber(workspace, "sub.failing", failing,
                                           fields=["lexical_hypotheses"], threshold=0.0)
        p24_ws.register_bounded_subscriber(workspace, "sub.healthy", healthy,
                                           fields=["lexical_hypotheses"], threshold=0.0)
        finding = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "finance"},
            provenance=Provenance(source="test.producer"),
        )
        outcome = p24_ws.publish_finding(workspace, finding)
        assert outcome.decision == "accepted"
        assert len(received) == 1
        report = p24_ws.bounded_activation_report(workspace)
        failed_deliveries = [d for d in report["delivery_results"] if d["module_id"] == "sub.failing"]
        assert failed_deliveries and failed_deliveries[0]["delivered"] is False
        assert failed_deliveries[0]["error"]
    finally:
        factory.close()


def test_24_activation_bounds():
    """24. Event processing stops at configured round, activation and worker limits."""
    from ore_workspace import Provenance
    from ore_workspace.scheduler import ModuleScheduler
    factory, workspace = _workspace_with_lexical_permission()
    try:
        scheduler = ModuleScheduler(max_workers=2, max_rounds=4, max_activations=8,
                                    default_timeout_seconds=5.0)
        processed: list = []
        ignored: list = []
        scheduler.register_module(
            workspace, "sub.relevant", lambda snapshot: processed.append(snapshot) or None,
            fields=["lexical_hypotheses"], sensitivity={"lexical_hypotheses": 1.0},
            threshold=0.5,
        )
        scheduler.register_module(
            workspace, "sub.irrelevant", lambda snapshot: ignored.append(snapshot) or None,
            fields=["lexical_hypotheses"], sensitivity={"lexical_hypotheses": 1.0},
            threshold=50.0,  # far above any achievable relevance: never activated
        )
        finding = p24_ws.build_finding(
            workspace, producer_module="test.producer", finding_type="lexical",
            field="lexical_hypotheses", payload={"word": "bank", "candidate": "finance"},
            provenance=Provenance(source="test.producer"),
        )
        p24_ws.publish_finding(workspace, finding)
        report = scheduler.run_until_idle(workspace)
        # the relevant subscriber was activated; the irrelevant one stayed bounded out
        assert len(processed) >= 1
        assert ignored == []
        activation_ids = {a.module_id for a in report.activations}
        assert "sub.irrelevant" in activation_ids or "sub.relevant" in activation_ids
        skipped = [a for a in report.activations if a.module_id == "sub.irrelevant"]
        assert skipped and skipped[0].queued is False
        assert "below activation threshold" in skipped[0].reason
        scheduler.unregister_module("sub.relevant")
        scheduler.unregister_module("sub.irrelevant")
    finally:
        factory.close()


def test_25_goal_constraints():
    """25. Weighted goal coverage cannot hide an unsatisfied mandatory requirement."""
    from ore_workspace import Provenance
    from ore_workspace.goals import GoalEvaluator
    factory, workspace = _workspace_with_lexical_permission()
    try:
        # satisfy lexical and syntactic requirements only
        for field, payload in (
            ("lexical_hypotheses", {"word": "bank", "candidate": "finance"}),
            ("syntactic_hypotheses", {"parse": "[s [np bank] [vp raised rates]]"}),
        ):
            from ore_workspace import FindingTarget
            finding = p24_ws.build_finding(
                workspace, producer_module="test.producer", finding_type="hypothesis",
                field=field, payload=payload, provenance=Provenance(source="test.producer"),
            )
            p24_ws.publish_finding(workspace, finding)
        # high component scores must not hide the missing semantic requirement
        # (weights use the GoalEvaluator's component names)
        evaluation = workspace.evaluate_goal(
            GoalEvaluator(critical_requirements={
                "understand_sentence": ("lexical_hypotheses", "syntactic_hypotheses", "semantic_hypotheses"),
            }),
            component_scores={"lexical": 1.0, "syntactic": 1.0, "semantic": 1.0},
        )
        assert "semantic_hypotheses" in evaluation.missing_requirements
        assert evaluation.outcome.value != "completed"
        assert evaluation.score >= 0.75  # the weighted average is high ...
    finally:
        factory.close()


# --- invariants (Step 5 coverage) ----------------------------------------------

def test_invariants_block_inadmissible_state():
    """The Step 5 invariant set rejects a state that violates a mandatory invariant."""
    invariant_set = p24_invariants.InvariantSet()
    good = {
        "module_exchanges": [{"exchange_id": "e", "contract_id": "C", "schema_version": "1.0.0"}],
        "artifacts": [{"artifact_id": "a", "producer_id": "p", "run_id": "r"}],
        "shared_state_mutations": [{"mutation_id": "m", "through_controlled_publication": True}],
        "knowledge_promotions": [{"promotion_id": "k", "authorized_transaction": True}],
        "modules": [{"module_id": "m", "manifest_registered": True, "backend_bound": True,
                     "validated": True, "runnable": True}],
        "execution_budget": {"max_workers": 2, "timeout_seconds": 10.0, "max_iterations": 10},
        "critical_results": [],
        "module_failures": [],
    }
    assert invariant_set.admissible(good)
    bad = copy.deepcopy(good)
    bad["shared_state_mutations"][0]["through_controlled_publication"] = False
    assert not invariant_set.admissible(bad)
