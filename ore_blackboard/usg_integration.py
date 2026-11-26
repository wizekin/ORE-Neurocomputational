"""Read-only USG knowledge integration for the blackboard coordinator (Step 5).

The blackboard never duplicates the USG database into the workspace and never
writes ORE inferences back into USG. Only a compact bundle reference is
published, while specialists receive USG-derived resources as typed,
read-only input artifacts.
"""
from __future__ import annotations

from typing import Any, Mapping

from ore_contracts import ContractEnvelope, EvidenceKind
from ore_contracts.module_artifacts import ContractArtifact
from ore_usg import USGKnowledgeAdapter, USGKnowledgeIndex, load_knowledge_bundle
from ore_usg.contracts import ORE_USG_KNOWLEDGE_CONTRACT, ORE_USG_LEXICON_CONTRACT
from ore_workspace.models import Provenance, utc_now

from .contracts import COGNITIVE_CONTROL_PRODUCER, USGSnapshotReference, sha256_json


class BlackboardUSGSnapshot:
    """Hold one digest-verified USG bundle with its adapted ORE resources (Step 5)."""

    def __init__(self, bundle: Mapping[str, Any], *, expected_digest: str | None) -> None:
        """Load, verify, and adapt the caller-supplied bundle without mutating it."""
        snapshot = load_knowledge_bundle(bundle, expected_digest=expected_digest)
        adapter = USGKnowledgeAdapter()
        self.bundle_snapshot = snapshot
        self.resource = adapter.convert(snapshot.bundle)
        self.index = USGKnowledgeIndex(self.resource)
        lexical = self.resource.lexical
        self.reference = USGSnapshotReference(
            namespace=self.resource.namespace,
            bundle_id=self.resource.bundle_id,
            bundle_digest=self.resource.bundle_digest,
            adapter_version=lexical.adapter_version,
            resource_categories={
                "lexical_entries": len(lexical.entries),
                "meanings": lexical.report.meanings_created,
                "concepts": len(self.resource.semantic.concepts),
                "semantic_classes": len(self.resource.structures.classes),
                "predicate_frames": len(self.resource.structures.predicate_frames),
                "relations": len(self.resource.structures.relations),
            },
        )

    def lexicon_artifact(self, run_id: str) -> ContractArtifact:
        """Wrap the imported lexicon resource as a read-only typed input artifact."""
        return ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=ORE_USG_LEXICON_CONTRACT,
                schema_version="1.0.0",
                run_id=run_id,
                producer_id="ore.blackboard.usg_reference",
                config_hash=sha256_json({"artifact": "usg_lexicon"}),
                created_at=utc_now(),
                evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
                data_hash=self.resource.bundle_digest,
            ),
            payload=self.resource.lexical,
            metadata={
                "usg_bundle_id": self.reference.bundle_id,
                "usg_bundle_digest": self.reference.bundle_digest,
                "read_only": True,
                "source_database_mutation_performed": False,
            },
        )

    def knowledge_artifact(self, run_id: str) -> ContractArtifact:
        """Wrap the unified knowledge resource as a read-only typed input artifact."""
        return ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=ORE_USG_KNOWLEDGE_CONTRACT,
                schema_version="1.0.0",
                run_id=run_id,
                producer_id="ore.blackboard.usg_reference",
                config_hash=sha256_json({"artifact": "usg_knowledge"}),
                created_at=utc_now(),
                evidence_kind=EvidenceKind.EXTERNAL_REFERENCE,
                data_hash=self.resource.bundle_digest,
            ),
            payload=self.resource,
            metadata={
                "usg_bundle_id": self.reference.bundle_id,
                "usg_bundle_digest": self.reference.bundle_digest,
                "read_only": True,
                "source_database_mutation_performed": False,
            },
        )


def publish_usg_reference(workspace, reference: USGSnapshotReference) -> Any:
    """Publish the compact bundle reference into ``artifact_references`` (Step 5)."""
    from ore_workspace.publication import FindingPublisher

    publisher = FindingPublisher(workspace, COGNITIVE_CONTROL_PRODUCER)
    finding = publisher.create_finding(
        "artifact_references",
        reference.to_dict(),
        key=f"usg_bundle:{reference.bundle_id}",
        finding_type="usg_knowledge_bundle_reference",
        provenance=Provenance(
            source=f"usg_knowledge_bundle:{reference.bundle_id}",
            rules_applied=("bundle_digest_verification", "read_only_import"),
            evidence=({
                "bundle_id": reference.bundle_id,
                "bundle_digest": reference.bundle_digest,
                "namespace": reference.namespace,
            },),
            notes="USG remains an independently owned, read-only knowledge source.",
        ),
    )
    return workspace.publish(finding)


__all__ = ["BlackboardUSGSnapshot", "publish_usg_reference"]
