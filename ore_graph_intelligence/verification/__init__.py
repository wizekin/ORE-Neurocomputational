"""Symbolic and evidence-based verification of graph candidates (Phase D)."""
from .contradiction_search import ContradictionSearch, CounterEvidenceReport
from .evidence_checker import EvidenceCheck, EvidenceChecker
from .promotion_gate import (
    AuthorizedUpdateProposal,
    CandidateVerifier,
    PromotionGate,
    VerificationResult,
)
from .type_checker import CandidateTypeChecker, TypeCheckResult
from .usg_rule_adapter import RuleViolation, USGRuleAdapter

__all__ = [
    "AuthorizedUpdateProposal", "CandidateTypeChecker", "CandidateVerifier",
    "ContradictionSearch", "CounterEvidenceReport", "EvidenceCheck", "EvidenceChecker",
    "PromotionGate", "RuleViolation", "TypeCheckResult", "USGRuleAdapter",
    "VerificationResult",
]
