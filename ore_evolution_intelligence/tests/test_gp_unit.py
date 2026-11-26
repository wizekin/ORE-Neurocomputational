"""Genetic-programming unit tests (Phase D, steps 31-38; acceptance tests 16-18)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.domain.actions import ActionCall
from ore_evolution_intelligence.evaluation.evaluator import SeededRandom
from ore_evolution_intelligence.search.genetic_programming import (
    ACTION_OPS, BOOL_OPS, COMPARE_OPS, DEFAULT_GRAMMAR, NUM_OPS, POLICY_CONSTANTS,
    POLICY_FEATURES, ActionNode, BoolOpNode, CombineActionNode, CompareNode,
    ConstantNode, FeatureNode, IfActionNode, NumOpNode, PolicyIRGenome,
    TypedGrammar, WhenActionNode, node_from_dict, random_tree, static_constant_value,
    subtree_crossover, subtree_mutation,
)
from ore_evolution_intelligence.verification.semantic_rule_check import (
    SemanticRuleCheck,
)
from ore_evolution_intelligence.verification.static_policy_check import (
    StaticPolicyCheck,
)

CONTEXT = {
    "complexity": 0.6, "novelty": 0.3, "latency_sensitivity": 0.5,
    "resource_pressure": 0.0, "safety_critical": 0.0, "task_family_index": 2.0,
}


class TestGrammarBounds:
    """Steps 31-32: typed grammar with depth/node bounds (acceptance test 16)."""

    def test_random_trees_are_type_and_bound_valid(self):
        rng = SeededRandom(42)
        for _ in range(50):
            tree = random_tree(DEFAULT_GRAMMAR, rng)
            assert tree.ir_type == "action"
            assert DEFAULT_GRAMMAR.validate(tree) == []
            assert tree.depth() <= DEFAULT_GRAMMAR.max_depth
            assert tree.size() <= DEFAULT_GRAMMAR.max_nodes

    def test_depth_bound_violation_detected(self):
        grammar = TypedGrammar(max_depth=3, max_nodes=32)
        deep = IfActionNode(
            CompareNode("lt", (FeatureNode("complexity"), ConstantNode(0.5))),
            IfActionNode(
                CompareNode("gt", (FeatureNode("novelty"), ConstantNode(0.25))),
                IfActionNode(
                    CompareNode("gt", (FeatureNode("novelty"), ConstantNode(0.25))),
                    ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
                    ActionNode(ActionCall("defer_task", {"reason": "resource_pressure"})),
                ),
                ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
            ),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        )
        errors = grammar.validate(deep)
        assert any("depth" in error for error in errors)

    def test_unknown_feature_detected(self):
        tree = ActionNode(ActionCall("activate_module", {"module": "ore_syntax"}))
        grammar = TypedGrammar(
            features=POLICY_FEATURES[:2],  # complexity, novelty only
        )
        genome = PolicyIRGenome(
            IfActionNode(
                CompareNode("gt", (FeatureNode("safety_critical"), ConstantNode(0.5))),
                tree, tree,
            ),
            grammar=grammar,
        )
        errors = genome.validate()
        assert any("unknown feature terminal" in e for e in errors)

    def test_type_mismatch_detected(self):
        # A numeric operand in a boolean position must fail validation.
        broken = BoolOpNode("and", (ConstantNode(0.5), ConstantNode(0.5)))
        root = WhenActionNode(
            CompareNode("gt", (FeatureNode("complexity"), ConstantNode(0.5))),
            ActionNode(ActionCall("activate_module", {"module": "ore_syntax"})),
        )
        bad_root = WhenActionNode(broken, root)
        errors = DEFAULT_GRAMMAR.validate(bad_root)
        assert any("needs 'bool' operands" in e for e in errors)

    def test_unapproved_constant_detected(self):
        tree = IfActionNode(
            CompareNode("gt", (FeatureNode("complexity"), ConstantNode(0.33))),
            ActionNode(ActionCall("activate_module", {"module": "ore_reading"})),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        )
        errors = DEFAULT_GRAMMAR.validate(tree)
        assert any("approved terminal" in e for e in errors)

    def test_operator_allowlists(self):
        assert set(NUM_OPS) <= {
            "add", "sub", "mul", "div", "min", "max", "clamp01",
        }
        assert set(BOOL_OPS) <= {"and", "or", "not"}
        assert set(COMPARE_OPS) <= {"lt", "gt"}
        assert set(ACTION_OPS) <= {"if", "when", "combine"}


class TestPolicyIRInterpreter:
    """Step 33: declarative policy IR with bounded, guarded evaluation."""

    def _escalate_tree(self):
        safety = FeatureNode("safety_critical")
        activate = ActionNode(ActionCall("activate_module", {"module": "ore_semantic"}))
        return PolicyIRGenome(
            CombineActionNode(
                WhenActionNode(
                    CompareNode("gt", (safety, ConstantNode(0.5))),
                    ActionNode(ActionCall("escalate_verification", {})),
                ),
                activate,
            ),
            grammar=DEFAULT_GRAMMAR,
        )

    def test_evaluation_deterministic(self):
        genome = self._escalate_tree()
        actions_one, steps_one, terminated_one = genome.evaluate_actions(CONTEXT)
        actions_two, steps_two, terminated_two = genome.evaluate_actions(CONTEXT)
        assert (actions_one, steps_one, terminated_one) == (
            actions_two, steps_two, terminated_two,
        )
        assert not terminated_one
        assert any(call.name == "activate_module" for call in actions_one)

    def test_step_budget_terminates(self):
        genome = self._escalate_tree()
        actions, steps, terminated = genome.evaluate_actions(CONTEXT)  # warm
        tiny = PolicyIRGenome(
            genome.root, grammar=DEFAULT_GRAMMAR, max_steps=1,
        )
        actions, steps, terminated = tiny.evaluate_actions(CONTEXT)
        assert terminated
        assert steps >= 1

    def test_missing_feature_terminates_as_invalid(self):
        genome = self._escalate_tree()
        actions, steps, terminated = genome.evaluate_actions({"complexity": 0.5})
        assert terminated

    def test_guarded_division(self):
        division = NumOpNode(
            "div", (ConstantNode(0.5), NumOpNode("sub", (ConstantNode(0.25), ConstantNode(0.25)))),
        )
        assert division.evaluate({}, _budget()) == 1.0  # guard: zero denominator

    def test_serialization_round_trip(self):
        rng = SeededRandom(99)
        genome = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
        restored = PolicyIRGenome.from_dict(genome.as_dict())
        assert restored.content_payload() == genome.content_payload()
        assert node_from_dict(genome.root.as_dict()).as_dict() == genome.root.as_dict()


class TestSubtreeOperators:
    """Step 34: subtree crossover/mutation with revalidation (acceptance 15)."""

    def test_subtree_crossover_valid_or_rejected(self):
        rng = SeededRandom(7)
        for _ in range(40):
            a = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
            b = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
            offspring = subtree_crossover(a, b, rng)
            assert offspring is None or offspring.validate() == []

    def test_subtree_mutation_valid_or_rejected(self):
        rng = SeededRandom(8)
        for _ in range(40):
            genome = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
            mutant = subtree_mutation(genome, rng)
            assert mutant is None or mutant.validate() == []

    def test_cross_grammar_crossover_rejected(self):
        rng = SeededRandom(9)
        # A grammar with an extra feature terminal produces trees that are invalid
        # under the default grammar, so offspring are rejected or stay valid.
        other_grammar = TypedGrammar(
            features=POLICY_FEATURES + (("mystery_feature", "not in default"),),
        )
        a = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
        b = PolicyIRGenome(random_tree(other_grammar, rng), other_grammar)
        offspring = subtree_crossover(a, b, rng)
        assert offspring is None or offspring.validate() == []


class TestStaticChecks:
    """Step 36: static and semantic checks (acceptance tests 17, 18)."""

    def test_generated_trees_pass_static_checks(self):
        # Acceptance test 17: policy IR passes static checks before execution.
        rng = SeededRandom(21)
        check = StaticPolicyCheck()
        for _ in range(25):
            genome = PolicyIRGenome(random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR)
            report = check.check(genome)
            assert report.ok, [f.as_dict() for f in report.findings]

    def test_unreachable_branch_detected(self):
        # Step 36: unreachable branches flagged statically.
        constant_true = CompareNode("lt", (ConstantNode(0.25), ConstantNode(0.5)))
        tree = IfActionNode(
            constant_true,
            ActionNode(ActionCall("activate_module", {"module": "ore_syntax"})),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        )
        genome = PolicyIRGenome(tree, DEFAULT_GRAMMAR)
        report = StaticPolicyCheck().check(genome)
        codes = [finding.code for finding in report.findings]
        assert "unreachable-branch" in codes

    def test_complexity_tracked(self):
        # Step 35: complexity tracked (tree size/depth).
        rng = SeededRandom(5)
        genome = PolicyIRGeneme = PolicyIRGenome(
            random_tree(DEFAULT_GRAMMAR, rng), DEFAULT_GRAMMAR,
        )
        report = StaticPolicyCheck().check(genome)
        assert report.complexity["nodes"] == genome.root.size()
        assert report.complexity["depth"] == genome.root.depth()

    def test_constant_folding_helper(self):
        assert static_constant_value(ConstantNode(0.5)) == 0.5
        folded = NumOpNode(
            "add", (ConstantNode(0.25), ConstantNode(0.5)),
        )
        assert static_constant_value(folded) == pytest.approx(0.75)
        assert static_constant_value(FeatureNode("complexity")) is None


class TestSemanticChecks:
    """Step 37: semantic rule checks (acceptance test 18)."""

    def _guard(self, feature="complexity", op="gt", constant=0.5):
        return CompareNode(op, (FeatureNode(feature), ConstantNode(constant)))

    def test_contradictory_activate_deactivate_detected(self):
        # Acceptance test 18: contradictory rules flagged.
        guard = self._guard()
        rules = CombineActionNode(
            WhenActionNode(
                guard,
                ActionNode(ActionCall("activate_module", {"module": "ore_semantic"})),
            ),
            WhenActionNode(
                self._guard(),
                ActionNode(ActionCall("deactivate_module", {"module": "ore_semantic"})),
            ),
        )
        genome = PolicyIRGenome(rules, DEFAULT_GRAMMAR)
        report = SemanticRuleCheck().check(genome)
        codes = [finding.code for finding in report.findings]
        assert "contradictory-rules" in codes
        assert not report.ok

    def test_deactivation_without_activation_warned(self):
        rules = WhenActionNode(
            self._guard(),
            ActionNode(ActionCall("deactivate_module", {"module": "ore_syntax"})),
        )
        genome = PolicyIRGenome(rules, DEFAULT_GRAMMAR)
        report = SemanticRuleCheck().check(genome)
        codes = [finding.code for finding in report.findings]
        assert "deactivation-without-activation" in codes

    def test_defer_and_activate_conflict_detected(self):
        guard = self._guard()
        rules = IfActionNode(
            guard,
            CombineActionNode(
                ActionNode(ActionCall("activate_module", {"module": "ore_reading"})),
                ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
            ),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        )
        genome = PolicyIRGenome(rules, DEFAULT_GRAMMAR)
        report = SemanticRuleCheck().check(genome)
        assert "defer-conflict" in [f.code for f in report.findings]

    def test_clean_tree_has_no_errors(self):
        safety = FeatureNode("safety_critical")
        genome = PolicyIRGenome(
            CombineActionNode(
                WhenActionNode(
                    CompareNode("gt", (safety, ConstantNode(0.5))),
                    ActionNode(ActionCall("escalate_verification", {})),
                ),
                ActionNode(ActionCall("activate_module", {"module": "ore_reading"})),
            ),
            DEFAULT_GRAMMAR,
        )
        report = SemanticRuleCheck().check(genome)
        assert report.ok

    def test_non_ir_genomes_have_empty_semantic_surface(self):
        from .fixtures import make_baseline_genome
        report = SemanticRuleCheck().check(make_baseline_genome())
        assert report.rules_checked == 0


def _budget():
    from ore_evolution_intelligence.search.genetic_programming import (
        _StepBudget,
    )
    return _StepBudget(256)


class TestIRStructure:
    def test_depth_and_size(self):
        tree = IfActionNode(
            CompareNode("lt", (FeatureNode("complexity"), ConstantNode(0.5))),
            ActionNode(ActionCall("activate_module", {"module": "ore_syntax"})),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        )
        assert tree.depth() == 3
        assert tree.size() == 6
        assert CombineActionNode(
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
            ActionNode(ActionCall("defer_task", {"reason": "low_confidence"})),
        ).size() == 3
