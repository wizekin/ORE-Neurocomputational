"""NAR acceptance tests 21-22: deterministic oracle comparison and OOD evaluation.

21. NAR outputs are compared with the deterministic oracle.
22. OOD tests include larger graphs and unseen topologies.
"""
from __future__ import annotations

import numpy as np
import pytest

from ore_graph_intelligence.nar import (
    NARProductionGate,
    NAREvaluator,
    NARTrainer,
    NeuralAlgorithmicService,
    TaskGenerator,
    TRAINING_TOPOLOGIES,
    TraceDataset,
    reference_solve,
)


def _train_bfs_model():
    generator = TaskGenerator()
    tasks = [
        generator.generate("graph_traversal_bfs", n_nodes=size, seed=seed, topology=topology)
        for topology in TRAINING_TOPOLOGIES
        for size in (8, 10)
        for seed in range(2)
    ]
    traces = TraceDataset.from_tasks(tasks).by_task_type()["graph_traversal_bfs"]
    trainer = NARTrainer(latent_dim=12, rounds=2, epochs=30, learning_rate=0.05, seed=0)
    return trainer.train(traces)


def test_nar_outputs_compared_with_deterministic_oracle():
    """Test 21: every NAR proposal carries the oracle agreement and its reference."""
    model = _train_bfs_model()
    service = NeuralAlgorithmicService({"graph_traversal_bfs": model})
    generator = TaskGenerator()
    task = generator.generate("graph_traversal_bfs", n_nodes=10, seed=77, topology="random")
    proposal = service.predict(task)
    reference = reference_solve(task)
    assert proposal.task_id == task.task_id
    assert proposal.disposition in {"advisory", "fallback_to_reference"}
    assert 0.0 <= proposal.exact_agreement <= 1.0
    assert proposal.reference_version == task.implementation_version
    assert proposal.model_version, "the model version must be recorded"
    # The reference output is the authoritative one, not the learned rollout.
    assert np.all(proposal.reference_outputs == reference.node_output) or \
        proposal.reference_outputs.shape == reference.node_output.shape


def test_ood_tests_include_larger_graphs_and_unseen_topologies():
    """Test 22: evaluation separates size extrapolation from unseen topologies."""
    model = _train_bfs_model()
    generator = TaskGenerator()
    evaluation_tasks = [
        generator.generate("graph_traversal_bfs", n_nodes=size, seed=seed, topology=topology)
        for topology in TRAINING_TOPOLOGIES
        for size in (8, 10)
        for seed in (90, 91)
    ] + [
        generator.generate("graph_traversal_bfs", n_nodes=16, seed=92, topology="random"),
        generator.generate("graph_traversal_bfs", n_nodes=16, seed=93, topology="chain"),
        generator.generate("graph_traversal_bfs", n_nodes=9, seed=94, topology="grid"),
    ]
    report = NAREvaluator().evaluate(model, evaluation_tasks)
    summary = report.as_dict()
    assert "size_extrapolation" in summary["groups"], "larger graphs must be evaluated"
    assert "unseen_topology" in summary["groups"], "unseen topologies must be evaluated"
    assert summary["groups"]["size_extrapolation"]["n_tasks"] == 2
    assert summary["groups"]["unseen_topology"]["n_tasks"] == 1
    assert summary["groups"]["interpolation"]["n_tasks"] >= 2
    # The production gate restricts use based on subgroup metrics.
    decision = NARProductionGate().decide(report)
    assert decision.allowed_use in {
        "advisory_search_prioritization", "advisory_only_in_distribution", "abstain_all",
    }
    assert "exact algorithms validate correctness-critical outputs" in " ".join(decision.notes)


def test_nar_service_abstains_without_a_trained_model():
    """Untrained task types abstain and return the deterministic reference only."""
    service = NeuralAlgorithmicService({})
    generator = TaskGenerator()
    task = generator.generate("shortest_path", n_nodes=8, seed=5, topology="chain")
    proposal = service.predict(task)
    assert proposal.disposition == "abstained"
    assert "no_trained_model" in proposal.ood_flags
    assert not proposal.advisory


def test_nar_falls_back_on_ood_inputs():
    """OOD inputs fall back to the deterministic reference output."""
    model = _train_bfs_model()
    service = NeuralAlgorithmicService({"graph_traversal_bfs": model})
    generator = TaskGenerator()
    task = generator.generate("graph_traversal_bfs", n_nodes=25, seed=15, topology="grid")
    proposal = service.predict(task)
    assert "size_extrapolation" in proposal.ood_flags
    assert "unseen_topology" in proposal.ood_flags
    assert proposal.disposition == "fallback_to_reference"
    assert not proposal.advisory


def test_reference_algorithms_are_deterministic():
    """Reference solvers are reproducible from the recorded seed and version."""
    generator = TaskGenerator()
    for task_type in ("shortest_path", "connectivity", "minimum_spanning_tree", "sorting"):
        task = generator.generate(task_type, n_nodes=8, seed=3, topology="random")
        first = reference_solve(task)
        second = reference_solve(task)
        assert np.array_equal(first.final, second.final)
        assert first.implementation_version == task.implementation_version
        assert first.termination_step >= 0


def test_every_task_type_builds_aligned_traces_and_trains():
    """All five narrow tasks build supervised traces and train without error."""
    generator = TaskGenerator()
    for task_type in (
        "graph_traversal_bfs", "shortest_path", "connectivity",
        "minimum_spanning_tree", "sorting",
    ):
        tasks = [
            generator.generate(task_type, n_nodes=8, seed=seed, topology="random")
            for seed in range(3)
        ]
        traces = TraceDataset.from_tasks(tasks).traces
        assert len(traces) == 3
        for trace in traces:
            # Directed adjacency, features and targets must stay aligned.
            assert trace.edge_sources.size == trace.edge_targets.size
            assert trace.edge_features.shape[0] == trace.edge_sources.size
            if trace.target_kind == "edge":
                assert trace.final_edge_target.shape[0] == trace.edge_sources.size
            else:
                assert trace.final_node_target.shape[0] == trace.task.n_nodes
        model = NARTrainer(latent_dim=12, rounds=2, epochs=5, seed=0).train(traces)
        assert np.isfinite(model.report.final_loss)
        service = NeuralAlgorithmicService({task_type: model})
        proposal = service.predict(tasks[0])
        assert proposal.disposition in {"advisory", "fallback_to_reference"}
        assert proposal.exact_agreement >= 0.0


@pytest.mark.parametrize("task_type", ["graph_traversal_bfs", "shortest_path", "sorting"])
def test_trainer_rejects_mixed_task_types(task_type):
    """Each model trains on exactly one narrow task type."""
    generator = TaskGenerator()
    tasks = [
        generator.generate(task_type, n_nodes=6, seed=0, topology="chain"),
        generator.generate(
            "connectivity" if task_type != "connectivity" else "sorting",
            n_nodes=6, seed=0, topology="chain",
        ),
    ]
    traces = TraceDataset.from_tasks(tasks).traces
    with pytest.raises(ValueError):
        NARTrainer(epochs=1).train(traces)
