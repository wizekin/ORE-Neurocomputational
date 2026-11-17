"""Acceptance tests for source-ordered Point 4 semantic composition."""

from __future__ import annotations

import ast
from pathlib import Path
import shutil

from ore_logger import LoggerDatabase
from ore_semantic_composition import PredicateFrame, SemanticCompositionConfig
from ore_language_comprehension import (
    POINT4_STEP_SPECS,
    Point4Request,
    Point4SemanticComposer,
    register_point04_annotations,
)


def _item(form, category, features=None, *, entry_id=None, meanings=None):
    """Build one Point 2-like lexical record with explicit semantic features and provenance."""
    meaning_rows = meanings or [{
        "meaning_id": "sense-1",
        "features": dict(features or {}),
        "posterior_within_entry": 1.0,
        "source_reference": f"test://{form}",
    }]
    return {
        "form": form,
        "confidence": 0.95,
        "category_candidates": {category.upper(): 0.98},
        "lexical_candidates": [{
            "entry_id": entry_id or form,
            "canonical_word_form": form,
            "lexical_category": category,
            "frequency": 12.0,
            "posterior": 0.96,
            "form_reliability": 0.95,
            "meaning_candidates": meaning_rows,
            "source_reference": f"test://{form}",
        }],
    }


def _syntax(tokens, selected, arcs, items):
    """Build a compact Point 3 result mapping with per-token incremental syntax snapshots."""
    updates = []
    for end in range(1, len(tokens) + 1):
        prefix_arcs = [
            dict(arc) for arc in arcs
            if int(arc["dependent_index"]) < end
            and (arc.get("head_index") is None or int(arc["head_index"]) < end)
        ]
        updates.append({
            "prefix_end_exclusive": end,
            "selected_pos": [dict(row) for row in selected[:end]],
            "dependency_relations": prefix_arcs,
            "constituent_groups": [],
            "constituency_alternatives": [],
        })
    return {
        "schema": "ore.language_comprehension.point03.v1",
        "tokens": list(tokens),
        "selected_pos": list(selected),
        "dependency_relations": list(arcs),
        "lexical_item_records": list(items),
        "constituent_groups": [],
        "constituency_tree": {},
        "syntax_hierarchy": {},
        "incremental_updates": updates,
    }


def _frame(predicate="chase", roles=("Agent", "Patient"), required=("Agent", "Patient"),
           preferences=None):
    """Return one explicit, provenance-labelled predicate frame for controlled tests."""
    return PredicateFrame(
        predicate=predicate,
        roles=roles,
        required_roles=required,
        selectional_preferences=preferences or {
            "Agent": {"ANIMATE": 1.0},
            "Patient": {"ANIMATE": 0.9},
        },
        source_reference=f"test://{predicate}-frame",
    )


def _chase_request(*, punctuation=True, frame=None):
    """Construct DOG-CHASE-CAT lexical and syntactic evidence with optional final punctuation."""
    tokens = ["dog", "chase", "cat"] + (["."] if punctuation else [])
    items = [
        _item("dog", "noun", {"ANIMATE": 1.0, "CANINE": 1.0}),
        _item("chase", "verb", {"EVENT": 1.0, "ACTION": 1.0}),
        _item("cat", "noun", {"ANIMATE": 1.0, "FELINE": 1.0}),
    ]
    selected = [
        {"token_index": 0, "token": "dog", "tag": "NOUN", "confidence": 0.98},
        {"token_index": 1, "token": "chase", "tag": "VERB", "confidence": 0.98},
        {"token_index": 2, "token": "cat", "tag": "NOUN", "confidence": 0.98},
    ]
    arcs = [
        {"dependent_index": 0, "head_index": 1, "relation": "nsubj", "confidence": 0.96},
        {"dependent_index": 1, "head_index": None, "relation": "root", "confidence": 1.0},
        {"dependent_index": 2, "head_index": 1, "relation": "obj", "confidence": 0.96},
    ]
    if punctuation:
        selected.append({"token_index": 3, "token": ".", "tag": "PUNCT", "confidence": 1.0})
        arcs.append({"dependent_index": 3, "head_index": 1, "relation": "punct", "confidence": 1.0})
        items.append({"form": ".", "confidence": 1.0, "lexical_candidates": []})
    return Point4Request.from_point3_result(
        "point4-chase",
        _syntax(tokens, selected, arcs, items),
        predicate_frames=(frame or _frame(),),
    )


def _composer(simulate_biology=False):
    """Create a small deterministic composer with optional reference LIF simulation."""
    return Point4SemanticComposer(
        config=SemanticCompositionConfig(
            simulate_biology=simulate_biology,
            population_size=4,
            steps_per_operation=1,
            max_scope_readings=4,
        ),
        register_annotations=False,
        max_candidate_readings=4,
    )


def test_point4_registry_maps_fourteen_steps_to_all_fifty_four_operations():
    """Verify stable source-level IDs, mathematical/biological descriptions and operation coverage."""
    assert len(POINT4_STEP_SPECS) == 14
    assert [row["operation_id"] for row in POINT4_STEP_SPECS] == [
        f"LC-P04-{index:02d}" for index in range(1, 15)
    ]
    assert all(row["mathematical_operations"] and row["biological_reference"] for row in POINT4_STEP_SPECS)
    from ore_language_comprehension.point04 import _POINT4_TO_SMC
    assert set(number for values in _POINT4_TO_SMC.values() for number in values) == set(range(1, 55))


def test_point3_handoff_builds_role_preserving_chase_proposition():
    """Compose the dog and cat into explicit Agent and Patient links around chase."""
    request = _chase_request()
    result = _composer().process(request)

    assert result.diagnostics["valid"] is True
    assert len(result.functional_trace) == 14
    assert len(result.word_concepts) == len(request.tokens)
    assert len(result.propositions) == 1
    proposition = result.propositions[0]
    assert proposition["predicate"] == "chase"
    assert {arg["surface"]: arg["role"] for arg in proposition["arguments"]} == {
        "dog": "Agent", "cat": "Patient"
    }
    assert result.proposition_graph["representation_type"] == "typed_sentence_semantic_graph"
    assert result.proposition_completeness[0]["status"] == "complete"


def test_subject_can_be_experiencer_instead_of_agent():
    """Use semantic feature preferences to map Maya to Experiencer and storm to Stimulus."""
    tokens = ["Maya", "fear", "storm", "."]
    items = [
        _item("Maya", "noun", {"HUMAN": 1.0, "ANIMATE": 1.0}),
        _item("fear", "verb", {"EVENT": 1.0, "MENTAL_STATE": 1.0}),
        _item("storm", "noun", {"WEATHER_EVENT": 1.0}),
        {"form": ".", "confidence": 1.0, "lexical_candidates": []},
    ]
    selected = [
        {"token_index": 0, "token": "Maya", "tag": "PROPN", "confidence": 0.98},
        {"token_index": 1, "token": "fear", "tag": "VERB", "confidence": 0.98},
        {"token_index": 2, "token": "storm", "tag": "NOUN", "confidence": 0.98},
        {"token_index": 3, "token": ".", "tag": "PUNCT", "confidence": 1.0},
    ]
    arcs = [
        {"dependent_index": 0, "head_index": 1, "relation": "nsubj"},
        {"dependent_index": 1, "head_index": None, "relation": "root"},
        {"dependent_index": 2, "head_index": 1, "relation": "obj"},
        {"dependent_index": 3, "head_index": 1, "relation": "punct"},
    ]
    frame = _frame(
        "fear",
        ("Experiencer", "Stimulus"),
        ("Experiencer", "Stimulus"),
        {"Experiencer": {"HUMAN": 1.0, "ANIMATE": 0.8}, "Stimulus": {"WEATHER_EVENT": 1.0}},
    )
    result = _composer().process(Point4Request.from_point3_result(
        "point4-experiencer", _syntax(tokens, selected, arcs, items), predicate_frames=(frame,)
    ))
    assert {row["argument"]: row["selected_role"] for row in result.thematic_roles} == {
        "Maya": "Experiencer", "storm": "Stimulus"
    }


def test_modifier_relations_are_composed_in_the_proposition_graph():
    """Create adjective-to-noun and adverb-to-event modifier edges from supplied syntax."""
    tokens = ["red", "dog", "chase", "cat", "quickly", "."]
    items = [
        _item("red", "adjective", {"COLOR_RED": 1.0}),
        _item("dog", "noun", {"ANIMATE": 1.0, "CANINE": 1.0}),
        _item("chase", "verb", {"EVENT": 1.0, "ACTION": 1.0}),
        _item("cat", "noun", {"ANIMATE": 1.0, "FELINE": 1.0}),
        _item("quickly", "adverb", {"MANNER_FAST": 1.0}),
        {"form": ".", "confidence": 1.0, "lexical_candidates": []},
    ]
    selected = [
        {"token_index": i, "token": token, "tag": tag, "confidence": 0.98}
        for i, (token, tag) in enumerate(zip(tokens, ["ADJ", "NOUN", "VERB", "NOUN", "ADV", "PUNCT"]))
    ]
    arcs = [
        {"dependent_index": 0, "head_index": 1, "relation": "amod"},
        {"dependent_index": 1, "head_index": 2, "relation": "nsubj"},
        {"dependent_index": 2, "head_index": None, "relation": "root"},
        {"dependent_index": 3, "head_index": 2, "relation": "obj"},
        {"dependent_index": 4, "head_index": 2, "relation": "advmod"},
        {"dependent_index": 5, "head_index": 2, "relation": "punct"},
    ]
    result = _composer().process(Point4Request.from_point3_result(
        "point4-modifiers",
        _syntax(tokens, selected, arcs, items),
        predicate_frames=(_frame(),),
    ))
    assert {row["modifier"] for row in result.modifier_relations} == {"red", "quickly"}
    assert {row["semantic_relation"] for row in result.modifier_relations} == {"HAS_PROPERTY", "MODIFIES"}
    assert any(row["edge_type"] == "modifier_composition" for row in result.proposition_graph["edges"])


def test_negation_modality_and_quantifier_scope_are_preserved():
    """Represent every/may/not as explicit operators over a proposition."""
    tokens = ["every", "dog", "may", "not", "chase", "cat", "."]
    items = [
        {"form": "every", "confidence": 0.98, "lexical_candidates": []},
        _item("dog", "noun", {"ANIMATE": 1.0, "CANINE": 1.0}),
        {"form": "may", "confidence": 0.98, "lexical_candidates": []},
        {"form": "not", "confidence": 0.98, "lexical_candidates": []},
        _item("chase", "verb", {"EVENT": 1.0}),
        _item("cat", "noun", {"ANIMATE": 1.0, "FELINE": 1.0}),
        {"form": ".", "confidence": 1.0, "lexical_candidates": []},
    ]
    selected = [
        {"token_index": i, "token": token, "tag": tag, "confidence": 0.98}
        for i, (token, tag) in enumerate(zip(tokens, ["DET", "NOUN", "AUX", "PART", "VERB", "NOUN", "PUNCT"]))
    ]
    arcs = [
        {"dependent_index": 0, "head_index": 1, "relation": "det"},
        {"dependent_index": 1, "head_index": 4, "relation": "nsubj"},
        {"dependent_index": 2, "head_index": 4, "relation": "aux"},
        {"dependent_index": 3, "head_index": 4, "relation": "advmod"},
        {"dependent_index": 4, "head_index": None, "relation": "root"},
        {"dependent_index": 5, "head_index": 4, "relation": "obj"},
        {"dependent_index": 6, "head_index": 4, "relation": "punct"},
    ]
    result = _composer().process(Point4Request.from_point3_result(
        "point4-operators", _syntax(tokens, selected, arcs, items), predicate_frames=(_frame(),)
    ))
    assert result.quantifier_scopes[0]["symbol"] == "∀"
    assert result.propositions[0]["polarity"] == "negative"
    assert result.propositions[0]["modal_operators"][0]["token"] == "may"


def test_ambiguous_bank_senses_remain_separate():
    """Keep finance and river-edge meanings as distinct candidates."""
    items = [{
        "form": "bank",
        "confidence": 0.85,
        "lexical_candidates": [
            {
                "entry_id": "bank-finance", "canonical_word_form": "bank",
                "lexical_category": "noun", "frequency": 10, "posterior": 0.55,
                "meaning_candidates": [{"meaning_id": "financial", "features": {"FINANCIAL_INSTITUTION": 1.0}, "posterior_within_entry": 1.0}],
            },
            {
                "entry_id": "bank-river", "canonical_word_form": "bank",
                "lexical_category": "noun", "frequency": 10, "posterior": 0.45,
                "meaning_candidates": [{"meaning_id": "river-edge", "features": {"RIVER_EDGE": 1.0}, "posterior_within_entry": 1.0}],
            },
        ],
    }]
    syntax = _syntax(
        ["bank"],
        [{"token_index": 0, "token": "bank", "tag": "NOUN", "confidence": 0.72}],
        [{"dependent_index": 0, "head_index": None, "relation": "root"}],
        items,
    )
    result = _composer().process(Point4Request.from_point3_result("point4-bank", syntax))
    ids = {row["concept_id"] for row in result.word_concepts[0]["candidates"]}
    assert ids == {"bank-finance::financial", "bank-river::river-edge"}


def test_incremental_updates_and_missing_argument_states_are_explicit():
    """Produce one partial semantic state per token and distinguish pending from unexpressed slots."""
    request = _chase_request(punctuation=False)
    frame = _frame(required=("Agent", "Patient"))
    open_request = Point4Request(
        sequence_id=request.sequence_id,
        tokens=request.tokens[:2],
        syntax_result={
            **dict(request.syntax_result),
            "tokens": list(request.tokens[:2]),
            "selected_pos": list(request.syntax_result["selected_pos"][:2]),
            "dependency_arcs": list(request.syntax_result["dependency_arcs"][:2]),
            "incremental_updates": list(request.syntax_result["incremental_updates"][:2]),
        },
        lexical_items=list(request.lexical_items[:2]),
        predicate_frames=(frame,),
        utterance_complete=False,
    )
    result = _composer().process(open_request)
    assert len(result.incremental_updates) == len(result.tokens)
    assert result.incremental_updates[-1]["is_final_sentence_state"] is True
    assert result.proposition_completeness[0]["missing_role_interpretation"] == "unresolved_pending_input"


def test_lif_reference_and_sqlite_annotations_remain_separate_and_shared(tmp_path: Path):
    """Validate the 14-step LIF aggregation and annotation registration in the existing SQLite catalogue."""
    result = _composer(True).process(_chase_request())
    assert result.biological_summary["model"] == "current_based_leaky_integrate_and_fire_reference"
    assert len(result.biological_summary["layers"]) == 9
    assert len(result.biological_summary["operation_states"]) == 54
    assert len(result.biological_trace) == 14
    assert result.biological_trace[0]["status"] == "simulated_reference_not_measured_human_data"
    assert result.diagnostics["biological_reference_is_measured_human_data"] is False

    source_root = Path(__file__).resolve().parents[1]
    isolated = tmp_path / "ore-project"
    isolated.mkdir()
    for package in (
        "ore_logger", "ore_language_comprehension", "ore_semantic_composition",
        "ore_semantic", "ore_syntax",
    ):
        shutil.copytree(source_root / package, isolated / package)
    (isolated / "pyproject.toml").write_text("[project]\nname='point4-annotation-test'\n", encoding="utf-8")
    database_path = tmp_path / "point4-annotations.sqlite3"
    report = register_point04_annotations(isolated, database_path)
    rows = LoggerDatabase(database_path).search_annotations("ore_language_comprehension", limit=1500)
    assert report["symbols_written"] > 0
    assert report["database_path"] == str(database_path)
    assert any(row["qualified_name"] == "Point4SemanticComposer.process" for row in rows)
    assert all(row["source_docstring"] for row in rows)


def test_every_language_comprehension_class_and_function_has_a_docstring():
    """Ensure Point 4 classes, helpers and earlier stage definitions are ready for SQLite indexing."""
    root = Path(__file__).resolve().parents[1] / "ore_language_comprehension"
    missing = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is None:
                    missing.append(f"{source.name}:{node.lineno}:{node.name}")
    assert missing == []
