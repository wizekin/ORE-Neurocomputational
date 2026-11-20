"""Tests for deterministic, read-only USG knowledge bundle discovery/loading."""
from __future__ import annotations

import json

import pytest

from ore_usg import (
    USGBundleLoadError,
    discover_knowledge_bundle,
    load_knowledge_bundle,
)
from test_usg_lexical_adapter import _bundle


def test_load_bundle_file_validates_and_returns_canonical_digest(tmp_path):
    bundle = _bundle()
    path = tmp_path / "knowledge_bundle.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

    snapshot = load_knowledge_bundle(path)

    assert snapshot.bundle == bundle
    assert snapshot.source_path == str(path.resolve())
    assert len(snapshot.bundle_digest) == 64
    assert snapshot.byte_length == path.stat().st_size


def test_directory_discovery_is_deterministic_and_rejects_ambiguity(tmp_path):
    bundle = _bundle()
    first = tmp_path / "knowledge_bundle.json"
    first.write_text(json.dumps(bundle), encoding="utf-8")
    assert discover_knowledge_bundle(tmp_path) == first
    (tmp_path / "bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
    with pytest.raises(USGBundleLoadError, match="Multiple conventional"):
        discover_knowledge_bundle(tmp_path)


def test_expected_digest_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps(_bundle()), encoding="utf-8")
    with pytest.raises(USGBundleLoadError, match="digest mismatch"):
        load_knowledge_bundle(path, expected_digest="0" * 64)


def test_duplicate_json_keys_and_oversized_files_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"contract":"x","contract":"y"}', encoding="utf-8")
    with pytest.raises(USGBundleLoadError, match="Duplicate JSON object key"):
        load_knowledge_bundle(duplicate)
    with pytest.raises(USGBundleLoadError, match="exceeding configured limit"):
        load_knowledge_bundle(duplicate, max_bytes=2)


def test_mapping_input_is_detached_from_caller():
    bundle = _bundle()
    snapshot = load_knowledge_bundle(bundle)
    bundle["records"].clear()
    assert snapshot.bundle["records"]
    assert snapshot.source_path is None
