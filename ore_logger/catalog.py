"""AST-backed annotation catalogue for every Python source file in the repository."""
from __future__ import annotations

import ast
import hashlib
import inspect
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .database import LoggerDatabase, utc_now


_SKIP_PARTS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".ruff_cache", "build", "dist", "ore_logs", "site-packages",
}
_ACTIONS = {
    "validate": "Checks whether supplied values or state satisfy the required rules",
    "check": "Checks a condition and reports whether it is satisfied",
    "ensure": "Makes sure a required condition or invariant holds",
    "build": "Builds a structured result from the supplied inputs",
    "construct": "Constructs a new representation from the supplied inputs",
    "create": "Creates a new object, representation, or resource",
    "make": "Creates or prepares the named result",
    "compute": "Computes the requested numerical or logical result",
    "calculate": "Calculates the requested value from its inputs",
    "estimate": "Estimates a value from available evidence",
    "measure": "Measures or quantifies the requested property",
    "normalize": "Normalizes values into a comparable scale or valid distribution",
    "normalise": "Normalizes values into a comparable scale or valid distribution",
    "convert": "Converts data from one representation into another",
    "adapt": "Adapts an input to the contract expected by another component",
    "transform": "Transforms an input representation while preserving its intended information",
    "load": "Loads the requested data or model resource",
    "retrieve": "Retrieves the requested information from the supplied resource",
    "read": "Reads and interprets the requested source data",
    "save": "Saves the requested result or state",
    "write": "Writes the requested data to its destination",
    "export": "Exports data into the requested external representation",
    "import": "Imports and prepares data from another module or format",
    "run": "Runs the named processing procedure",
    "process": "Processes the input through the component's configured operations",
    "execute": "Executes the selected operation using the supplied arguments",
    "pipeline": "Coordinates the sequence of processing stages",
    "detect": "Detects the requested pattern or condition in the available evidence",
    "find": "Finds matching items or evidence in the supplied data",
    "identify": "Identifies the requested item or category from the available evidence",
    "infer": "Infers the requested state from the supplied evidence",
    "predict": "Predicts likely values or states from the available model information",
    "rank": "Ranks candidate alternatives according to their scores or evidence",
    "select": "Selects an alternative according to the configured criteria",
    "choose": "Chooses an output according to the supplied evidence and decision rules",
    "update": "Updates an existing state or representation",
    "set": "Sets the named configuration or state value",
    "initialize": "Initializes the state required by later operations",
    "initialise": "Initializes the state required by later operations",
    "reset": "Resets the relevant state to its defined initial condition",
    "fit": "Fits or prepares a model using the supplied examples",
    "train": "Trains the model using the supplied training evidence",
    "learn": "Updates a learning representation under the configured learning rules",
    "simulate": "Simulates the specified dynamical or computational process",
    "record": "Records the requested observation or diagnostic information",
    "emit": "Emits the requested event or structured output",
    "mark": "Records that a named processing operation has been reached",
    "adapt": "Adapts the input or output to a component contract",
    "summarize": "Summarizes the supplied data or result",
    "inspect": "Inspects the supplied object and extracts relevant properties",
    "compare": "Compares alternatives or representations using the implemented criteria",
    "merge": "Merges compatible values or representations",
    "combine": "Combines evidence or representations into a joint result",
    "split": "Splits a representation into the specified parts",
    "parse": "Parses the input into a structured representation",
    "tokenize": "Divides text into token units",
    "resolve": "Resolves competing alternatives using the available constraints and evidence",
    "apply": "Applies the named transformation or rule to the input",
    "generate": "Generates the requested candidates or representation",
    "prepare": "Prepares data or state required by later processing",
    "initialize": "Initializes the requested computational state",
    "decode": "Decodes the input representation into candidate output symbols",
    "encode": "Encodes the input into the requested representation",
    "pool": "Pools local information into a more compact representation",
    "filter": "Filters the input according to the configured criteria",
    "smooth": "Smooths local variation while retaining the larger-scale signal",
    "reset": "Resets the selected internal state",
    "forward": "Computes the forward-direction transformation or prediction",
    "backward": "Computes the reverse-direction transformation or prediction",
    "simulate": "Advances a computational simulation and records its state",
}


def _module_name(relative_path: str) -> str:
    path = Path(relative_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__root__"


def _symbol_id(source_path: str, kind: str, qualified_name: str) -> str:
    raw = f"{source_path}|{kind}|{qualified_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def _clean_docstring(node: ast.AST) -> str:
    value = ast.get_docstring(node, clean=True)
    if not value:
        return ""
    return value.strip()


def _argument_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = node.args
    values = [item.arg for item in args.posonlyargs + args.args]
    if args.vararg:
        values.append("*" + args.vararg.arg)
    values.extend(item.arg for item in args.kwonlyargs)
    if args.kwarg:
        values.append("**" + args.kwarg.arg)
    return [value for value in values if value not in {"self", "cls"}]


def _direct_calls(node: ast.AST, limit: int = 6) -> list[str]:
    class CallVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.names: list[str] = []

        def visit_FunctionDef(self, item: ast.FunctionDef) -> None:
            if item is node:
                for child in item.body:
                    self.visit(child)

        def visit_AsyncFunctionDef(self, item: ast.AsyncFunctionDef) -> None:
            if item is node:
                for child in item.body:
                    self.visit(child)

        def visit_ClassDef(self, item: ast.ClassDef) -> None:
            return

        def visit_Lambda(self, item: ast.Lambda) -> None:
            return

        def visit_Call(self, item: ast.Call) -> None:
            name = ast.unparse(item.func)
            if name not in self.names:
                self.names.append(name)
            self.generic_visit(item)

    visitor = CallVisitor()
    visitor.visit(node)
    return visitor.names[:limit]


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
        result = f"({args})"
        if node.returns is not None:
            result += f" -> {ast.unparse(node.returns)}"
        return result
    except Exception:
        return "(signature unavailable)"


def _verb_explanation(name: str) -> str:
    words = [word for word in name.split("_") if word]
    if not words:
        return "Implements a Python processing operation."
    action = _ACTIONS.get(words[0].lower())
    if action:
        target = " ".join(words[1:]) or "the requested item"
        return f"{action}: {target}."
    return f"Implements the { ' '.join(words) } operation."


def _function_annotation(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    qualified_name: str,
    module_name: str,
) -> tuple[str, str, str]:
    docstring = _clean_docstring(node)
    signature = _function_signature(node)
    if docstring:
        annotation = docstring
        source = "docstring"
    else:
        annotation = _verb_explanation(node.name)
        params = _argument_names(node)
        if params:
            annotation += " It accepts " + ", ".join(params) + " as inputs."
        if node.returns is not None:
            try:
                annotation += f" Its declared return type is {ast.unparse(node.returns)}."
            except Exception:
                pass
        calls = _direct_calls(node)
        if calls:
            annotation += " Its implementation delegates part of the work to: " + ", ".join(calls) + "."
        loops = sum(isinstance(child, (ast.For, ast.AsyncFor, ast.While)) for child in ast.walk(node))
        branches = sum(isinstance(child, (ast.If, ast.Match)) for child in ast.walk(node))
        if loops:
            annotation += f" The implementation contains {loops} loop construct(s) to handle repeated items or iterative updates."
        if branches:
            annotation += f" It contains {branches} conditional branch(es) for input-dependent behavior."
        annotation += f" This callable is defined in the {module_name} module."
        source = "source_analysis"
    return annotation, signature, source


def _class_annotation(
    node: ast.ClassDef,
    qualified_name: str,
    module_name: str,
) -> tuple[str, str, str]:
    docstring = _clean_docstring(node)
    bases = [ast.unparse(base) for base in node.bases]
    methods = [
        item.name for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not item.name.startswith("_")
    ]
    if docstring:
        annotation = docstring
        source = "docstring"
    else:
        base_text = f" It extends {', '.join(bases)}." if bases else ""
        annotation = f"Defines the {node.name} class in the {module_name} module.{base_text}"
        if methods:
            annotation += " Its public methods include " + ", ".join(methods[:12]) + "."
            if len(methods) > 12:
                annotation += f" It defines {len(methods)} public methods in total."
        source = "source_analysis"
    if docstring and methods:
        annotation += "\n\nPublic methods: " + ", ".join(methods[:12]) + "."
    return annotation, "", source


def _module_annotation(node: ast.Module, module_name: str, source_path: str) -> str:
    docstring = _clean_docstring(node)
    if docstring:
        return docstring
    stem = Path(source_path).stem
    if stem == "__init__":
        subject = Path(source_path).parent.name or "the repository"
        return f"Package entry module for {subject}. It defines or exposes the package's public interface."
    if stem == "__main__":
        return f"Command-line entry point for {module_name}. It starts the package's demonstration or executable workflow."
    return f"Python module {module_name}, responsible for the {stem.replace('_', ' ')} part of the ORE implementation."


class _DefinitionCollector:
    def __init__(self, source_path: str, module_name: str, source_hash: str,
                 source_text: str) -> None:
        self.source_path = source_path
        self.module_name = module_name
        self.source_hash = source_hash
        self.source_text = source_text
        self.rows: list[dict[str, Any]] = []

    def collect(self, tree: ast.Module) -> list[dict[str, Any]]:
        module_doc = _module_annotation(tree, self.module_name, self.source_path)
        self._append("__module__", "module", module_doc, None, tree, "docstring" if _clean_docstring(tree) else "source_analysis")
        self._visit_body(tree.body, parent="")
        return self.rows

    def _append(
        self, qualname: str, kind: str, annotation: str,
        signature: str | None, node: ast.AST, source: str,
    ) -> None:
        source_segment = ast.get_source_segment(self.source_text, node)
        symbol_source_hash = (
            hashlib.sha256(source_segment.encode("utf-8")).hexdigest()
            if source_segment else self.source_hash
        )
        self.rows.append({
            "symbol_id": _symbol_id(self.source_path, kind, qualname),
            "module_name": self.module_name,
            "qualified_name": qualname,
            "symbol_kind": kind,
            "source_path": self.source_path,
            "line_start": getattr(node, "lineno", 1),
            "line_end": getattr(node, "end_lineno", getattr(node, "lineno", 1)),
            "signature": signature,
            "annotation": annotation,
            "source_docstring": _clean_docstring(node) or None,
            "annotation_source": source,
            "source_hash": symbol_source_hash,
        })

    def _visit_body(self, body: list[ast.stmt], parent: str) -> None:
        for node in body:
            self._visit_nested_definitions(node, parent)

    def _visit_nested_definitions(self, node: ast.AST, parent: str) -> None:
        """Find definitions inside control-flow blocks without entering another definition twice."""
        if isinstance(node, ast.ClassDef):
            qualname = f"{parent}.{node.name}" if parent else node.name
            annotation, signature, source = _class_annotation(node, qualname, self.module_name)
            self._append(qualname, "class", annotation, signature, node, source)
            self._visit_body(node.body, qualname)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = f"{parent}.{node.name}" if parent else node.name
            annotation, signature, source = _function_annotation(node, qualname, self.module_name)
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            self._append(qualname, kind, annotation, signature, node, source)
            local_parent = qualname + ".<locals>"
            self._visit_body(node.body, local_parent)
            return
        for child in ast.iter_child_nodes(node):
            self._visit_nested_definitions(child, parent)


def sync_repository(
    database: LoggerDatabase,
    repository_root: str | Path,
    include_paths: set[str] | None = None,
) -> dict[str, int]:
    """Index Python files, optionally restricting work to selected repository-relative paths.

    When include_paths is supplied, only those files/directories are visited and stale
    records are removed only within that same scope. This keeps package-specific indexing
    from scanning or deleting unrelated repository annotations.
    """
    root = Path(repository_root).resolve()
    scopes = {Path(item).as_posix().strip("/") for item in include_paths or set()}

    def in_scope(relative: str) -> bool:
        if not scopes:
            return True
        return any(relative == scope or relative.startswith(scope + "/") for scope in scopes)
    files_seen = 0
    files_indexed = 0
    symbols_written = 0
    parse_errors = 0
    valid_paths: set[str] = set()

    with database.connect() as connection:
        stored = {
            row["source_path"]: (row["mtime_ns"], row["size_bytes"])
            for row in connection.execute("SELECT source_path, mtime_ns, size_bytes FROM source_files")
        }

    for folder, directories, filenames in os.walk(root):
        folder_relative = Path(folder).relative_to(root).as_posix()
        if scopes:
            directories[:] = [
                name for name in directories
                if name not in _SKIP_PARTS
                and not name.startswith(".")
                and in_scope((Path(folder_relative) / name).as_posix())
            ]
        else:
            directories[:] = [
                name for name in directories
                if name not in _SKIP_PARTS and not name.startswith(".")
            ]
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            full_path = Path(folder) / filename
            relative = full_path.relative_to(root).as_posix()
            if not in_scope(relative):
                continue
            valid_paths.add(relative)
            files_seen += 1
            try:
                stat = full_path.stat()
                previous = stored.get(relative)
                if previous == (stat.st_mtime_ns, stat.st_size):
                    continue
                source_bytes = full_path.read_bytes()
                source_hash = hashlib.sha256(source_bytes).hexdigest()
                source_text = source_bytes.decode("utf-8")
                tree = ast.parse(source_text, filename=relative)
                module = _module_name(relative)
                collector = _DefinitionCollector(relative, module, source_hash, source_text)
                rows = collector.collect(tree)
                with database.connect() as connection:
                    for row in rows:
                        connection.execute(
                            """INSERT INTO annotations
                               (symbol_id, module_name, qualified_name, symbol_kind, source_path,
                                line_start, line_end, signature, annotation, source_docstring,
                                annotation_source, source_hash, stale, updated_at_utc)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                               ON CONFLICT(symbol_id) DO UPDATE SET
                                 module_name=excluded.module_name,
                                 qualified_name=excluded.qualified_name,
                                 symbol_kind=excluded.symbol_kind,
                                 source_path=excluded.source_path,
                                 line_start=excluded.line_start,
                                 line_end=excluded.line_end,
                                 signature=excluded.signature,
                                 annotation=CASE
                                   WHEN annotations.annotation_source='manual'
                                   THEN annotations.annotation ELSE excluded.annotation END,
                                 source_docstring=excluded.source_docstring,
                                 annotation_source=CASE
                                   WHEN annotations.annotation_source='manual'
                                   THEN 'manual' ELSE excluded.annotation_source END,
                                 source_hash=excluded.source_hash,
                                 stale=CASE
                                   WHEN annotations.annotation_source='manual'
                                   AND annotations.source_hash IS NOT excluded.source_hash
                                   THEN 1 ELSE 0 END,
                                 updated_at_utc=excluded.updated_at_utc""",
                            (
                                row["symbol_id"], row["module_name"], row["qualified_name"],
                                row["symbol_kind"], row["source_path"], row["line_start"],
                                row["line_end"], row["signature"], row["annotation"],
                                row["source_docstring"], row["annotation_source"],
                                row["source_hash"], utc_now(),
                            ),
                        )
                    ids = [row["symbol_id"] for row in rows]
                    if ids:
                        placeholders = ",".join("?" for _ in ids)
                        connection.execute(
                            f"DELETE FROM annotations WHERE source_path=? AND symbol_id NOT IN ({placeholders})",
                            [relative, *ids],
                        )
                    else:
                        connection.execute(
                            "DELETE FROM annotations WHERE source_path=?", (relative,)
                        )
                    connection.execute(
                        """INSERT INTO source_files
                           (source_path, mtime_ns, size_bytes, sha256, indexed_at_utc, parse_error)
                           VALUES (?, ?, ?, ?, ?, NULL)
                           ON CONFLICT(source_path) DO UPDATE SET
                             mtime_ns=excluded.mtime_ns, size_bytes=excluded.size_bytes,
                             sha256=excluded.sha256, indexed_at_utc=excluded.indexed_at_utc,
                             parse_error=NULL""",
                        (relative, stat.st_mtime_ns, stat.st_size, source_hash, utc_now()),
                    )
                files_indexed += 1
                symbols_written += len(rows)
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                parse_errors += 1
                try:
                    with database.connect() as connection:
                        connection.execute(
                            """INSERT INTO source_files
                               (source_path, mtime_ns, size_bytes, sha256, indexed_at_utc, parse_error)
                               VALUES (?, ?, ?, '', ?, ?)
                               ON CONFLICT(source_path) DO UPDATE SET
                                 indexed_at_utc=excluded.indexed_at_utc,
                                 parse_error=excluded.parse_error""",
                            (relative, 0, 0, utc_now(), f"{type(exc).__name__}: {exc}"),
                        )
                except Exception:
                    pass

    with database.connect() as connection:
        old_paths = {
            row["source_path"] for row in connection.execute("SELECT source_path FROM source_files")
        }
        missing = {
            relative for relative in old_paths - valid_paths
            if in_scope(relative)
        }
        for relative in missing:
            connection.execute("DELETE FROM annotations WHERE source_path=?", (relative,))
            connection.execute("DELETE FROM source_files WHERE source_path=?", (relative,))

    return {
        "files_seen": files_seen,
        "files_indexed": files_indexed,
        "symbols_written": symbols_written,
        "parse_errors": parse_errors,
    }
