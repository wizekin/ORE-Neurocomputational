"""Least-privilege sandbox for generated code (Point 14, Step 26).

Generated text, code, data, and hypotheses enter the workspace as untrusted
candidates; generated code executes only in this sandbox.  Two layers of defence:

1. a static AST screen for forbidden constructs (file/network/process/eval/import
   of disallowed modules), and
2. runtime execution with a restricted ``__builtins__`` and a guarded import hook
   that only admits the configured module allowlist.

This is a demonstration-grade containment for the reference implementation; real
deployments add OS-level isolation (Section 9: resource exhaustion, supply chain).
"""
from __future__ import annotations

import ast
import io
import contextlib
from typing import Any, Mapping

from ..config import LearningConfig

__all__ = ("SandboxViolation", "SandboxResult", "LeastPrivilegeSandbox",
           "STATIC_FORBIDDEN_NAMES")


STATIC_FORBIDDEN_NAMES: frozenset[str] = frozenset({
    "open", "eval", "exec", "compile", "__import__", "input", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
})

SAFE_BUILTINS: frozenset[str] = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float",
    "int", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed",
    "round", "set", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
})


class SandboxViolation(ValueError):
    """Raised when generated code attempts a forbidden operation."""


class SandboxResult:
    """The outcome of one sandboxed execution."""

    def __init__(self, status: str, stdout: str = "", result: Any = None,
                 violations: list[str] | None = None) -> None:
        """Store the outcome."""
        self.status = status  # executed | rejected | error
        self.stdout = stdout
        self.result = result
        self.violations = list(violations or [])

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"status": self.status, "stdout": self.stdout,
                "result": self.result if isinstance(
                    (self.result), (int, float, str, bool, list, dict, tuple)) else None,
                "violations": list(self.violations),
                "note": "generated code executed only inside the least-privilege sandbox"}


class LeastPrivilegeSandbox:
    """Step 26 -- execute generated code with least privilege and full audit."""

    purpose = "generated code executes only here, never on the host"

    def __init__(self, config: LearningConfig | None = None,
                 allowed_modules: tuple[str, ...] | None = None) -> None:
        """Configure the module allowlist."""
        self.config = config or LearningConfig()
        self.allowed_modules = frozenset(
            allowed_modules if allowed_modules is not None
            else self.config.sandbox_allowed_modules)
        self.executions: list[SandboxResult] = []

    def static_check(self, code_text: str) -> list[str]:
        """Layer 1: AST screen for forbidden constructs and disallowed imports."""
        violations: list[str] = []
        try:
            tree = ast.parse(code_text)
        except SyntaxError as exc:
            return [f"syntax-error: {exc.msg}"]
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in STATIC_FORBIDDEN_NAMES:
                violations.append(f"forbidden name: {node.id}")
            if isinstance(node, ast.Attribute):
                name = node.attr
                if name in ("system", "popen", "spawn", "fork", "connect", "urlopen",
                            "load", "loads_from_file", "write_file"):
                    violations.append(f"forbidden attribute access: {name}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in self.allowed_modules:
                        violations.append(f"disallowed import: {alias.name}")
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if node.module and root not in self.allowed_modules:
                    violations.append(f"disallowed import-from: {node.module}")
        return violations

    def execute(self, code_text: str, *, entry_variable: str = "result") -> SandboxResult:
        """Run generated code under both defence layers; never on the host.

        Returns a :class:`SandboxResult`; rejections carry the violation list so the
        caller can record an auditable event instead of silently swallowing it.
        """
        violations = self.static_check(code_text)
        if violations:
            outcome = SandboxResult("rejected", violations=violations)
            self.executions.append(outcome)
            return outcome

        buffer = io.StringIO()
        environment: dict[str, Any] = {"__builtins__": self._safe_builtins(buffer)}
        try:
            with contextlib.redirect_stdout(buffer):
                # pylint: disable=exec-used -- the entire point of the sandbox
                exec(compile(code_text, "<sandbox>", "exec"), environment, environment)  # noqa: S102
        except SandboxViolation as exc:
            outcome = SandboxResult("rejected", stdout=buffer.getvalue(),
                                    violations=[str(exc)])
        except Exception as exc:  # noqa: BLE001 - sandbox isolates failures
            outcome = SandboxResult("error", stdout=buffer.getvalue(),
                                    violations=[f"runtime-error: {type(exc).__name__}: "
                                               f"{exc}"])
        else:
            outcome = SandboxResult("executed", stdout=buffer.getvalue(),
                                    result=environment.get(entry_variable))
        self.executions.append(outcome)
        return outcome

    def _safe_builtins(self, buffer: io.StringIO) -> dict[str, Any]:
        """Build the restricted builtins mapping with a guarded import hook."""
        import builtins as _builtins

        def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
            root = name.split(".")[0]
            if root not in self.allowed_modules:
                raise SandboxViolation(
                    f"import of {name!r} is outside the sandbox allowlist")
            return __import__(name, *args, **kwargs)  # noqa: PLC0414

        safe: dict[str, Any] = {name: getattr(_builtins, name) for name in SAFE_BUILTINS
                                if name not in ("True", "False", "None")}
        safe["True"], safe["False"], safe["None"] = True, False, None
        safe["__import__"] = guarded_import
        safe["__name__"] = "__sandbox__"
        safe["__build_class__"] = _builtins.__build_class__  # class definitions allowed
        return safe

    def describe(self) -> dict[str, Any]:
        """Summarize sandbox configuration and activity."""
        return {"allowed_modules": sorted(self.allowed_modules),
                "forbidden_names": sorted(STATIC_FORBIDDEN_NAMES),
                "executions": len(self.executions),
                "rejected": sum(1 for item in self.executions
                                if item.status == "rejected"),
                "purpose": self.purpose}
