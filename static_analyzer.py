"""
Static Analyzer — Phase 2
==========================
Takes a raw diff string and returns a list of structured issues found by
deterministic static analysis (no LLM calls). This runs first in the
pipeline; only issues it can't catch go to the LLM layer.

Three sub-layers:
  1. AST layer    — missing docstrings, naming conventions, long functions
  2. pyflakes     — unused imports, undefined names
  3. radon        — cyclomatic complexity

Output schema (same shape as eval_issues.jsonl so Phase 5 can compare):
  {
    "path":     str,         # file path from the diff
    "line":     int | None,  # line number if known
    "body":     str,         # human-readable issue description
    "category": str,         # bug | style | design | performance | ...
    "source":   str,         # "ast" | "pyflakes" | "radon"
    "severity": str,         # "error" | "warning" | "info"
  }

Usage:
    from static_analyzer import analyze_diff
    issues = analyze_diff(diff_text)
"""

import ast
import re
import subprocess
import sys
import tempfile
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

def parse_diff(diff_text: str) -> list[dict]:
    """
    Parse a unified diff into a list of file chunks.
    Returns: [{"path": str, "added_lines": [(lineno, content), ...]}]

    We only analyze ADDED lines (+) — we don't want to flag issues
    in code that was deleted or unchanged.
    """
    files = []
    current_file = None
    current_lines = []
    current_lineno = 0  # line number in the NEW file

    for line in diff_text.splitlines():
        if line.startswith("diff --git"):
            if current_file:
                files.append({"path": current_file, "added_lines": current_lines})
            current_file = None
            current_lines = []

        elif line.startswith("+++ b/"):
            current_file = line[6:]  # strip "+++ b/"

        elif line.startswith("@@ "):
            # @@ -old_start,old_count +new_start,new_count @@
            match = re.search(r"\+(\d+)", line)
            if match:
                current_lineno = int(match.group(1)) - 1  # will be incremented below

        elif current_file:
            if line.startswith("+") and not line.startswith("+++"):
                current_lineno += 1
                current_lines.append((current_lineno, line[1:]))
            elif not line.startswith("-"):
                current_lineno += 1  # context line, still advances new-file line count

    if current_file:
        files.append({"path": current_file, "added_lines": current_lines})

    return files


def normalize_indent(source: str) -> tuple[str, int]:
    """
    Strip the common leading indentation from all non-empty lines so that
    mid-function diff fragments parse as valid Python.
    Returns (normalized_source, indent_offset) where indent_offset is how
    many spaces were stripped (so line numbers stay meaningful).
    """
    lines = source.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return source, 0
    min_indent = min(len(l) - len(l.lstrip()) for l in non_empty)
    normalized = "\n".join(
        l[min_indent:] if l.strip() else "" for l in lines
    )
    return normalized, min_indent


def is_python_file(path: str) -> bool:
    return path.endswith(".py")


def is_cpp_file(path: str) -> bool:
    return path.endswith((".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx"))


# ---------------------------------------------------------------------------
# Layer 1: AST analysis (Python only)
# ---------------------------------------------------------------------------

SNAKE_CASE_RE = re.compile(r"^[a-z_][a-z0-9_]*$")
UPPER_CASE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PASCAL_CASE_RE = re.compile(r"^[A-Z][a-zA-Z0-9]*$")

# Functions/methods shorter than this are exempt from docstring requirement
MIN_LINES_FOR_DOCSTRING = 5
# Functions longer than this get a complexity warning
MAX_FUNCTION_LINES = 50
# Max McCabe complexity before we flag it
MAX_COMPLEXITY = 10


def has_docstring(node) -> bool:
    """Check if an AST function/class node has a docstring."""
    return (
        isinstance(node.body, list)
        and len(node.body) > 0
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )


def function_line_count(node) -> int:
    """Estimate line count of a function from its AST node."""
    if not node.body:
        return 0
    end = getattr(node, "end_lineno", None)
    if end:
        return end - node.lineno + 1
    return len(node.body)


def ast_issues(path: str, source: str, added_line_nos: set) -> list[dict]:
    """
    Run AST checks on a Python source file, returning issues on added lines only.

    Many diff fragments are mid-function code (e.g. a single elif branch) that
    can't be parsed as standalone Python even after indent normalization. In that
    case we silently skip AST analysis rather than reporting a false syntax error —
    pyflakes and radon will still run on the same source.
    """
    issues = []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Fragment is not self-contained Python — skip AST silently.
        # This is expected for mid-function diffs (elif/except/continuation lines).
        return []

    for node in ast.walk(tree):
        is_test_file = "test" in path.lower()

        # --- Naming conventions ---
        if isinstance(node, ast.FunctionDef):
            if node.lineno not in added_line_nos:
                continue
            name = node.name
            if name.startswith("__") and name.endswith("__"):
                pass  # dunder methods exempt
            elif is_test_file and (name.startswith("test") or name.startswith("assert")):
                pass  # unittest/pytest convention: test_* and assert* in test files are exempt
            elif not SNAKE_CASE_RE.match(name):
                issues.append({
                    "path": path, "line": node.lineno, "source": "ast",
                    "category": "style", "severity": "warning",
                    "body": f"Function '{name}' should use snake_case naming.",
                })

        elif isinstance(node, ast.ClassDef):
            if node.lineno not in added_line_nos:
                continue
            if not PASCAL_CASE_RE.match(node.name):
                issues.append({
                    "path": path, "line": node.lineno, "source": "ast",
                    "category": "style", "severity": "warning",
                    "body": f"Class '{node.name}' should use PascalCase naming.",
                })

        # --- Docstrings on public functions/classes ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno not in added_line_nos:
                continue
            is_private = node.name.startswith("_")
            is_test_method = node.name.startswith("test") or node.name.startswith("assert")
            line_count = function_line_count(node)
            if (
                not is_private
                and not (is_test_file and is_test_method)  # test methods don't need docstrings
                and line_count >= MIN_LINES_FOR_DOCSTRING
                and not has_docstring(node)
            ):
                issues.append({
                    "path": path, "line": node.lineno, "source": "ast",
                    "category": "documentation", "severity": "info",
                    "body": f"Public function '{node.name}' ({line_count} lines) is missing a docstring.",
                })

        # --- Long functions ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno not in added_line_nos:
                continue
            line_count = function_line_count(node)
            if line_count > MAX_FUNCTION_LINES:
                issues.append({
                    "path": path, "line": node.lineno, "source": "ast",
                    "category": "design", "severity": "warning",
                    "body": f"Function '{node.name}' is {line_count} lines long. "
                            f"Consider breaking it up (threshold: {MAX_FUNCTION_LINES}).",
                })

        # --- Mutable default arguments ---
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno not in added_line_nos:
                continue
            for default in node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append({
                        "path": path, "line": node.lineno, "source": "ast",
                        "category": "bug", "severity": "error",
                        "body": f"Function '{node.name}' uses a mutable default argument "
                                f"({type(default).__name__}). This is a common Python bug — "
                                f"the default is shared across all calls.",
                    })

        # --- Bare except ---
        if isinstance(node, ast.ExceptHandler):
            if node.lineno not in added_line_nos:
                continue
            if node.type is None:
                issues.append({
                    "path": path, "line": node.lineno, "source": "ast",
                    "category": "bug", "severity": "warning",
                    "body": "Bare `except:` clause catches all exceptions including "
                            "KeyboardInterrupt and SystemExit. Use `except Exception:` instead.",
                })

    return issues


# ---------------------------------------------------------------------------
# Layer 2: pyflakes (unused imports, undefined names)
# ---------------------------------------------------------------------------

def pyflakes_issues(path: str, source: str, added_line_nos: set) -> list[dict]:
    """Run pyflakes on source, return issues on added lines only."""
    try:
        from pyflakes import api as pyflakes_api
        from pyflakes import reporter as pyflakes_reporter
        import io
    except ImportError:
        return []  # pyflakes not installed, skip silently

    warning_stream = io.StringIO()
    error_stream = io.StringIO()

    class StringReporter(pyflakes_reporter.Reporter):
        def unexpectedError(self, filename, msg):
            error_stream.write(f"{filename}: {msg}\n")
        def syntaxError(self, filename, msg, lineno, offset, text):
            warning_stream.write(f"{filename}:{lineno}: syntax error: {msg}\n")
        def flake(self, message):
            warning_stream.write(str(message) + "\n")

    pyflakes_api.check(source, path, reporter=StringReporter(warning_stream, error_stream))

    issues = []
    # Parse output: "path:line: message"
    for line in warning_stream.getvalue().splitlines():
        match = re.match(r".+:(\d+):\d+ (.+)", line)
        if not match:
            match = re.match(r".+:(\d+): (.+)", line)
        if not match:
            continue
        lineno = int(match.group(1))
        msg = match.group(2).strip()
        if lineno not in added_line_nos:
            continue

        # Categorize based on message content
        if "imported but unused" in msg or "redefinition of unused" in msg:
            category, severity = "style", "warning"
        elif "undefined name" in msg:
            category, severity = "bug", "error"
        elif "local variable" in msg and "referenced before assignment" in msg:
            category, severity = "bug", "error"
        else:
            category, severity = "style", "info"

        issues.append({
            "path": path, "line": lineno, "source": "pyflakes",
            "category": category, "severity": severity,
            "body": msg,
        })

    return issues


# ---------------------------------------------------------------------------
# Layer 3: radon (cyclomatic complexity, Python only)
# ---------------------------------------------------------------------------

def radon_issues(path: str, source: str, added_line_nos: set) -> list[dict]:
    """Run radon complexity check, flag functions above threshold."""
    try:
        from radon.complexity import cc_visit
        from radon.complexity import SCORE
    except ImportError:
        return []

    issues = []
    try:
        results = cc_visit(source)
    except Exception:
        return []

    for block in results:
        if block.lineno not in added_line_nos:
            continue
        if block.complexity > MAX_COMPLEXITY:
            issues.append({
                "path": path, "line": block.lineno, "source": "radon",
                "category": "performance", "severity": "warning",
                "body": f"'{block.name}' has cyclomatic complexity {block.complexity} "
                        f"(threshold: {MAX_COMPLEXITY}). High complexity makes code harder "
                        f"to test and reason about.",
            })

    return issues


# ---------------------------------------------------------------------------
# C++ basic checks (regex-based, no full parser)
# ---------------------------------------------------------------------------

CPP_CHECKS = [
    (
        re.compile(r"\bmalloc\s*\("),
        "style", "info",
        "Prefer `new` or smart pointers (std::unique_ptr / std::shared_ptr) over malloc() in C++.",
    ),
    (
        re.compile(r"\bprintf\s*\("),
        "style", "info",
        "Prefer std::cout or fmt:: over printf() in modern C++.",
    ),
    (
        re.compile(r"\busing namespace std\b"),
        "style", "warning",
        "`using namespace std;` in a header pollutes the global namespace for all includers.",
    ),
    (
        re.compile(r"\bNULL\b"),
        "style", "info",
        "Prefer `nullptr` over NULL in C++11 and later.",
    ),
    (
        re.compile(r"catch\s*\(\s*\.\.\.\s*\)"),
        "bug", "warning",
        "Bare catch(...) swallows all exceptions including system signals. "
        "Catch specific exception types where possible.",
    ),
    (
        re.compile(r"\bdelete\b(?!\[\])"),
        "bug", "info",
        "Manual `delete` detected. Prefer RAII / smart pointers to avoid memory leaks.",
    ),
]


def cpp_issues(path: str, added_lines: list[tuple]) -> list[dict]:
    """Regex-based checks on C++ added lines."""
    issues = []
    for lineno, content in added_lines:
        for pattern, category, severity, body in CPP_CHECKS:
            if pattern.search(content):
                issues.append({
                    "path": path, "line": lineno, "source": "ast",
                    "category": category, "severity": severity,
                    "body": body,
                })
    return issues


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_diff(diff_text: str) -> list[dict]:
    """
    Run all static analysis layers on a diff.
    Returns a flat list of issue dicts, sorted by (path, line).
    """
    all_issues = []
    file_chunks = parse_diff(diff_text)

    for chunk in file_chunks:
        path = chunk["path"]
        added_lines = chunk["added_lines"]
        added_line_nos = {ln for ln, _ in added_lines}
        source_code = "\n".join(content for _, content in added_lines)

        if is_python_file(path):
            source_code = "\n".join(content for _, content in added_lines)
            normalized, _ = normalize_indent(source_code)
            all_issues.extend(ast_issues(path, normalized, added_line_nos))
            all_issues.extend(pyflakes_issues(path, normalized, added_line_nos))
            all_issues.extend(radon_issues(path, normalized, added_line_nos))

        elif is_cpp_file(path):
            all_issues.extend(cpp_issues(path, added_lines))

    # Sort by file then line number
    all_issues.sort(key=lambda x: (x["path"], x["line"] or 0))
    return all_issues


# ---------------------------------------------------------------------------
# CLI: run directly on a diff file for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    # Colab passes internal flags like '-f' to sys.argv — skip CLI if that's the case
    args = [a for a in sys.argv[1:] if not a.startswith("-f") and not a.endswith(".json")]
    if not args:
        # Running inside Colab or with no args — don't execute CLI block
        pass
    elif args[0] == "-":
        diff = sys.stdin.read()
        issues = analyze_diff(diff)
        if not issues:
            print("No issues found.")
        else:
            print(f"Found {len(issues)} issue(s):\n")
            for issue in issues:
                sev = issue["severity"].upper()
                print(f"[{sev:7s}] [{issue['category']:12s}] {issue['path']}:{issue['line']}")
                print(f"          {issue['body']}")
                print(f"          (source: {issue['source']})")
                print()
    else:
        diff = open(args[0]).read()
        issues = analyze_diff(diff)
        if not issues:
            print("No issues found.")
        else:
            print(f"Found {len(issues)} issue(s):\n")
            for issue in issues:
                sev = issue["severity"].upper()
                print(f"[{sev:7s}] [{issue['category']:12s}] {issue['path']}:{issue['line']}")
                print(f"          {issue['body']}")
                print(f"          (source: {issue['source']})")
                print()
