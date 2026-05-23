from __future__ import annotations

"""
AST Parser — extracts structural information from Python source files.

Uses ONLY the stdlib ast module. Zero regex.

Extracts:
- Module-level imports (import X, from X import Y)
- Top-level functions (sync and async)
- Classes with inheritance, docstrings, and all methods
- Function calls within function bodies
- Decorators, arguments, return annotations, docstrings
- Exact line numbers for every symbol

Call extraction handles:
- Simple calls: foo()
- Attribute calls: obj.method()
- Chained calls: a.b.c()
"""

import ast
import logging
from pathlib import Path
from typing import Optional

from app.parsing.scanner import file_to_module_name
from app.schemas.models import (
    ParsedCall,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_attr_name(node: ast.expr) -> str:
    """
    Recursively resolve an AST expression to a dotted name string.

    ast.Name("foo")               → "foo"
    ast.Attribute(ast.Name("a"), "b") → "a.b"
    ast.Attribute(ast.Attribute(...), "c") → "a.b.c"
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_attr_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _get_decorator_name(node: ast.expr) -> str:
    """Stringify a decorator node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _resolve_attr_name(node)
    if isinstance(node, ast.Call):
        return _get_decorator_name(node.func)
    return ast.unparse(node)


def _get_annotation(node: Optional[ast.expr]) -> Optional[str]:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def _extract_calls(func_body: list[ast.stmt]) -> list[ParsedCall]:
    """
    Walk every node in a function body and extract all ast.Call nodes.
    Returns a list of ParsedCall with the raw call name and line number.
    """
    calls: list[ParsedCall] = []
    seen: set[tuple[str, int]] = set()

    for node in ast.walk(ast.Module(body=func_body, type_ignores=[])):
        if isinstance(node, ast.Call):
            name = _resolve_attr_name(node.func)
            if name:
                line = getattr(node, "lineno", 0)
                key = (name, line)
                if key not in seen:
                    seen.add(key)
                    calls.append(ParsedCall(name=name, line=line))

    return calls


def _extract_args(args: ast.arguments) -> list[str]:
    """Extract argument names from an ast.arguments node."""
    result: list[str] = []
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        result.append(arg.arg)
    if args.vararg:
        result.append(f"*{args.vararg.arg}")
    if args.kwarg:
        result.append(f"**{args.kwarg.arg}")
    return result


def _get_source_segment(source_lines: list[str], start: int, end: int) -> str:
    """
    Extract source lines [start, end] (1-indexed, inclusive).
    Falls back to empty string if line numbers are out of range.
    """
    if start < 1 or end < start:
        return ""
    slice_lines = source_lines[start - 1 : end]
    return "".join(slice_lines)


def _last_line_of(node: ast.AST) -> int:
    """
    Walk an AST node to find its last line number.
    ast.end_lineno is available on Python 3.8+.
    """
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


def _extract_local_assignments(nodes: list[ast.stmt]) -> dict[str, str]:
    """
    Search for assignments in a list of statements:
        var = ClassName(...)
        var = ClassName
    Returns a mapping of variable name to class name.
    """
    assignments = {}
    for node in ast.walk(ast.Module(body=nodes, type_ignores=[])):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                val = node.value
                if isinstance(val, ast.Call):
                    class_name = _resolve_attr_name(val.func)
                    if class_name:
                        assignments[var_name] = class_name
                elif isinstance(val, (ast.Name, ast.Attribute)):
                    class_name = _resolve_attr_name(val)
                    if class_name:
                        assignments[var_name] = class_name
    return assignments


def _extract_class_assignments(class_node: ast.ClassDef) -> dict[str, str]:
    """
    Search for attribute assignments on 'self' or at class level:
        self.attr = ClassName(...)
        attr = ClassName(...)
    Returns a mapping of attr_name -> ClassName.
    """
    assignments = {}
    for child in ast.walk(class_node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                    attr_name = f"self.{target.attr}"
                    val = child.value
                    if isinstance(val, ast.Call):
                        class_name = _resolve_attr_name(val.func)
                        if class_name:
                            assignments[attr_name] = class_name
                    elif isinstance(val, (ast.Name, ast.Attribute)):
                        class_name = _resolve_attr_name(val)
                        if class_name:
                            assignments[attr_name] = class_name
                elif isinstance(target, ast.Name):
                    attr_name = target.id
                    val = child.value
                    if isinstance(val, ast.Call):
                        class_name = _resolve_attr_name(val.func)
                        if class_name:
                            assignments[attr_name] = class_name
                    elif isinstance(val, (ast.Name, ast.Attribute)):
                        class_name = _resolve_attr_name(val)
                        if class_name:
                            assignments[attr_name] = class_name
    return assignments


# ---------------------------------------------------------------------------
# Function extraction
# ---------------------------------------------------------------------------



def _extract_function(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    source_lines: list[str],
    module_name: str,
    file_path: str,
    class_name: Optional[str] = None,
) -> ParsedFunction:
    """Extract a ParsedFunction from an ast.FunctionDef or AsyncFunctionDef node."""

    name = node.name
    is_method = class_name is not None

    if is_method:
        qualified_name = f"{module_name}.{class_name}.{name}"
    else:
        qualified_name = f"{module_name}.{name}"

    line_start = node.lineno
    line_end = _last_line_of(node)

    source = _get_source_segment(source_lines, line_start, line_end)
    docstring = ast.get_docstring(node)
    args = _extract_args(node.args)
    return_annotation = _get_annotation(node.returns)
    decorators = [_get_decorator_name(d) for d in node.decorator_list]
    calls = _extract_calls(node.body)
    is_async = isinstance(node, ast.AsyncFunctionDef)

    return ParsedFunction(
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source,
        docstring=docstring,
        args=args,
        return_annotation=return_annotation,
        decorators=decorators,
        calls=calls,
        is_async=is_async,
        is_method=is_method,
        class_name=class_name,
        assignments=_extract_local_assignments(node.body),
    )



# ---------------------------------------------------------------------------
# Class extraction
# ---------------------------------------------------------------------------


def _extract_class(
    node: ast.ClassDef,
    source_lines: list[str],
    module_name: str,
    file_path: str,
) -> ParsedClass:
    """Extract a ParsedClass including all its methods."""

    name = node.name
    qualified_name = f"{module_name}.{name}"
    line_start = node.lineno
    line_end = _last_line_of(node)
    source = _get_source_segment(source_lines, line_start, line_end)
    docstring = ast.get_docstring(node)
    decorators = [_get_decorator_name(d) for d in node.decorator_list]

    # Resolve base class names
    bases: list[str] = []
    for base in node.bases:
        base_name = _resolve_attr_name(base)
        if not base_name:
            try:
                base_name = ast.unparse(base)
            except Exception:
                base_name = ""
        if base_name:
            bases.append(base_name)

    # Extract methods (direct children only — not nested class methods)
    methods: list[ParsedFunction] = []
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            method = _extract_function(
                child,
                source_lines,
                module_name,
                file_path,
                class_name=name,
            )
            methods.append(method)

    return ParsedClass(
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source,
        docstring=docstring,
        bases=bases,
        methods=methods,
        decorators=decorators,
        assignments=_extract_class_assignments(node),
    )



# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------


def _extract_imports(tree: ast.Module) -> list[ParsedImport]:
    """Extract all import statements from the module-level AST."""
    imports: list[ParsedImport] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ParsedImport(
                        module=alias.name,
                        names=[],
                        alias=alias.asname,
                        line=node.lineno,
                        is_from=False,
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = [alias.name for alias in node.names]
            # Handle aliased names: from X import Y as Z — store all names,
            # alias field stores the last alias (useful for single-name imports)
            alias = node.names[0].asname if len(node.names) == 1 else None
            imports.append(
                ParsedImport(
                    module=module,
                    names=names,
                    alias=alias,
                    line=node.lineno,
                    is_from=True,
                )
            )

    return imports


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def parse_file(file_path: Path, repo_root: Path) -> ParsedFile:
    """
    Parse a single Python file using the ast module.

    Args:
        file_path: Absolute path to the .py file.
        repo_root: Repository root (used to compute module_name).

    Returns:
        ParsedFile with all extracted symbols, or ParsedFile with parse_error set.
    """
    path_str = str(file_path)
    module_name = file_to_module_name(file_path, repo_root)

    try:
        source_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.warning("Cannot read file %s: %s", path_str, exc)
        return ParsedFile(
            path=path_str,
            module_name=module_name,
            parse_error=str(exc),
        )

    try:
        tree = ast.parse(source_text, filename=path_str)
    except SyntaxError as exc:
        logger.warning("Syntax error in %s: %s", path_str, exc)
        return ParsedFile(
            path=path_str,
            module_name=module_name,
            parse_error=str(exc),
        )

    source_lines = source_text.splitlines(keepends=True)

    imports = _extract_imports(tree)

    # Top-level functions (not inside a class)
    top_level_functions: list[ParsedFunction] = []
    top_level_classes: list[ParsedClass] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fn = _extract_function(node, source_lines, module_name, path_str)
            top_level_functions.append(fn)

        elif isinstance(node, ast.ClassDef):
            cls = _extract_class(node, source_lines, module_name, path_str)
            top_level_classes.append(cls)

    logger.debug(
        "Parsed %s → %d functions, %d classes, %d imports",
        module_name,
        len(top_level_functions),
        len(top_level_classes),
        len(imports),
    )

    return ParsedFile(
        path=path_str,
        module_name=module_name,
        imports=imports,
        functions=top_level_functions,
        classes=top_level_classes,
    )
