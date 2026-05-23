from __future__ import annotations

"""
Symbol Registry — cross-file symbol resolution for the call graph.

The symbol registry solves the fundamental problem of resolving raw call
names (as they appear in Python source) to fully-qualified symbol names.

Problem:
    File: auth/routes.py
    Code:
        from auth.jwt import validate_token
        ...
        validate_token(token)

    The AST call node gives us the name "validate_token".
    We need to know this maps to "auth.jwt.validate_token".

Solution (3-pass algorithm):
    Pass 1 — Index definitions:
        Build a global table of every defined function/class by qualified name.

    Pass 2 — Build per-file import maps:
        For each file, analyse its import statements to build a mapping of
        local_name → qualified_name for everything imported.

        Examples:
            import auth.jwt
              → "jwt" (last component) maps to module "auth.jwt"
              → "auth.jwt" maps to module "auth.jwt"

            from auth.jwt import validate_token
              → "validate_token" maps to "auth.jwt.validate_token"

            from auth.jwt import validate_token as vt
              → "vt" maps to "auth.jwt.validate_token"

    Pass 3 — Resolve calls:
        For each function's ParsedCall, look up the raw call name in the
        file's import map, then check the global definition table.
        Handles: "foo", "module.foo", "alias.foo".
"""

import logging
from pathlib import Path
from typing import Optional

from app.schemas.models import (
    ImportResolution,
    ParsedCall,
    ParsedFile,
    ParsedFunction,
    ParsedClass,
    SymbolRegistryEntry,
    SymbolType,
)

logger = logging.getLogger(__name__)


class SymbolRegistry:
    """
    Global symbol resolution registry for a single indexed repository.

    Attributes:
        _symbols: qualified_name → SymbolRegistryEntry for all defined symbols
        _name_index: simple_name → list[qualified_name] for partial lookup
        _file_import_maps: file_path → {local_name: qualified_name}
    """

    def __init__(self) -> None:
        self._symbols: dict[str, SymbolRegistryEntry] = {}
        # simple name → list of qualified names (for find_by_name)
        self._name_index: dict[str, list[str]] = {}
        # per-file: local name → resolved qualified name
        self._file_import_maps: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, parsed_files: list[ParsedFile]) -> None:
        """
        Execute all 3 passes to build the complete symbol registry.

        Args:
            parsed_files: All ParsedFile results from the AST parser.
        """
        self._pass1_index_definitions(parsed_files)
        self._pass2_build_import_maps(parsed_files)
        self._pass3_resolve_calls(parsed_files)

        logger.info(
            "Symbol registry built — %d symbols, %d files indexed",
            len(self._symbols),
            len(self._file_import_maps),
        )

    def _pass1_index_definitions(self, parsed_files: list[ParsedFile]) -> None:
        """Register every defined function, method, and class."""
        for pf in parsed_files:
            if pf.parse_error:
                continue

            # Top-level functions
            for fn in pf.functions:
                self._register(
                    SymbolRegistryEntry(
                        qualified_name=fn.qualified_name,
                        symbol_type=SymbolType.FUNCTION,
                        file_path=pf.path,
                        class_name=None,
                        line_start=fn.line_start,
                    )
                )

            # Classes and their methods
            for cls in pf.classes:
                self._register(
                    SymbolRegistryEntry(
                        qualified_name=cls.qualified_name,
                        symbol_type=SymbolType.CLASS,
                        file_path=pf.path,
                        class_name=None,
                        line_start=cls.line_start,
                    )
                )
                for method in cls.methods:
                    self._register(
                        SymbolRegistryEntry(
                            qualified_name=method.qualified_name,
                            symbol_type=SymbolType.METHOD,
                            file_path=pf.path,
                            class_name=cls.name,
                            line_start=method.line_start,
                        )
                    )

        logger.debug("Pass 1 — indexed %d symbols", len(self._symbols))

    def _pass2_build_import_maps(self, parsed_files: list[ParsedFile]) -> None:
        """Build per-file import resolution maps."""
        for pf in parsed_files:
            if pf.parse_error:
                continue

            import_map: dict[str, str] = {}

            for imp in pf.imports:
                if imp.is_from:
                    # from auth.jwt import validate_token [as vt]
                    # from auth.jwt import A, B, C
                    if imp.names:
                        for name in imp.names:
                            # Find the alias for this specific name
                            # (ParsedImport.alias only stores alias for single-name imports)
                            candidate = f"{imp.module}.{name}"
                            if len(imp.names) == 1 and imp.alias:
                                # "from X import Y as alias" — alias maps to X.Y
                                import_map[imp.alias] = candidate
                            else:
                                import_map[name] = candidate
                else:
                    # import auth.jwt [as jwt]
                    # import os, sys
                    local_name = imp.alias if imp.alias else imp.module
                    import_map[local_name] = imp.module

                    # Also allow the last component as a shorthand
                    # e.g. "import auth.jwt" → also map "jwt" → "auth.jwt"
                    parts = imp.module.split(".")
                    if len(parts) > 1 and not imp.alias:
                        import_map[parts[-1]] = imp.module

            self._file_import_maps[pf.path] = import_map

        logger.debug("Pass 2 — built import maps for %d files", len(self._file_import_maps))

    def _pass3_resolve_calls(self, parsed_files: list[ParsedFile]) -> None:
        """Resolve each call in every function to its qualified name."""
        from app.schemas.models import ResolutionType
        resolved_count = 0
        unresolved_count = 0

        for pf in parsed_files:
            if pf.parse_error:
                continue

            import_map = self._file_import_maps.get(pf.path, {})

            # 1. Resolve top-level functions (without class context)
            for fn in pf.functions:
                for call in fn.calls:
                    resolved, res_type = self._resolve_call_name(
                        call.name,
                        pf.module_name,
                        import_map,
                        fn_assignments=fn.assignments,
                    )
                    if resolved:
                        call.resolved = resolved
                        call.resolution_type = res_type
                        resolved_count += 1
                    else:
                        call.resolution_type = ResolutionType.UNRESOLVED
                        unresolved_count += 1

            # 2. Resolve class methods (with class and self assignments)
            for cls in pf.classes:
                class_assignments = cls.assignments
                for fn in cls.methods:
                    for call in fn.calls:
                        resolved, res_type = self._resolve_call_name(
                            call.name,
                            pf.module_name,
                            import_map,
                            fn_assignments=fn.assignments,
                            class_assignments=class_assignments,
                            calling_class_qn=cls.qualified_name,
                        )
                        if resolved:
                            call.resolved = resolved
                            call.resolution_type = res_type
                            resolved_count += 1
                        else:
                            call.resolution_type = ResolutionType.UNRESOLVED
                            unresolved_count += 1

        logger.debug(
            "Pass 3 — resolved %d calls, %d unresolved (builtins/dynamic)",
            resolved_count,
            unresolved_count,
        )

    def _resolve_call_name(
        self,
        raw_name: str,
        calling_module: str,
        import_map: dict[str, str],
        fn_assignments: dict[str, str] = None,
        class_assignments: dict[str, str] = None,
        calling_class_qn: Optional[str] = None,
    ) -> tuple[Optional[str], ResolutionType]:
        """
        Resolve a raw call name to a fully-qualified symbol name.
        Returns (resolved_name, resolution_type).
        """
        from app.schemas.models import ResolutionType
        if not raw_name:
            return None, ResolutionType.UNRESOLVED

        # --- 1. Direct import map ---
        if raw_name in import_map:
            candidate = import_map[raw_name]
            return candidate, ResolutionType.IMPORT

        # --- 2. Attribute & Method Resolution Heuristics ---
        if "." in raw_name:
            parts = raw_name.rsplit(".", 1)
            prefix = parts[0]
            method = parts[1]

            # 2a. Calls to self: self.my_method()
            if prefix == "self" and calling_class_qn:
                candidate = f"{calling_class_qn}.{method}"
                if candidate in self._symbols:
                    return candidate, ResolutionType.DIRECT

            # 2b. Calls to self attribute: self.model.forward()
            elif prefix.startswith("self.") and class_assignments and prefix in class_assignments:
                raw_class = class_assignments[prefix]
                resolved_class, _ = self._resolve_call_name(raw_class, calling_module, import_map)
                if resolved_class:
                    candidate = f"{resolved_class}.{method}"
                    return candidate, ResolutionType.HEURISTIC

            # 2c. Calls to local variables: trainer.fit()
            elif fn_assignments and prefix in fn_assignments:
                raw_class = fn_assignments[prefix]
                resolved_class, _ = self._resolve_call_name(raw_class, calling_module, import_map)
                if resolved_class:
                    candidate = f"{resolved_class}.{method}"
                    return candidate, ResolutionType.HEURISTIC

        # --- 3. Dotted prefix (e.g. "jwt.decode") ---
        if "." in raw_name:
            parts = raw_name.split(".", 1)
            prefix = parts[0]
            rest = parts[1]

            if prefix in import_map:
                resolved_prefix = import_map[prefix]
                candidate = f"{resolved_prefix}.{rest}"
                if candidate in self._symbols:
                    return candidate, ResolutionType.IMPORT
                matches = self._suffix_match(raw_name)
                if matches:
                    return matches[0], ResolutionType.HEURISTIC
                return candidate, ResolutionType.IMPORT

        # --- 4. Same-module (called without import, defined in same file) ---
        same_module_candidate = f"{calling_module}.{raw_name}"
        if same_module_candidate in self._symbols:
            return same_module_candidate, ResolutionType.DIRECT

        # --- 5. Exact global lookup ---
        if raw_name in self._symbols:
            return raw_name, ResolutionType.IMPORT

        # --- 6. Suffix match in symbol table ---
        suffix_matches = self._suffix_match(raw_name)
        if suffix_matches:
            return suffix_matches[0], ResolutionType.HEURISTIC

        return None, ResolutionType.UNRESOLVED

    def _find_in_symbols_by_module(self, module: str) -> bool:
        """Check if any symbol's qualified name starts with the given module."""
        prefix = module + "."
        return any(k.startswith(prefix) for k in self._symbols)

    def _suffix_match(self, name: str) -> list[str]:
        """
        Find symbols whose qualified name ends with the given name fragment.
        Used as a last-resort resolution strategy.
        """
        suffix = f".{name}"
        return [qn for qn in self._symbols if qn.endswith(suffix) or qn == name]

    def _register(self, entry: SymbolRegistryEntry) -> None:
        """Add a symbol to the registry and update the name index."""
        self._symbols[entry.qualified_name] = entry

        # Extract simple name (last component)
        simple_name = entry.qualified_name.split(".")[-1]
        if simple_name not in self._name_index:
            self._name_index[simple_name] = []
        if entry.qualified_name not in self._name_index[simple_name]:
            self._name_index[simple_name].append(entry.qualified_name)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def resolve_call(self, file_path: str, call_name: str) -> Optional[str]:
        """
        Resolve a call name using the import map for a specific file.

        Args:
            file_path: The file where the call occurs.
            call_name: The raw call name as seen in AST.

        Returns:
            Fully-qualified name, or None if unresolvable.
        """
        import_map = self._file_import_maps.get(file_path, {})
        # We need the module name for same-module resolution
        # Find it from any symbol in this file
        calling_module = self._get_module_for_file(file_path)
        resolved, _ = self._resolve_call_name(call_name, calling_module, import_map)
        return resolved

    def get_symbol(self, qualified_name: str) -> Optional[SymbolRegistryEntry]:
        """Look up a symbol by its exact qualified name."""
        return self._symbols.get(qualified_name)

    def find_by_name(self, name: str) -> list[SymbolRegistryEntry]:
        """
        Find all symbols matching a simple (unqualified) name.

        Args:
            name: Simple name like "validate_token" (not the full qualified name).

        Returns:
            List of matching SymbolRegistryEntry objects.
        """
        qualified_names = self._name_index.get(name, [])
        return [self._symbols[qn] for qn in qualified_names if qn in self._symbols]

    def find_by_name_fragment(self, fragment: str) -> list[SymbolRegistryEntry]:
        """
        Find symbols whose name contains the given fragment (case-insensitive).
        Used for fuzzy lookup in queries.
        """
        fragment_lower = fragment.lower()
        results: list[SymbolRegistryEntry] = []
        for qn, entry in self._symbols.items():
            simple_name = qn.split(".")[-1].lower()
            if fragment_lower in simple_name or fragment_lower in qn.lower():
                results.append(entry)
        return results

    def get_all_symbols(self) -> dict[str, SymbolRegistryEntry]:
        """Return the complete symbol table."""
        return dict(self._symbols)

    def get_file_import_map(self, file_path: str) -> dict[str, str]:
        """Return the import resolution map for a specific file."""
        return dict(self._file_import_maps.get(file_path, {}))

    def _get_module_for_file(self, file_path: str) -> str:
        """Infer the module name for a file by inspecting its symbols."""
        for qn, entry in self._symbols.items():
            if entry.file_path == file_path:
                # module is everything except the last component
                parts = qn.split(".")
                if len(parts) > 1:
                    return ".".join(parts[:-1])
                return qn
        return ""

    def to_dict(self) -> dict:
        """Serialize registry to a JSON-compatible dict (for persistence)."""
        return {
            "symbols": {k: v.model_dump() for k, v in self._symbols.items()},
            "name_index": self._name_index,
            "file_import_maps": self._file_import_maps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SymbolRegistry":
        """Restore a SymbolRegistry from a serialized dict."""
        registry = cls()
        for qn, entry_data in data.get("symbols", {}).items():
            registry._symbols[qn] = SymbolRegistryEntry(**entry_data)
        registry._name_index = data.get("name_index", {})
        registry._file_import_maps = data.get("file_import_maps", {})
        return registry


def build_registry(parsed_files: list[ParsedFile]) -> SymbolRegistry:
    """Convenience function: build and return a SymbolRegistry from parsed files."""
    registry = SymbolRegistry()
    registry.build(parsed_files)
    return registry
