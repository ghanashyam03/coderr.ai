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
                        bases=cls.bases,
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
                    resolved, res_type, confidence, evidence, provenance = self._resolve_call_name(
                        call.name,
                        pf.module_name,
                        import_map,
                        fn_assignments=fn.assignments,
                    )
                    call.resolved = resolved
                    call.resolution_type = res_type
                    call.confidence = confidence
                    call.evidence = evidence
                    call.provenance = provenance
                    if resolved:
                        resolved_count += 1
                    else:
                        unresolved_count += 1

            # 2. Resolve class methods (with class and self assignments)
            for cls in pf.classes:
                class_assignments = cls.assignments
                for fn in cls.methods:
                    for call in fn.calls:
                        resolved, res_type, confidence, evidence, provenance = self._resolve_call_name(
                            call.name,
                            pf.module_name,
                            import_map,
                            fn_assignments=fn.assignments,
                            class_assignments=class_assignments,
                            calling_class_qn=cls.qualified_name,
                        )
                        call.resolved = resolved
                        call.resolution_type = res_type
                        call.confidence = confidence
                        call.evidence = evidence
                        call.provenance = provenance
                        if resolved:
                            resolved_count += 1
                        else:
                            unresolved_count += 1

        logger.debug(
            "Pass 3 — resolved %d calls, %d unresolved (builtins/dynamic)",
            resolved_count,
            unresolved_count,
        )

    def _find_class_constructor(self, class_qn: str, calling_module: str, import_map: dict[str, str]) -> Optional[str]:
        """Find the __init__ method for a class, searching up the inheritance tree if needed."""
        # 1. Direct constructor
        init_qn = f"{class_qn}.__init__"
        if init_qn in self._symbols:
            return init_qn
            
        # 2. Inherited constructor
        class_entry = self._symbols.get(class_qn)
        if class_entry and hasattr(class_entry, "bases"):
            queue = list(class_entry.bases)
            visited = set(class_entry.bases)
            while queue:
                base = queue.pop(0)
                resolved_base, _, _, _, _ = self._resolve_call_name(base, calling_module, import_map, skip_constructor_resolve=True)
                if resolved_base:
                    cand = f"{resolved_base}.__init__"
                    if cand in self._symbols:
                        return cand
                    curr_entry = self._symbols.get(resolved_base)
                    if curr_entry and hasattr(curr_entry, "bases"):
                        for parent in curr_entry.bases:
                            if parent not in visited:
                                visited.add(parent)
                                queue.append(parent)
        return None

    def _resolve_call_name(
        self,
        raw_name: str,
        calling_module: str,
        import_map: dict[str, str],
        fn_assignments: dict[str, str] = None,
        class_assignments: dict[str, str] = None,
        calling_class_qn: Optional[str] = None,
        skip_constructor_resolve: bool = False,
    ) -> tuple[Optional[str], ResolutionType, float, str, str]:
        """
        Resolve a raw call name to a fully-qualified symbol name.
        Returns (resolved_name, resolution_type, confidence, evidence, provenance).
        """
        from app.schemas.models import ResolutionType
        if not raw_name:
            return None, ResolutionType.UNRESOLVED, 0.0, "Empty raw name", "none"

        resolved = None
        res_type = ResolutionType.UNRESOLVED
        confidence = 0.0
        evidence = "Unresolved static call"
        provenance = "none"

        _GENERIC_METHOD_NAMES = frozenset({
            "__init__", "__call__", "forward", "fit", "predict", "save", "load",
            "run", "start", "stop", "close", "open", "update", "delete", "create",
            "get", "post", "put", "patch", "options", "head", "main", "execute"
        })

        # --- 0. super() Constructor & Method Resolution ---
        if raw_name.startswith("super.") and calling_class_qn:
            method_name = raw_name.split(".", 1)[1]
            class_entry = self._symbols.get(calling_class_qn)
            if class_entry and hasattr(class_entry, "bases"):
                queue = list(class_entry.bases)
                visited_bases = set(class_entry.bases)
                while queue:
                    base = queue.pop(0)
                    resolved_base, _, _, _, _ = self._resolve_call_name(base, calling_module, import_map, skip_constructor_resolve=True)
                    if resolved_base:
                        candidate = f"{resolved_base}.{method_name}"
                        if candidate in self._symbols:
                            resolved = candidate
                            res_type = ResolutionType.DIRECT
                            confidence = 1.0
                            evidence = f"super call resolved to base class method: {resolved}"
                            provenance = "inheritance"
                            break
                        curr_entry = self._symbols.get(resolved_base)
                        if curr_entry and hasattr(curr_entry, "bases"):
                            for parent in curr_entry.bases:
                                if parent not in visited_bases:
                                    visited_bases.add(parent)
                                    queue.append(parent)
                if resolved:
                    # Apply constructor mapping check before returning
                    if not skip_constructor_resolve and resolved in self._symbols:
                        entry = self._symbols[resolved]
                        if entry.symbol_type == SymbolType.CLASS:
                            constructor_qn = self._find_class_constructor(resolved, calling_module, import_map)
                            if constructor_qn:
                                return constructor_qn, res_type, confidence, f"Class instantiation resolved to constructor: {constructor_qn}", provenance
                    return resolved, res_type, confidence, evidence, provenance

        # --- 1. Framework / Callable Instance Call Resolution ---
        # e.g., model(x) where model maps to a class like GPT, or self(x)
        if "." not in raw_name:
            if raw_name == "self" and calling_class_qn:
                forward_qn = f"{calling_class_qn}.forward"
                if forward_qn in self._symbols:
                    resolved = forward_qn
                    res_type = ResolutionType.DIRECT
                    confidence = 1.0
                    evidence = f"self instance call resolved directly to forward: {resolved}"
                    provenance = "static"
                else:
                    call_qn = f"{calling_class_qn}.__call__"
                    if call_qn in self._symbols:
                        resolved = call_qn
                        res_type = ResolutionType.DIRECT
                        confidence = 1.0
                        evidence = f"self instance call resolved directly to __call__: {resolved}"
                        provenance = "static"
            
            elif fn_assignments and raw_name in fn_assignments:
                raw_class = fn_assignments[raw_name]
                resolved_class, _, _, _, _ = self._resolve_call_name(raw_class, calling_module, import_map, skip_constructor_resolve=True)
                if resolved_class:
                    forward_qn = f"{resolved_class}.forward"
                    if forward_qn in self._symbols:
                        resolved = forward_qn
                        res_type = ResolutionType.HEURISTIC
                        confidence = 0.8
                        evidence = f"Local callable instance variable '{raw_name}' mapped to forward: {resolved}"
                        provenance = "heuristic"
                    else:
                        call_qn = f"{resolved_class}.__call__"
                        if call_qn in self._symbols:
                            resolved = call_qn
                            res_type = ResolutionType.HEURISTIC
                            confidence = 0.8
                            evidence = f"Local callable instance variable '{raw_name}' mapped to __call__: {resolved}"
                            provenance = "heuristic"

        # --- 2. Direct import map ---
        if not resolved and raw_name in import_map:
            resolved = import_map[raw_name]
            res_type = ResolutionType.IMPORT
            confidence = 1.0
            evidence = f"Explicitly imported local name: {resolved}"
            provenance = "static"

        # --- 3. Attribute & Method Resolution Heuristics ---
        if not resolved and "." in raw_name:
            parts = raw_name.rsplit(".", 1)
            prefix = parts[0]
            method = parts[1]

            # 3a. Calls to self: self.my_method() or self.model()
            if prefix == "self" and calling_class_qn:
                candidate = f"{calling_class_qn}.{method}"
                if candidate in self._symbols:
                    resolved = candidate
                    res_type = ResolutionType.DIRECT
                    confidence = 1.0
                    evidence = f"Direct method call on self: {resolved}"
                    provenance = "static"
                else:
                    # Check base classes for self.method()
                    class_entry = self._symbols.get(calling_class_qn)
                    if class_entry and hasattr(class_entry, "bases"):
                        queue = list(class_entry.bases)
                        visited_bases = set(class_entry.bases)
                        while queue:
                            base = queue.pop(0)
                            resolved_base, _, _, _, _ = self._resolve_call_name(base, calling_module, import_map, skip_constructor_resolve=True)
                            if resolved_base:
                                cand = f"{resolved_base}.{method}"
                                if cand in self._symbols:
                                    resolved = cand
                                    res_type = ResolutionType.DIRECT
                                    confidence = 1.0
                                    evidence = f"Inherited method call on self resolved to base class: {resolved}"
                                    provenance = "inheritance"
                                    break
                                curr_entry = self._symbols.get(resolved_base)
                                if curr_entry and hasattr(curr_entry, "bases"):
                                    for parent in curr_entry.bases:
                                        if parent not in visited_bases:
                                            visited_bases.add(parent)
                                            queue.append(parent)

                # Check if it is a self attribute call (like self.model()) mapping to a callable class
                if not resolved and class_assignments and raw_name in class_assignments:
                    raw_class = class_assignments[raw_name]
                    resolved_class, _, _, _, _ = self._resolve_call_name(raw_class, calling_module, import_map, skip_constructor_resolve=True)
                    if resolved_class:
                        forward_qn = f"{resolved_class}.forward"
                        if forward_qn in self._symbols:
                            resolved = forward_qn
                            res_type = ResolutionType.HEURISTIC
                            confidence = 0.8
                            evidence = f"Self attribute instance variable '{raw_name}' mapped to forward: {resolved}"
                            provenance = "heuristic"
                        else:
                            call_qn = f"{resolved_class}.__call__"
                            if call_qn in self._symbols:
                                resolved = call_qn
                                res_type = ResolutionType.HEURISTIC
                                confidence = 0.8
                                evidence = f"Self attribute instance variable '{raw_name}' mapped to __call__: {resolved}"
                                provenance = "heuristic"

            # 3b. Calls to self attribute: self.model.forward()
            elif prefix.startswith("self.") and class_assignments and prefix in class_assignments:
                raw_class = class_assignments[prefix]
                resolved_class, _, _, _, _ = self._resolve_call_name(raw_class, calling_module, import_map, skip_constructor_resolve=True)
                if resolved_class:
                    candidate = f"{resolved_class}.{method}"
                    if candidate in self._symbols:
                        resolved = candidate
                        res_type = ResolutionType.HEURISTIC
                        confidence = 0.8
                        evidence = f"Method call on self attribute: {resolved}"
                        provenance = "heuristic"

            # 3c. Calls to local variables: trainer.fit()
            elif fn_assignments and prefix in fn_assignments:
                raw_class = fn_assignments[prefix]
                resolved_class, _, _, _, _ = self._resolve_call_name(raw_class, calling_module, import_map, skip_constructor_resolve=True)
                if resolved_class:
                    candidate = f"{resolved_class}.{method}"
                    if candidate in self._symbols:
                        resolved = candidate
                        res_type = ResolutionType.HEURISTIC
                        confidence = 0.8
                        evidence = f"Method call on local variable instance: {resolved}"
                        provenance = "heuristic"

        # --- 4. Dotted prefix (e.g. "jwt.decode") ---
        if not resolved and "." in raw_name:
            parts = raw_name.split(".", 1)
            prefix = parts[0]
            rest = parts[1]

            if prefix in import_map:
                resolved_prefix = import_map[prefix]
                candidate = f"{resolved_prefix}.{rest}"
                if candidate in self._symbols:
                    resolved = candidate
                    res_type = ResolutionType.IMPORT
                    confidence = 1.0
                    evidence = f"Dotted prefix resolved via import: {resolved}"
                    provenance = "static"
                else:
                    # Suffix match only if not a generic method name
                    last_seg = rest.split(".")[-1]
                    if last_seg not in _GENERIC_METHOD_NAMES:
                        matches = self._suffix_match(raw_name)
                        if matches:
                            resolved = matches[0]
                            res_type = ResolutionType.HEURISTIC
                            confidence = 0.5
                            evidence = f"Fallback suffix match on dotted import prefix: {resolved}"
                            provenance = "heuristic"

        # --- 5. Same-module (called without import, defined in same file) ---
        if not resolved:
            same_module_candidate = f"{calling_module}.{raw_name}"
            if same_module_candidate in self._symbols:
                resolved = same_module_candidate
                res_type = ResolutionType.DIRECT
                confidence = 1.0
                evidence = f"Same-module function/class call: {resolved}"
                provenance = "static"

        # --- 6. Exact global lookup ---
        if not resolved and raw_name in self._symbols:
            resolved = raw_name
            res_type = ResolutionType.IMPORT
            confidence = 1.0
            evidence = f"Exact global symbol registry match: {resolved}"
            provenance = "static"

        # --- 7. Framework-Aware Execution / Call Inference ---
        if not resolved:
            # Detect common neural network / module instance calls, e.g. model(x) or self.model(x)
            last_seg = raw_name.split(".")[-1]
            if last_seg.lower() in ("model", "net", "network", "module", "gpt", "backbone", "layer", "encoder", "decoder", "classifier"):
                nn_module_classes = []
                for qn, entry in self._symbols.items():
                    if entry.symbol_type == SymbolType.CLASS:
                        # Subclass defining forward or __call__
                        if f"{qn}.forward" in self._symbols or f"{qn}.__call__" in self._symbols:
                            nn_module_classes.append(qn)

                candidate_models = []
                for qn in nn_module_classes:
                    name_lower = qn.split(".")[-1].lower()
                    # Suppress common helper/normalization block classes to avoid incorrect path mapping
                    if any(layer in name_lower for layer in ("layernorm", "norm", "embedding", "linear", "attention", "mlp", "block", "loss")):
                        continue
                    candidate_models.append(qn)

                if not candidate_models:
                    candidate_models = nn_module_classes # fallback

                if candidate_models:
                    # Select the most significant orchestrating module (longest path is repository-defined main model)
                    candidate_models.sort(key=lambda qn: len(self._symbols.get(f"{qn}.__init__").file_path if f"{qn}.__init__" in self._symbols else ""), reverse=True)
                    best_model = candidate_models[0]
                    target_method = f"{best_model}.forward" if f"{best_model}.forward" in self._symbols else f"{best_model}.__call__"
                    if target_method in self._symbols:
                        resolved = target_method
                        res_type = ResolutionType.HEURISTIC
                        confidence = 0.5
                        evidence = f"Generalized framework inference: {raw_name} -> {resolved}"
                        provenance = "framework_inferred"

        # --- 8. Suffix match in symbol table ---
        if not resolved:
            last_seg = raw_name.split(".")[-1]
            if last_seg not in _GENERIC_METHOD_NAMES:
                suffix_matches = self._suffix_match(raw_name)
                if suffix_matches:
                    resolved = suffix_matches[0]
                    res_type = ResolutionType.HEURISTIC
                    confidence = 0.5
                    evidence = f"Fallback suffix match in symbol table: {resolved}"
                    provenance = "heuristic"

        # --- 9. Class Instantiation mapping to Constructor ---
        if resolved and resolved in self._symbols and not skip_constructor_resolve:
            entry = self._symbols[resolved]
            if entry.symbol_type == SymbolType.CLASS:
                constructor_qn = self._find_class_constructor(resolved, calling_module, import_map)
                if constructor_qn:
                    return constructor_qn, res_type, confidence, f"Class instantiation resolved to constructor: {constructor_qn}", provenance

        if resolved:
            return resolved, res_type, confidence, evidence, provenance

        return None, ResolutionType.UNRESOLVED, 0.0, "Unresolved static call", "none"


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
        resolved, _, _, _, _ = self._resolve_call_name(call_name, calling_module, import_map)
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
