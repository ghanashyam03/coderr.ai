from __future__ import annotations

"""
Tests for app.parsing.symbol_registry.SymbolRegistry and build_registry().

Strategy:
- Build ParsedFile, ParsedFunction, ParsedClass objects manually (no file I/O)
- Call build_registry(parsed_files) and inspect results
- Test symbol indexing, import resolution, call resolution, and serialisation
"""

import pytest

from app.parsing.symbol_registry import SymbolRegistry, build_registry
from app.schemas.models import (
    ParsedCall,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    SymbolType,
)


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def _make_function(
    name: str,
    module: str,
    file_path: str,
    *,
    class_name: str | None = None,
    calls: list[ParsedCall] | None = None,
    line_start: int = 1,
    line_end: int = 5,
) -> ParsedFunction:
    """Create a minimal ParsedFunction for testing."""
    prefix = f"{module}.{class_name}.{name}" if class_name else f"{module}.{name}"
    return ParsedFunction(
        name=name,
        qualified_name=prefix,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=f"def {name}(): pass",
        calls=calls or [],
        is_async=False,
        is_method=class_name is not None,
        class_name=class_name,
    )


def _make_class(
    name: str,
    module: str,
    file_path: str,
    methods: list[ParsedFunction] | None = None,
    bases: list[str] | None = None,
) -> ParsedClass:
    """Create a minimal ParsedClass for testing."""
    return ParsedClass(
        name=name,
        qualified_name=f"{module}.{name}",
        file_path=file_path,
        line_start=1,
        line_end=20,
        source=f"class {name}: pass",
        bases=bases or [],
        methods=methods or [],
    )


def _make_file(
    path: str,
    module: str,
    *,
    imports: list[ParsedImport] | None = None,
    functions: list[ParsedFunction] | None = None,
    classes: list[ParsedClass] | None = None,
) -> ParsedFile:
    """Create a minimal ParsedFile for testing."""
    return ParsedFile(
        path=path,
        module_name=module,
        imports=imports or [],
        functions=functions or [],
        classes=classes or [],
    )


# ---------------------------------------------------------------------------
# Test: Register function
# ---------------------------------------------------------------------------


class TestRegisterFunction:
    def test_register_function(self) -> None:
        """Single ParsedFile with one function must be findable by qualified name."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])

        registry = build_registry([pf])
        symbols = registry.get_all_symbols()

        assert "auth.jwt.validate_token" in symbols
        entry = symbols["auth.jwt.validate_token"]
        assert entry.symbol_type == SymbolType.FUNCTION
        assert entry.file_path == "/repo/auth/jwt.py"

    def test_function_not_present_if_absent(self) -> None:
        """Symbols not in the parsed files must not appear in the registry."""
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt")
        registry = build_registry([pf])
        assert registry.get_symbol("auth.jwt.nonexistent") is None


# ---------------------------------------------------------------------------
# Test: Register class and methods
# ---------------------------------------------------------------------------


class TestRegisterClassAndMethods:
    def test_register_class_and_methods(self) -> None:
        """Class + 2 methods → 3 registry entries (class + method_a + method_b)."""
        method_a = _make_function("encode", "auth.jwt", "/repo/auth/jwt.py", class_name="JWTHandler")
        method_b = _make_function("decode", "auth.jwt", "/repo/auth/jwt.py", class_name="JWTHandler")
        cls = _make_class("JWTHandler", "auth.jwt", "/repo/auth/jwt.py", methods=[method_a, method_b])
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", classes=[cls])

        registry = build_registry([pf])
        symbols = registry.get_all_symbols()

        assert "auth.jwt.JWTHandler" in symbols
        assert "auth.jwt.JWTHandler.encode" in symbols
        assert "auth.jwt.JWTHandler.decode" in symbols

        # Verify symbol types
        assert symbols["auth.jwt.JWTHandler"].symbol_type == SymbolType.CLASS
        assert symbols["auth.jwt.JWTHandler.encode"].symbol_type == SymbolType.METHOD
        assert symbols["auth.jwt.JWTHandler.decode"].symbol_type == SymbolType.METHOD

    def test_method_class_name_stored(self) -> None:
        """Method registry entry must store class_name."""
        method = _make_function("run", "svc", "/svc.py", class_name="Service")
        cls = _make_class("Service", "svc", "/svc.py", methods=[method])
        pf = _make_file("/svc.py", "svc", classes=[cls])

        registry = build_registry([pf])
        entry = registry.get_symbol("svc.Service.run")
        assert entry is not None
        assert entry.class_name == "Service"


# ---------------------------------------------------------------------------
# Test: Resolve from-import
# ---------------------------------------------------------------------------


class TestResolveFromImport:
    def test_resolve_from_import(self) -> None:
        """
        File with 'from auth.jwt import validate_token' and a call to validate_token
        must have the call resolved to 'auth.jwt.validate_token'.
        """
        # Define validate_token in auth.jwt
        fn_jwt = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf_jwt = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn_jwt])

        # Define login in auth.routes, importing validate_token
        call = ParsedCall(name="validate_token", line=5)
        fn_login = _make_function(
            "login", "auth.routes", "/repo/auth/routes.py", calls=[call]
        )
        imp = ParsedImport(module="auth.jwt", names=["validate_token"], is_from=True, line=1)
        pf_routes = _make_file(
            "/repo/auth/routes.py", "auth.routes",
            imports=[imp],
            functions=[fn_login],
        )

        registry = build_registry([pf_jwt, pf_routes])

        # After registry build, the call's resolved field must be set
        login_fn = pf_routes.functions[0]
        resolved_call = login_fn.calls[0]
        assert resolved_call.resolved == "auth.jwt.validate_token"

    def test_unresolvable_call_remains_none(self) -> None:
        """A call to an unknown external function must have resolved=None."""
        call = ParsedCall(name="os_open_something_unknown", line=3)
        fn = _make_function("use_it", "mymod", "/mymod.py", calls=[call])
        pf = _make_file("/mymod.py", "mymod", functions=[fn])

        registry = build_registry([pf])
        assert fn.calls[0].resolved is None


# ---------------------------------------------------------------------------
# Test: Resolve import alias
# ---------------------------------------------------------------------------


class TestResolveImportAlias:
    def test_resolve_import_alias(self) -> None:
        """
        'from auth import jwt as j' then call 'j.decode' should resolve correctly
        to something referencing auth.jwt.decode.
        """
        # Define decode in auth.jwt
        fn_decode = _make_function("decode", "auth.jwt", "/repo/auth/jwt.py")
        pf_jwt = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn_decode])

        # Consumer imports jwt module as alias 'j'
        call = ParsedCall(name="j.decode", line=5)
        fn_consumer = _make_function("consumer", "app.main", "/repo/app/main.py", calls=[call])
        # "from auth import jwt as j" → alias='j', module='auth', names=['jwt']
        imp = ParsedImport(module="auth", names=["jwt"], alias="j", is_from=True, line=1)
        pf_main = _make_file(
            "/repo/app/main.py", "app.main",
            imports=[imp],
            functions=[fn_consumer],
        )

        registry = build_registry([pf_jwt, pf_main])

        # The import map for main.py should map 'j' -> 'auth.jwt'
        import_map = registry.get_file_import_map("/repo/app/main.py")
        assert "j" in import_map
        assert import_map["j"] == "auth.jwt"


# ---------------------------------------------------------------------------
# Test: find_by_name
# ---------------------------------------------------------------------------


class TestFindByName:
    def test_find_by_name(self) -> None:
        """registry.find_by_name('validate_token') returns the correct entry."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])

        registry = build_registry([pf])
        entries = registry.find_by_name("validate_token")

        assert len(entries) == 1
        assert entries[0].qualified_name == "auth.jwt.validate_token"

    def test_find_by_name_returns_empty_for_unknown(self) -> None:
        """find_by_name with an unknown name must return an empty list."""
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt")
        registry = build_registry([pf])
        assert registry.find_by_name("nonexistent_func") == []

    def test_find_by_name_multiple_matches(self) -> None:
        """Two files each defining 'helper' must both appear in find_by_name results."""
        fn_a = _make_function("helper", "mod_a", "/mod_a.py")
        fn_b = _make_function("helper", "mod_b", "/mod_b.py")
        pf_a = _make_file("/mod_a.py", "mod_a", functions=[fn_a])
        pf_b = _make_file("/mod_b.py", "mod_b", functions=[fn_b])

        registry = build_registry([pf_a, pf_b])
        entries = registry.find_by_name("helper")
        qnames = {e.qualified_name for e in entries}

        assert "mod_a.helper" in qnames
        assert "mod_b.helper" in qnames


# ---------------------------------------------------------------------------
# Test: find_by_name_fragment
# ---------------------------------------------------------------------------


class TestFindByNameFragment:
    def test_find_by_name_fragment(self) -> None:
        """registry.find_by_name_fragment('validate') must return validate_token."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])

        registry = build_registry([pf])
        entries = registry.find_by_name_fragment("validate")

        qnames = [e.qualified_name for e in entries]
        assert "auth.jwt.validate_token" in qnames

    def test_find_by_name_fragment_case_insensitive(self) -> None:
        """Fragment search must be case-insensitive."""
        fn = _make_function("JWTHandler", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])

        registry = build_registry([pf])
        entries = registry.find_by_name_fragment("jwthandler")
        qnames = [e.qualified_name for e in entries]
        assert "auth.jwt.JWTHandler" in qnames

    def test_find_by_name_fragment_empty(self) -> None:
        """Fragment not present must return empty list."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])
        registry = build_registry([pf])
        assert registry.find_by_name_fragment("xyz_no_match_abc") == []


# ---------------------------------------------------------------------------
# Test: Same-module call resolution
# ---------------------------------------------------------------------------


class TestSameModuleCallResolution:
    def test_same_module_call_resolution(self) -> None:
        """
        Function A in module 'mymod' calling function B (also in 'mymod')
        without any import statement must resolve B to 'mymod.b_func'.
        """
        call = ParsedCall(name="b_func", line=3)
        fn_a = _make_function("a_func", "mymod", "/mymod.py", calls=[call])
        fn_b = _make_function("b_func", "mymod", "/mymod.py")
        pf = _make_file("/mymod.py", "mymod", functions=[fn_a, fn_b])

        registry = build_registry([pf])

        # After build, a_func's call should resolve to mymod.b_func
        resolved = fn_a.calls[0].resolved
        assert resolved == "mymod.b_func"

    def test_same_module_class_call(self) -> None:
        """A method calling a module-level function in same module must resolve."""
        call = ParsedCall(name="helper", line=5)
        method = _make_function("run", "svc", "/svc.py", class_name="Runner", calls=[call])
        fn_helper = _make_function("helper", "svc", "/svc.py")
        cls = _make_class("Runner", "svc", "/svc.py", methods=[method])
        pf = _make_file("/svc.py", "svc", functions=[fn_helper], classes=[cls])

        registry = build_registry([pf])
        assert method.calls[0].resolved == "svc.helper"


# ---------------------------------------------------------------------------
# Test: to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestToDictRoundtrip:
    def test_to_dict_roundtrip(self) -> None:
        """Serialise registry to dict, restore with from_dict, verify symbols match."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        cls_method = _make_function("encode", "auth.jwt", "/repo/auth/jwt.py", class_name="JWTHandler")
        cls = _make_class("JWTHandler", "auth.jwt", "/repo/auth/jwt.py", methods=[cls_method])
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn], classes=[cls])

        original = build_registry([pf])
        data = original.to_dict()

        restored = SymbolRegistry.from_dict(data)
        restored_symbols = restored.get_all_symbols()

        # Verify all original symbols are present
        for qn in original.get_all_symbols():
            assert qn in restored_symbols, f"Missing symbol: {qn}"

    def test_to_dict_produces_serialisable_dict(self) -> None:
        """to_dict() must produce a plain dict (JSON-serialisable structure)."""
        import json

        fn = _make_function("my_fn", "mymod", "/mymod.py")
        pf = _make_file("/mymod.py", "mymod", functions=[fn])
        registry = build_registry([pf])
        data = registry.to_dict()

        # Must be a dict
        assert isinstance(data, dict)
        # Must be JSON serialisable (raises if not)
        serialised = json.dumps(data)
        assert len(serialised) > 10

    def test_from_dict_preserves_name_index(self) -> None:
        """Restored registry must support find_by_name correctly."""
        fn = _make_function("validate_token", "auth.jwt", "/repo/auth/jwt.py")
        pf = _make_file("/repo/auth/jwt.py", "auth.jwt", functions=[fn])

        original = build_registry([pf])
        restored = SymbolRegistry.from_dict(original.to_dict())

        entries = restored.find_by_name("validate_token")
        assert len(entries) == 1
        assert entries[0].qualified_name == "auth.jwt.validate_token"

    def test_from_dict_preserves_file_import_maps(self) -> None:
        """Restored registry must preserve file import maps for resolution."""
        imp = ParsedImport(module="auth.jwt", names=["validate_token"], is_from=True, line=1)
        pf = _make_file("/repo/routes.py", "routes", imports=[imp])

        original = build_registry([pf])
        restored = SymbolRegistry.from_dict(original.to_dict())

        import_map = restored.get_file_import_map("/repo/routes.py")
        assert "validate_token" in import_map
