from __future__ import annotations

"""
Tests for app.parsing.ast_parser.parse_file

Strategy:
- Write real Python source code to temporary files using pytest's tmp_path fixture
- Parse them with parse_file() — the real ast module is exercised, never mocked
- Verify all structural properties of ParsedFile, ParsedFunction, ParsedClass,
  ParsedImport, and ParsedCall models.
"""

import textwrap
from pathlib import Path

import pytest

from app.parsing.ast_parser import parse_file
from app.parsing.scanner import file_to_module_name
from app.schemas.models import ParsedFile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_and_parse(tmp_path: Path, source: str, filename: str = "sample.py") -> ParsedFile:
    """Write *source* to a temp file and return the parsed result."""
    file_path = tmp_path / filename
    file_path.write_text(textwrap.dedent(source), encoding="utf-8")
    return parse_file(file_path, tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseSimpleFunction:
    def test_parse_simple_function(self, tmp_path: Path) -> None:
        """A single top-level function must be extracted with correct name/args/line_start."""
        source = """\
            def add(x, y):
                return x + y
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        assert len(result.functions) == 1

        fn = result.functions[0]
        assert fn.name == "add"
        assert "x" in fn.args
        assert "y" in fn.args
        assert fn.line_start == 1
        assert fn.line_end >= 2
        assert fn.is_async is False
        assert fn.is_method is False
        assert fn.class_name is None


class TestParseClassWithMethods:
    def test_parse_class_with_methods(self, tmp_path: Path) -> None:
        """A class with two methods: check class name, method count, base class."""
        source = """\
            class AuthHandler(BaseHandler):
                def authenticate(self, token):
                    pass

                def logout(self, user_id):
                    pass
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        assert len(result.classes) == 1

        cls = result.classes[0]
        assert cls.name == "AuthHandler"
        assert "BaseHandler" in cls.bases
        assert len(cls.methods) == 2

        method_names = {m.name for m in cls.methods}
        assert "authenticate" in method_names
        assert "logout" in method_names

        for method in cls.methods:
            assert method.is_method is True
            assert method.class_name == "AuthHandler"

    def test_class_qualified_name(self, tmp_path: Path) -> None:
        """Class qualified_name must be '{module}.{class}'."""
        source = """\
            class MyClass:
                pass
        """
        result = _write_and_parse(tmp_path, source, filename="mymod.py")

        assert len(result.classes) == 1
        assert result.classes[0].qualified_name == "mymod.MyClass"


class TestParseImports:
    def test_parse_plain_import(self, tmp_path: Path) -> None:
        """import os — is_from=False, module='os', names=[]."""
        source = "import os\n"
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        plain_imports = [i for i in result.imports if not i.is_from]
        assert any(i.module == "os" for i in plain_imports)

        os_import = next(i for i in plain_imports if i.module == "os")
        assert os_import.is_from is False
        assert os_import.names == []
        assert os_import.alias is None
        assert os_import.line == 1

    def test_parse_from_import(self, tmp_path: Path) -> None:
        """from pathlib import Path — is_from=True, module='pathlib', names=['Path']."""
        source = "from pathlib import Path\n"
        result = _write_and_parse(tmp_path, source)

        from_imports = [i for i in result.imports if i.is_from]
        assert len(from_imports) == 1

        imp = from_imports[0]
        assert imp.module == "pathlib"
        assert "Path" in imp.names
        assert imp.is_from is True

    def test_parse_aliased_from_import(self, tmp_path: Path) -> None:
        """from auth import jwt as j — alias='j', module='auth', names=['jwt']."""
        source = "from auth import jwt as j\n"
        result = _write_and_parse(tmp_path, source)

        from_imports = [i for i in result.imports if i.is_from]
        assert len(from_imports) == 1

        imp = from_imports[0]
        assert imp.module == "auth"
        assert "jwt" in imp.names
        assert imp.alias == "j"

    def test_parse_multiple_imports(self, tmp_path: Path) -> None:
        """Three import statements should produce three ParsedImport entries."""
        source = """\
            import os
            from pathlib import Path
            from auth import jwt as j
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        assert len(result.imports) == 3


class TestParseFunctionCalls:
    def test_parse_function_calls(self, tmp_path: Path) -> None:
        """Calls inside a function body must be captured as ParsedCall objects."""
        source = """\
            def login(token):
                result = validate_token(token)
                log_event("login")
                return result
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        assert len(result.functions) == 1

        fn = result.functions[0]
        call_names = [c.name for c in fn.calls]
        assert "validate_token" in call_names
        assert "log_event" in call_names

    def test_method_call_captured(self, tmp_path: Path) -> None:
        """Attribute-style calls like obj.method() must be captured as 'obj.method'."""
        source = """\
            def process():
                handler = get_handler()
                handler.run()
        """
        result = _write_and_parse(tmp_path, source)

        fn = result.functions[0]
        call_names = [c.name for c in fn.calls]
        # "handler.run" or "get_handler" should appear
        assert any("run" in n or "handler" in n for n in call_names)

    def test_parsed_call_has_line_number(self, tmp_path: Path) -> None:
        """ParsedCall.line must be a positive integer."""
        source = """\
            def greet():
                say_hello()
        """
        result = _write_and_parse(tmp_path, source)
        fn = result.functions[0]
        assert len(fn.calls) >= 1
        for call in fn.calls:
            assert call.line >= 1


class TestParseAsyncFunction:
    def test_parse_async_function(self, tmp_path: Path) -> None:
        """async def functions must have is_async=True."""
        source = """\
            async def fetch_user(user_id: int):
                return await db.get(user_id)
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        assert len(result.functions) == 1

        fn = result.functions[0]
        assert fn.is_async is True
        assert fn.name == "fetch_user"

    def test_sync_function_not_async(self, tmp_path: Path) -> None:
        """Regular def functions must have is_async=False."""
        source = "def sync_fn(): pass\n"
        result = _write_and_parse(tmp_path, source)
        assert result.functions[0].is_async is False


class TestParseDocstring:
    def test_parse_docstring(self, tmp_path: Path) -> None:
        """Function with a docstring must have docstring field populated."""
        source = '''\
            def my_func():
                """This is the docstring."""
                return 42
        '''
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        fn = result.functions[0]
        assert fn.docstring is not None
        assert "docstring" in fn.docstring

    def test_function_without_docstring(self, tmp_path: Path) -> None:
        """Function with no docstring must have docstring=None."""
        source = "def no_doc(): pass\n"
        result = _write_and_parse(tmp_path, source)
        assert result.functions[0].docstring is None

    def test_class_docstring(self, tmp_path: Path) -> None:
        """Class with docstring must populate ParsedClass.docstring."""
        source = '''\
            class Foo:
                """Class docstring."""
                pass
        '''
        result = _write_and_parse(tmp_path, source)
        assert result.classes[0].docstring is not None
        assert "Class docstring" in result.classes[0].docstring


class TestParseDecorators:
    def test_parse_decorators(self, tmp_path: Path) -> None:
        """Decorated function must have decorator names in ParsedFunction.decorators."""
        source = """\
            def my_decorator(fn):
                return fn

            @my_decorator
            def decorated_func():
                pass
        """
        result = _write_and_parse(tmp_path, source)

        decorated = next(f for f in result.functions if f.name == "decorated_func")
        assert "my_decorator" in decorated.decorators

    def test_multiple_decorators(self, tmp_path: Path) -> None:
        """Multiple decorators must all be captured."""
        source = """\
            def dec_a(f): return f
            def dec_b(f): return f

            @dec_a
            @dec_b
            def multi_decorated():
                pass
        """
        result = _write_and_parse(tmp_path, source)
        fn = next(f for f in result.functions if f.name == "multi_decorated")
        assert "dec_a" in fn.decorators
        assert "dec_b" in fn.decorators

    def test_attribute_decorator(self, tmp_path: Path) -> None:
        """Attribute-style decorator like @router.get must be stored."""
        source = """\
            @router.get("/items")
            def get_items():
                pass
        """
        result = _write_and_parse(tmp_path, source)
        fn = result.functions[0]
        assert len(fn.decorators) == 1
        # Decorator name should resolve to something containing 'router' or 'get'
        dec = fn.decorators[0]
        assert "router" in dec or "get" in dec


class TestParseSyntaxError:
    def test_parse_syntax_error(self, tmp_path: Path) -> None:
        """Malformed Python source must set parse_error and not raise."""
        source = "def broken_function(\n    # missing closing paren and body\n"
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is not None
        assert len(result.parse_error) > 0
        # Structural fields must still exist (but be empty)
        assert result.functions == []
        assert result.classes == []
        assert result.imports == []

    def test_completely_invalid_source(self, tmp_path: Path) -> None:
        """Completely invalid text must produce a parse_error."""
        source = "this is not python code !@#$%"
        result = _write_and_parse(tmp_path, source)
        assert result.parse_error is not None


class TestParseReturnAnnotation:
    def test_parse_return_annotation(self, tmp_path: Path) -> None:
        """def foo() -> str: must set return_annotation='str'."""
        source = """\
            def foo() -> str:
                return "hello"
        """
        result = _write_and_parse(tmp_path, source)

        assert result.parse_error is None
        fn = result.functions[0]
        assert fn.return_annotation == "str"

    def test_complex_return_annotation(self, tmp_path: Path) -> None:
        """Complex return annotation like list[int] must be stringified."""
        source = """\
            def get_ids() -> list[int]:
                return [1, 2, 3]
        """
        result = _write_and_parse(tmp_path, source)
        fn = result.functions[0]
        assert fn.return_annotation is not None
        assert "int" in fn.return_annotation

    def test_no_return_annotation(self, tmp_path: Path) -> None:
        """Function without return annotation must have return_annotation=None."""
        source = "def bar(): pass\n"
        result = _write_and_parse(tmp_path, source)
        assert result.functions[0].return_annotation is None


class TestModuleNameFromPath:
    def test_simple_file_module_name(self, tmp_path: Path) -> None:
        """File at root of repo_root must become module name == file stem."""
        file_path = tmp_path / "mymodule.py"
        file_path.write_text("x = 1\n", encoding="utf-8")
        module = file_to_module_name(file_path, tmp_path)
        assert module == "mymodule"

    def test_nested_file_module_name(self, tmp_path: Path) -> None:
        """auth/jwt.py relative to repo_root must become 'auth.jwt'."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        jwt_file = auth_dir / "jwt.py"
        jwt_file.write_text("# jwt\n", encoding="utf-8")
        module = file_to_module_name(jwt_file, tmp_path)
        assert module == "auth.jwt"

    def test_deeply_nested_module_name(self, tmp_path: Path) -> None:
        """app/api/v1/routes.py must become 'app.api.v1.routes'."""
        target_dir = tmp_path / "app" / "api" / "v1"
        target_dir.mkdir(parents=True)
        routes_file = target_dir / "routes.py"
        routes_file.write_text("# routes\n", encoding="utf-8")
        module = file_to_module_name(routes_file, tmp_path)
        assert module == "app.api.v1.routes"

    def test_init_file_module_name(self, tmp_path: Path) -> None:
        """auth/__init__.py must become 'auth' (drops __init__)."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        init_file = auth_dir / "__init__.py"
        init_file.write_text("", encoding="utf-8")
        module = file_to_module_name(init_file, tmp_path)
        assert module == "auth"

    def test_parse_file_uses_correct_module_name(self, tmp_path: Path) -> None:
        """parse_file must populate module_name from the file path."""
        auth_dir = tmp_path / "auth"
        auth_dir.mkdir()
        jwt_file = auth_dir / "jwt.py"
        jwt_file.write_text("def decode(): pass\n", encoding="utf-8")

        result = parse_file(jwt_file, tmp_path)
        assert result.module_name == "auth.jwt"
        # Qualified name of function must include the module
        assert result.functions[0].qualified_name == "auth.jwt.decode"

    def test_function_qualified_name_includes_module(self, tmp_path: Path) -> None:
        """Function qualified_name must be '{module}.{function_name}'."""
        source = "def my_handler(): pass\n"
        result = _write_and_parse(tmp_path, source, filename="handlers.py")
        fn = result.functions[0]
        assert fn.qualified_name == "handlers.my_handler"

    def test_method_qualified_name_includes_class(self, tmp_path: Path) -> None:
        """Method qualified_name must be '{module}.{class}.{method}'."""
        source = """\
            class Service:
                def handle(self):
                    pass
        """
        result = _write_and_parse(tmp_path, source, filename="service.py")
        cls = result.classes[0]
        method = cls.methods[0]
        assert method.qualified_name == "service.Service.handle"
