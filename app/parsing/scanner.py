from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Directories to completely skip during scanning
IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "venv",
        "env",
        ".env",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        ".tox",
        ".eggs",
        "*.egg-info",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "htmlcov",
        "site-packages",
        ".idea",
        ".vscode",
    }
)


def should_ignore_dir(directory: Path) -> bool:
    """Return True if this directory should be excluded from scanning."""
    name = directory.name
    # Exact name match
    if name in IGNORE_DIRS:
        return True
    # Egg-info suffix
    if name.endswith(".egg-info"):
        return True
    # Hidden directories (starting with dot, but not '.')
    if name.startswith(".") and len(name) > 1:
        return True
    return False


def scan_repository(repo_path: str | Path) -> list[Path]:
    """
    Recursively discover all Python source files in a repository.

    Ignores:
    - Hidden directories
    - Virtual environments
    - Build artifacts
    - Cache directories

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        Sorted list of absolute Path objects for every .py file found.
    """
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    logger.info("Scanning repository: %s", root)

    discovered: list[Path] = []
    skipped_dirs: list[str] = []

    def _walk(directory: Path) -> None:
        try:
            entries = sorted(directory.iterdir())
        except PermissionError:
            logger.warning("Permission denied reading directory: %s", directory)
            return

        for entry in entries:
            if entry.is_dir():
                if should_ignore_dir(entry):
                    skipped_dirs.append(entry.name)
                    logger.debug("Skipping directory: %s", entry)
                else:
                    _walk(entry)
            elif entry.is_file() and entry.suffix == ".py":
                discovered.append(entry)

    _walk(root)

    discovered.sort()
    logger.info(
        "Scan complete — %d Python files found (skipped dirs: %s)",
        len(discovered),
        sorted(set(skipped_dirs)),
    )
    return discovered


def file_to_module_name(file_path: Path, repo_root: Path) -> str:
    """
    Convert a file path to a dot-separated Python module name.

    Example:
        repo_root = /home/user/my_project
        file_path = /home/user/my_project/auth/jwt.py
        → "auth.jwt"

    Strips __init__.py to become the package name itself.
    """
    try:
        relative = file_path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        # File is not inside repo_root — use file stem
        return file_path.stem

    parts = list(relative.parts)

    # Drop the .py extension from last component
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]

    # __init__ → parent package name (drop the __init__ component)
    if parts and parts[-1] == "__init__":
        parts.pop()

    return ".".join(parts) if parts else file_path.stem
