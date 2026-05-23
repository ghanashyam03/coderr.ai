#!/usr/bin/env python3
"""
Coderr — Local-first AI Codebase Intelligence System.

Entry point for the CLI. Loads .env before importing any app modules
so that settings are correctly resolved from environment.
"""
from __future__ import annotations

from dotenv import load_dotenv

# Load .env before any app imports so pydantic-settings picks it up
load_dotenv()

from app.cli.commands import app  # noqa: E402

if __name__ == "__main__":
    app()
