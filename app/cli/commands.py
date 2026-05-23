from __future__ import annotations

"""
CLI Commands — Typer-based command-line interface for Coderr.

Commands:
    python main.py index <repo_path> [--name NAME] [--model MODEL]
    python main.py query "<question>" [--repo REPO] [--model MODEL]
    python main.py inspect-function <fn_name> [--repo REPO]
    python main.py dependencies <symbol> [--repo REPO] [--depth DEPTH]
    python main.py impact <symbol> [--repo REPO]
    python main.py list-repos
    python main.py serve [--port PORT] [--host HOST]
"""

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from app.config.settings import settings
from app.ingestion.ingestion_pipeline import IngestionPipeline, list_indexed_repos
from app.utils.logging_config import setup_logging

app = typer.Typer(
    name="coderr",
    help="Local-first AI Codebase Intelligence System for Python repositories.",
    add_completion=False,
    rich_markup_mode="rich",
)

console = Console()


def _get_intelligence(model: Optional[str] = None):
    """Lazy import to avoid loading ML models on startup."""
    from app.core.intelligence import CodeIntelligence
    return CodeIntelligence(model=model)


def _resolve_repo_name(repo: Optional[str]) -> str:
    """Resolve repo name from option or prompt user to pick from indexed repos."""
    if repo:
        return repo
    repos = list_indexed_repos()
    if not repos:
        console.print("[red]No repositories indexed yet.[/red]")
        console.print("Run: [cyan]python main.py index <repo_path>[/cyan]")
        raise typer.Exit(1)
    if len(repos) == 1:
        return repos[0]["repo_name"]
    console.print("[yellow]Multiple repos indexed. Specify with --repo:[/yellow]")
    for r in repos:
        console.print(f"  • [cyan]{r['repo_name']}[/cyan]  ({r['file_count']} files)")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("index")
def cmd_index(
    repo_path: str = typer.Argument(..., help="Path to the Python repository to index."),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Repository name (defaults to directory name)."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model override."),
    log_level: str = typer.Option(settings.LOG_LEVEL, "--log-level", help="Logging level."),
) -> None:
    """Index a Python repository — scan, parse, embed, and store."""
    setup_logging(level=log_level)

    path = Path(repo_path)
    if not path.exists():
        console.print(f"[red]Path does not exist: {repo_path}[/red]")
        raise typer.Exit(1)

    repo_name = name or path.resolve().name
    console.print(Panel(
        f"[bold cyan]Indexing Repository[/bold cyan]\n"
        f"Path: [green]{path.resolve()}[/green]\n"
        f"Name: [cyan]{repo_name}[/cyan]",
        title="Coderr",
        border_style="cyan",
    ))

    pipeline = IngestionPipeline()
    try:
        result = pipeline.index(repo_path=repo_path, repo_name=repo_name)
    except Exception as exc:
        console.print(f"[red]Indexing failed: {exc}[/red]")
        raise typer.Exit(1)

    # Summary table
    table = Table(title="Indexing Complete", border_style="green")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Files found", str(result.get("file_count", 0)))
    table.add_row("Files parsed", str(result.get("parsed_count", 0)))
    table.add_row("Parse errors", str(result.get("error_count", 0)))
    table.add_row("Symbols indexed", str(result.get("symbol_count", 0)))
    table.add_row("Graph nodes", str(result.get("node_count", 0)))
    table.add_row("Graph edges", str(result.get("edge_count", 0)))
    table.add_row("Duration", f"{result.get('duration_seconds', 0):.1f}s")
    console.print(table)
    console.print(f"\n[green]✓ Repository '{repo_name}' indexed successfully.[/green]")
    console.print(f"[dim]Query with: python main.py query \"your question\" --repo {repo_name}[/dim]")


@app.command("query")
def cmd_query(
    question: str = typer.Argument(..., help="Natural language question about the codebase."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository name."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model override."),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show retrieved source symbols."),
    log_level: str = typer.Option(settings.LOG_LEVEL, "--log-level"),
) -> None:
    """Query the indexed repository with a natural language question."""
    setup_logging(level=log_level)

    repo_name = _resolve_repo_name(repo)

    console.print(Panel(
        f"[bold]Question:[/bold] {question}\n"
        f"[dim]Repository: {repo_name}[/dim]",
        title="Coderr Query",
        border_style="cyan",
    ))

    intelligence = _get_intelligence(model=model)

    try:
        response = intelligence.query(repo_name=repo_name, question=question, model=model)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[red]Query failed: {exc}[/red]")
        raise typer.Exit(1)

    # Display answer
    console.print(Panel(
        response.answer,
        title=f"[green]Answer[/green] [dim](intent: {response.intent.value}, model: {response.model_used})[/dim]",
        border_style="green",
    ))

    if show_sources:
        console.print(f"\n[dim]Sources ({len(response.sources)} symbols, {response.context_chars} chars of context):[/dim]")
        for src in response.sources[:10]:
            console.print(f"  [dim]• {src}[/dim]")


@app.command("inspect-function")
def cmd_inspect(
    fn_name: str = typer.Argument(..., help="Function or method name to inspect."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository name."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Inspect a function: source location, callers, callees, and signature."""
    setup_logging(level=log_level)

    repo_name = _resolve_repo_name(repo)
    intelligence = _get_intelligence()

    try:
        result = intelligence.inspect_function(repo_name=repo_name, fn_name=fn_name)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    table = Table(title=f"Function Inspection: {result['qualified_name']}", border_style="cyan")
    table.add_column("Property", style="bold cyan")
    table.add_column("Value")

    table.add_row("Name", result.get("name", ""))
    table.add_row("Qualified Name", result.get("qualified_name", ""))
    table.add_row("Type", result.get("symbol_type", ""))
    table.add_row("File", result.get("file_path", ""))
    table.add_row("Line", str(result.get("line_start", "")))
    table.add_row("Class", result.get("class_name") or "—")
    table.add_row("Callers", str(result.get("callers_count", 0)))
    table.add_row("Callees", str(result.get("callees_count", 0)))
    console.print(table)

    if result.get("direct_callers"):
        console.print("\n[bold]Direct Callers:[/bold]")
        for caller in result["direct_callers"][:10]:
            console.print(f"  ← [yellow]{caller}[/yellow]")

    if result.get("direct_callees"):
        console.print("\n[bold]Direct Callees:[/bold]")
        for callee in result["direct_callees"][:10]:
            console.print(f"  → [green]{callee}[/green]")

    if len(result.get("all_matches", [])) > 1:
        console.print(f"\n[dim]Multiple matches found. Showing first. All: {result['all_matches']}[/dim]")


@app.command("dependencies")
def cmd_dependencies(
    symbol: str = typer.Argument(..., help="Symbol name to analyze dependencies for."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository name."),
    depth: int = typer.Option(2, "--depth", "-d", help="BFS traversal depth."),
    direction: str = typer.Option(
        "downstream",
        "--direction",
        "-dir",
        help="Dependency direction ('downstream', 'upstream', 'bidirectional', 'impact', 'execution').",
    ),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Show the dependency chain for a symbol (what it depends on)."""
    setup_logging(level=log_level)

    repo_name = _resolve_repo_name(repo)
    intelligence = _get_intelligence()

    try:
        result = intelligence.get_dependencies(
            repo_name=repo_name, symbol_name=symbol, depth=depth, direction=direction
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    if "error" in result:
        console.print(f"[red]{result['error']}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]{result['symbol']}[/bold]\n"
        f"[dim]{result['file_path']}[/dim]\n"
        f"[dim]Direction: {result['direction']} | BFS Depth: {result['dependency_depth']}[/dim]",
        title="Dependency Analysis",
        border_style="cyan",
    ))

    console.print(f"\n[bold cyan]Direct Dependencies ({len(result['direct_dependencies'])}):[/bold cyan]")
    edge_meta = result.get("edge_metadata", {})
    for dep in result["direct_dependencies"]:
        meta = edge_meta.get(dep)
        if meta and meta.get("resolution_type") != "unresolved":
            conf = meta.get("confidence", 1.0)
            prov = meta.get("provenance", "none")
            evidence = meta.get("evidence", "")
            meta_str = f" [dim](confidence: {conf:.1f}, provenance: {prov}"
            if evidence:
                meta_str += f" | {evidence}"
            meta_str += ")[/dim]"
        else:
            meta_str = ""
        console.print(f"  → [green]{dep}[/green]{meta_str}")

    if result.get("transitive_dependencies"):
        console.print(f"\n[bold yellow]Transitive Dependencies ({len(result['transitive_dependencies'])}):[/bold yellow]")
        for dep in result["transitive_dependencies"][:20]:
            console.print(f"  ⇒ [dim]{dep}[/dim]")
        if len(result["transitive_dependencies"]) > 20:
            console.print(f"  [dim]... and {len(result['transitive_dependencies']) - 20} more[/dim]")


@app.command("impact")
def cmd_impact(
    symbol: str = typer.Argument(..., help="Symbol to analyze impact for (e.g. 'validate_token')."),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository name."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Ollama model for summary."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Impact analysis: what breaks if you change this symbol?"""
    setup_logging(level=log_level)

    repo_name = _resolve_repo_name(repo)
    intelligence = _get_intelligence(model=model)

    try:
        impact = intelligence.analyze_impact(repo_name=repo_name, symbol_name=symbol)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    report = intelligence.format_impact_report(impact)
    console.print(report)


@app.command("analyze-architecture")
def cmd_analyze_architecture(
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="Repository name."),
    log_level: str = typer.Option("WARNING", "--log-level"),
) -> None:
    """Perform a deep architectural analysis of the repository."""
    setup_logging(level=log_level)

    repo_name = _resolve_repo_name(repo)
    intelligence = _get_intelligence()

    try:
        report = intelligence.analyze_architecture(repo_name=repo_name)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Repository Architecture Report[/bold cyan]\n"
        f"Repository: [green]{repo_name}[/green]",
        title="Coderr Architecture Intelligence",
        border_style="cyan",
    ))

    # 1. Circular Imports
    circ = report.get("circular_imports", [])
    if circ:
        console.print("\n[bold red]⚠️ Circular Import Cycles Detected:[/bold red]")
        for i, cycle in enumerate(circ):
            cycle_str = " → ".join(cycle)
            console.print(f"  [red]Cycle {i+1}:[/red] {cycle_str}")
    else:
        console.print("\n[bold green]✓ No Circular Imports Detected.[/bold green]")

    # 2. Dependency Hubs Table
    hubs = report.get("dependency_hubs", [])
    if hubs:
        table = Table(title="Top 10 Dependency Hubs (High In-Degree)", border_style="cyan")
        table.add_column("Symbol", style="bold green")
        table.add_column("Type", style="yellow")
        table.add_column("In-Degree (Incoming Calls)", justify="right")
        for h in hubs:
            table.add_row(h["qualified_name"], h["symbol_type"], str(h["in_degree"]))
        console.print("\n")
        console.print(table)

    # 3. Oversized Orchestrators Table
    orch = report.get("oversized_orchestrators", [])
    if orch:
        table = Table(title="Top 10 Oversized Orchestrators (High Out-Degree)", border_style="cyan")
        table.add_column("Symbol", style="bold yellow")
        table.add_column("Type", style="yellow")
        table.add_column("Out-Degree (Outgoing Calls)", justify="right")
        for o in orch:
            table.add_row(o["qualified_name"], o["symbol_type"], str(o["out_degree"]))
        console.print("\n")
        console.print(table)

    # 4. Coupling & Instability Table
    coupling = report.get("coupling_instability", {})
    if coupling:
        table = Table(title="Component Coupling & Instability Scores", border_style="cyan")
        table.add_column("Module File", style="dim")
        table.add_column("Afferent (In-coupling)", justify="right")
        table.add_column("Efferent (Out-coupling)", justify="right")
        table.add_column("Instability Score (I)", justify="right")
        
        # Sort files by instability descending, show top 10
        sorted_coupling = sorted(coupling.items(), key=lambda x: x[1]["instability"], reverse=True)[:10]
        for path, metrics in sorted_coupling:
            table.add_row(
                path,
                str(metrics["afferent_coupling"]),
                str(metrics["efferent_coupling"]),
                f"[bold red]{metrics['instability']}[/bold red]" if metrics["instability"] > 0.7 else str(metrics["instability"])
            )
        console.print("\n")
        console.print(table)

    # 5. Dead Code Candidates
    dead = report.get("dead_code_candidates", [])
    if dead:
        table = Table(title="Dead Code Candidates (No Incoming Calls)", border_style="cyan")
        table.add_column("Symbol", style="bold red")
        table.add_column("Type", style="yellow")
        table.add_column("File Path", style="dim")
        table.add_column("Line", justify="right")
        for d in dead[:10]:
            table.add_row(d["qualified_name"], d["symbol_type"], d["file_path"], str(d["line_start"]))
        console.print("\n")
        console.print(table)


@app.command("list-repos")
def cmd_list_repos(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON."),
) -> None:
    """List all indexed repositories."""
    repos = list_indexed_repos()

    if not repos:
        console.print("[yellow]No repositories indexed yet.[/yellow]")
        console.print("Run: [cyan]python main.py index <repo_path>[/cyan]")
        return

    if json_output:
        print(json.dumps(repos, indent=2, default=str))
        return

    table = Table(title="Indexed Repositories", border_style="cyan")
    table.add_column("Name", style="bold cyan")
    table.add_column("Files", justify="right")
    table.add_column("Symbols", justify="right")
    table.add_column("Nodes", justify="right")
    table.add_column("Indexed At")
    table.add_column("Path", style="dim")

    for r in repos:
        table.add_row(
            r.get("repo_name", "?"),
            str(r.get("file_count", "?")),
            str(r.get("symbol_count", "?")),
            str(r.get("node_count", "?")),
            str(r.get("indexed_at", ""))[:19],
            r.get("repo_path", ""),
        )

    console.print(table)


@app.command("serve")
def cmd_serve(
    host: str = typer.Option("127.0.0.1", "--host", help="API server host."),
    port: int = typer.Option(8000, "--port", "-p", help="API server port."),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes (dev mode)."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Start the FastAPI REST API server."""
    setup_logging(level=log_level)

    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn is not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]Coderr API Server[/bold]\n"
        f"Host: [cyan]{host}:{port}[/cyan]\n"
        f"Docs: [cyan]http://{host}:{port}/docs[/cyan]",
        border_style="cyan",
    ))

    uvicorn.run(
        "app.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
    )
