from __future__ import annotations

"""
Context Assembler — converts retrieval results into structured, token-budgeted
hierarchical context for LLM reasoning.

The assembled context is split into 6 execution-aware sections:

    [PROJECT PURPOSE]
        Global repository summary and directory architectural roles loaded from profile.json.
        Ensures the LLM understands the global architecture.

    [ENTRYPOINTS]
        HTTP routes, CLI command handlers, task definitions, or external functions.
        These are the surface area of the system.

    [HIGH LEVEL FLOW]
        Probable execution pathway traced from entrypoint down to services, formatted as
        an ASCII tree, followed by the source code of participating flow modules.

    [ORCHESTRATORS]
        Symbols orchestrating system operations (high out-degree symbols).

    [DEPENDENCIES]
        ベッドロック symbols acting as dependencies (high in-degree symbols).

    [IMPLEMENTATION DETAILS]
        Low-value utilities and leaves. These are compressed (signature + docstring only)
        when they qualify as low-value helpers to preserve token budget.

Deduplication:
    No symbol appears in more than one section.
    Within a section, no duplicate symbol_ids.
"""

import json
import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from app.schemas.models import (
    AssembledContext,
    CodeSymbol,
    QueryIntent,
    RetrievalResult,
    SymbolType,
)

if TYPE_CHECKING:
    from app.graph.graph_engine import CodeGraph

logger = logging.getLogger(__name__)

# Hierarchical section names
SECTION_PROJECT_PURPOSE = "PROJECT PURPOSE"
SECTION_ENTRY_POINTS = "ENTRYPOINTS"
SECTION_HIGH_LEVEL_FLOW = "HIGH LEVEL FLOW"
SECTION_ORCHESTRATORS = "ORCHESTRATORS"
SECTION_DEPENDENCIES = "DEPENDENCIES"
SECTION_IMPLEMENTATION_DETAILS = "IMPLEMENTATION DETAILS"
SECTION_IMPORTS = "IMPORTANT IMPORTS"

# Backward compatibility aliases for older tests
SECTION_CORE_FLOW = SECTION_HIGH_LEVEL_FLOW
SECTION_UTILITIES = SECTION_IMPLEMENTATION_DETAILS

# Budget allocation (fractions of max_chars)
_BUDGET_FRACTIONS = {
    SECTION_PROJECT_PURPOSE: 0.15,
    SECTION_ENTRY_POINTS: 0.25,
    SECTION_HIGH_LEVEL_FLOW: 0.20,
    SECTION_ORCHESTRATORS: 0.15,
    SECTION_DEPENDENCIES: 0.15,
    SECTION_IMPLEMENTATION_DETAILS: 0.10,
}

_ROUTE_DECORATOR_KEYWORDS = frozenset(
    {"route", "get", "post", "put", "delete", "patch", "head", "options", "websocket"}
)


class ContextAssembler:
    """
    Assembles retrieval results into an LLM-ready structured context string.

    Args:
        graph: CodeGraph used for topological ordering and entry point detection.
        max_chars: Maximum total character budget for assembled context.
    """

    def __init__(
        self,
        graph: "CodeGraph",
        max_chars: int = 8000,
    ) -> None:
        self.graph = graph
        self.max_chars = max_chars

    def _load_profile(self, repo_name: str) -> Optional[dict]:
        """Load the repository profile from disk if available."""
        from app.config.settings import settings
        profile_path = Path(settings.CODERR_DATA_DIR) / repo_name / "profile.json"
        if profile_path.exists():
            try:
                with open(profile_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning("Failed to load profile.json: %s", exc)
        return None

    def assemble(
        self,
        results: list[RetrievalResult],
        query: str,
        intent: QueryIntent,
        target_symbol: Optional[str] = None,
    ) -> AssembledContext:
        """
        Assemble retrieval results into structured hierarchical context.
        """
        if not results:
            return AssembledContext(
                sections={},
                full_text="No relevant code found for this query.",
                sources=[],
                total_chars=0,
                truncated=False,
            )

        logger.debug(
            "Assembling hierarchical context for query='%s' intent=%s from %d results",
            query[:60],
            intent.value,
            len(results),
        )

        # Deduplicate by symbol_id, keep highest score
        deduped = self._deduplicate(results)

        # Fetch repo name
        repo_name = deduped[0].symbol.repo_name if deduped else "unknown"

        # Load profile
        profile = self._load_profile(repo_name)

        # 1. Render PROJECT PURPOSE (Non-symbol based text section)
        purpose_budget = int(self.max_chars * _BUDGET_FRACTIONS[SECTION_PROJECT_PURPOSE])
        project_purpose_text = self._render_project_purpose(profile, deduped, purpose_budget)

        # 2. Classify symbols into their respective architectural bins
        classified = self._classify_into_bins(deduped, intent, target_symbol)
        
        # Extract the high-level execution tree diagram
        high_level_flow_diagram = classified.get("flow_diagram", "")
        symbols_by_section = classified["symbols"]

        # 3. Render symbol-based sections within budget
        built_sections: dict[str, str] = {}
        if project_purpose_text:
            built_sections[SECTION_PROJECT_PURPOSE] = project_purpose_text

        included_ids: list[str] = []
        total_chars = len(project_purpose_text)
        truncated = False

        sections_to_render = [
            SECTION_ENTRY_POINTS,
            SECTION_HIGH_LEVEL_FLOW,
            SECTION_ORCHESTRATORS,
            SECTION_DEPENDENCIES,
            SECTION_IMPLEMENTATION_DETAILS,
        ]

        for section_name in sections_to_render:
            symbols = symbols_by_section.get(section_name, [])
            if not symbols and section_name != SECTION_HIGH_LEVEL_FLOW:
                continue

            budget = int(self.max_chars * _BUDGET_FRACTIONS[section_name])
            remaining = self.max_chars - total_chars
            effective_budget = min(budget, remaining)

            if effective_budget < 50:
                truncated = True
                break

            # Special rendering for high-level flow (prepends tree diagram)
            prepend_text = ""
            if section_name == SECTION_HIGH_LEVEL_FLOW and high_level_flow_diagram:
                prepend_text = high_level_flow_diagram
                effective_budget -= len(prepend_text)

            section_text, section_ids, section_truncated = self._render_section(
                section_name, symbols, max(50, effective_budget)
            )

            if section_truncated:
                truncated = True

            final_section_text = prepend_text + section_text
            if final_section_text.strip():
                built_sections[section_name] = final_section_text
                included_ids.extend(section_ids)
                total_chars += len(final_section_text)

        # Always add imports section (compact, not strictly budgeted but low footprint)
        imports_text = self._build_imports_section(deduped, included_ids)
        if imports_text:
            built_sections[SECTION_IMPORTS] = imports_text
            total_chars += len(imports_text)

        # Build full text with section headers
        full_text = self._join_sections(built_sections)

        logger.debug(
            "Hierarchical context assembled — %d chars, %d symbols, %d sections, truncated=%s",
            total_chars,
            len(included_ids),
            len(built_sections),
            truncated,
        )

        return AssembledContext(
            sections=built_sections,
            full_text=full_text,
            sources=included_ids,
            total_chars=len(full_text),
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # Project Purpose rendering
    # ------------------------------------------------------------------

    def _render_project_purpose(self, profile: Optional[dict], deduped: list[RetrievalResult], budget: int) -> str:
        """Render the PROJECT PURPOSE section containing repo profiles & summaries."""
        if profile:
            overview = profile.get("system_overview", "")
            roles = profile.get("module_roles", {})
            
            lines = [
                "SYSTEM OVERVIEW:",
                overview,
                "",
                "ARCHITECTURAL ROLES BY COMPONENT:"
            ]
            for folder, desc in sorted(roles.items()):
                lines.append(f"- {folder}: {desc}")
                
            text = "\n".join(lines) + "\n"
        else:
            # Dynamic fallback summary
            files = {r.symbol.file_path for r in deduped}
            text = (
                "SYSTEM OVERVIEW (Dynamic Fallback):\n"
                f"This repository contains a set of related Python modules representing active codebase units.\n"
                f"Active files analyzed: {len(files)} files.\n"
            )
            
        if len(text) > budget:
            text = text[:budget - 50] + "\n... [Project Purpose summary truncated]\n"
        return text

    # ------------------------------------------------------------------
    # Section classification
    # ------------------------------------------------------------------

    def _execution_critical_score(self, sym: CodeSymbol) -> float:
        """
        Calculate an execution-critical score for a CodeSymbol.
        Higher score means the symbol is more critical to execution and reasoning.
        """
        score = 0.0

        # Route decorators (HTTP endpoints are highly critical)
        for decorator in sym.decorators:
            dec_lower = decorator.lower()
            if any(kw in dec_lower for kw in ("route", "get", "post", "put", "delete", "patch")):
                score += 2.0
            else:
                score += 0.5

        # Core naming patterns
        name_lower = sym.name.lower()
        critical_patterns = {
            "authenticate", "login", "authorize", "jwt", "token", "session",
            "execute", "process", "handle", "dispatch", "validate", "verify",
            "commit", "transaction", "save", "create", "delete", "update",
            "run", "main", "start", "router", "endpoint"
        }
        for pattern in critical_patterns:
            if pattern in name_lower:
                score += 1.0

        # Out-degree complexity
        if sym.calls:
            score += min(1.5, len(sym.calls) * 0.3)

        # In-degree complexity
        if sym.callers:
            score += min(1.0, len(sym.callers) * 0.2)

        if sym.docstring and len(sym.docstring.strip()) > 10:
            score += 0.5

        if sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
            score += 0.3
        elif sym.symbol_type == SymbolType.CLASS:
            score += 0.2

        return score

    def _classify_into_bins(
        self,
        results: list[RetrievalResult],
        intent: QueryIntent,
        target_symbol: Optional[str],
    ) -> dict:
        """
        Classify retrieved symbols into specialized execution bins:
        ENTRYPOINTS, HIGH LEVEL FLOW, ORCHESTRATORS, DEPENDENCIES, IMPLEMENTATION DETAILS.
        """
        assigned: set[str] = set()
        symbols_by_section: dict[str, list[CodeSymbol]] = {
            SECTION_ENTRY_POINTS: [],
            SECTION_HIGH_LEVEL_FLOW: [],
            SECTION_ORCHESTRATORS: [],
            SECTION_DEPENDENCIES: [],
            SECTION_IMPLEMENTATION_DETAILS: [],
        }

        # 1. Classify ENTRYPOINTS
        for r in results:
            sym = r.symbol
            if self._is_entry_point(sym, r):
                symbols_by_section[SECTION_ENTRY_POINTS].append(sym)
                assigned.add(sym.symbol_id)

        # SYMBOL_LOOKUP target goes directly to Entrypoints
        if intent == QueryIntent.SYMBOL_LOOKUP and target_symbol:
            for r in results:
                sym = r.symbol
                if sym.symbol_id not in assigned and sym.name.lower() == target_symbol.lower():
                    symbols_by_section[SECTION_ENTRY_POINTS].insert(0, sym)
                    assigned.add(sym.symbol_id)

        # Sort Entrypoints
        symbols_by_section[SECTION_ENTRY_POINTS].sort(key=self._execution_critical_score, reverse=True)

        # 2. Classify HIGH LEVEL FLOW using ExecutionFlowReconstructor
        flow_diagram = ""
        best_ep = symbols_by_section[SECTION_ENTRY_POINTS][0] if symbols_by_section[SECTION_ENTRY_POINTS] else None
        
        if best_ep:
            from app.graph.flow_reconstructor import ExecutionFlowReconstructor
            reconstructor = ExecutionFlowReconstructor(self.graph)
            best_ep_node_id = self.graph.get_node_id(best_ep.qualified_name)
            
            if best_ep_node_id:
                trace_data = reconstructor.trace_execution_flow(best_ep_node_id, max_depth=3)
                if "steps" in trace_data and trace_data["steps"]:
                    steps = trace_data["steps"]
                    
                    # Generate tree diagram
                    flow_diagram = "PROBABLE EXECUTION CALL TREE PATHWAY:\n"
                    for step in steps:
                        indent = "  " * step["depth"]
                        prefix = "└── " if step["depth"] > 0 else "-> "
                        role_str = f"[{step['role'].upper()}]"
                        
                        # Calculate visual confidence tag
                        conf = step.get("confidence", 1.0)
                        res_type = step.get("resolution_type", "direct")
                        if step["depth"] == 0:
                            conf_tag = "[VERIFIED ENTRYPOINT]"
                        elif res_type in ("direct", "import") and conf >= 0.99:
                            conf_tag = "[VERIFIED]"
                        else:
                            conf_tag = f"[HEURISTIC: {int(conf * 100)}%]"
                            
                        flow_diagram += f"{indent}{prefix}{role_str} {step['qualified_name']} {conf_tag} (in {step['file_path']})\n"
                    flow_diagram += "\n"

                    # Gather symbols in flow (except the entry point itself)
                    flow_qns = {step["qualified_name"] for step in steps if step["depth"] > 0}
                    for r in results:
                        sym = r.symbol
                        if sym.symbol_id not in assigned and sym.qualified_name in flow_qns:
                            symbols_by_section[SECTION_HIGH_LEVEL_FLOW].append(sym)
                            assigned.add(sym.symbol_id)

        # Sort High Level Flow
        symbols_by_section[SECTION_HIGH_LEVEL_FLOW].sort(key=self._execution_critical_score, reverse=True)

        # 3. Classify ORCHESTRATORS (High out-degree symbols, out_degree >= 2)
        for r in results:
            sym = r.symbol
            if sym.symbol_id in assigned:
                continue
                
            node_id = self.graph.get_node_id(sym.qualified_name)
            out_deg = 0
            if node_id and node_id in self.graph.graph:
                try:
                    out_deg = self.graph.graph.out_degree(node_id)
                except Exception:
                    pass

            if out_deg >= 2:
                symbols_by_section[SECTION_ORCHESTRATORS].append(sym)
                assigned.add(sym.symbol_id)

        symbols_by_section[SECTION_ORCHESTRATORS].sort(key=self._execution_critical_score, reverse=True)

        # 4. Classify DEPENDENCIES (High in-degree symbols, in_degree >= 2)
        for r in results:
            sym = r.symbol
            if sym.symbol_id in assigned:
                continue
                
            node_id = self.graph.get_node_id(sym.qualified_name)
            in_deg = 0
            if node_id and node_id in self.graph.graph:
                try:
                    in_deg = self.graph.graph.in_degree(node_id)
                except Exception:
                    pass

            if in_deg >= 2 or sym.symbol_type == SymbolType.CLASS:
                symbols_by_section[SECTION_DEPENDENCIES].append(sym)
                assigned.add(sym.symbol_id)

        symbols_by_section[SECTION_DEPENDENCIES].sort(key=self._execution_critical_score, reverse=True)

        # 5. Classify IMPLEMENTATION DETAILS (Everything else: helper utilities, leaf nodes)
        for r in results:
            sym = r.symbol
            if sym.symbol_id not in assigned:
                # Aggressively suppress/omit trivial utility helpers to prevent context pollution
                if self._is_low_value_helper(sym):
                    continue
                symbols_by_section[SECTION_IMPLEMENTATION_DETAILS].append(sym)
                assigned.add(sym.symbol_id)

        symbols_by_section[SECTION_IMPLEMENTATION_DETAILS].sort(key=self._execution_critical_score, reverse=True)

        return {
            "flow_diagram": flow_diagram,
            "symbols": symbols_by_section
        }

    def _is_entry_point(self, sym: CodeSymbol, result: RetrievalResult) -> bool:
        """Determine if a symbol qualifies as an entry point."""
        for decorator in sym.decorators:
            dec_lower = decorator.lower()
            for keyword in _ROUTE_DECORATOR_KEYWORDS:
                if keyword in dec_lower:
                    return True

        name_lower = sym.name.lower()
        entry_point_names = {"main", "run", "start", "execute", "handler", "entrypoint"}
        if name_lower in entry_point_names:
            return True

        if sym.symbol_type == SymbolType.CLASS:
            class_keywords = {"view", "controller", "handler", "router", "api", "endpoint"}
            for kw in class_keywords:
                if kw in name_lower:
                    return True

        # No callers in graph means externally called
        if not sym.callers and sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
            return True

        return False

    def _is_low_value_helper(self, sym: CodeSymbol) -> bool:
        """
        Identify low-value utility / helper leaf nodes.
        These are typically low out-degree, low in-degree leaves without decorators.
        """
        if sym.symbol_type not in (SymbolType.FUNCTION, SymbolType.METHOD):
            return False

        node_id = self.graph.get_node_id(sym.qualified_name)
        if not node_id:
            return True

        try:
            out_deg = self.graph.graph.out_degree(node_id) if node_id in self.graph.graph else 0
            in_deg = self.graph.graph.in_degree(node_id) if node_id in self.graph.graph else 0
        except Exception:
            out_deg = 0
            in_deg = 0

        # Helper criteria: low call complexity, not decorated, low degree
        if out_deg <= 1 and in_deg <= 1 and not sym.decorators:
            return True

        return False

    def _compress_source(self, source: str, docstring: Optional[str]) -> str:
        """Compress a helper symbol's source to show only signature and docstring."""
        lines = source.split("\n")
        signature_lines = []
        in_signature = True

        for line in lines:
            signature_lines.append(line)
            if in_signature:
                stripped = line.strip()
                if stripped.endswith(":"):
                    in_signature = False
                    break

        signature_text = "\n".join(signature_lines)

        if docstring:
            doc_lines = docstring.strip().split("\n")
            formatted_doc = "\n    ".join(doc_lines)
            signature_text += f'\n    """\n    {formatted_doc}\n    """'

        signature_text += "\n    # ... [implementation details body omitted for brevity]\n"
        return signature_text

    # ------------------------------------------------------------------
    # Section rendering
    # ------------------------------------------------------------------

    def _render_section(
        self,
        section_name: str,
        symbols: list[CodeSymbol],
        budget: int,
    ) -> tuple[str, list[str], bool]:
        """Render a section's symbols within a character budget."""
        if not symbols or budget < 50:
            return "", [], False

        per_symbol_budget = max(250, budget // max(len(symbols), 1))
        rendered_parts: list[str] = []
        included_ids: list[str] = []
        total = 0
        truncated = False

        for sym in symbols:
            if total >= budget:
                truncated = True
                break

            available = min(per_symbol_budget, budget - total)
            
            # Helper compression if classified under implementation details
            if section_name == SECTION_IMPLEMENTATION_DETAILS and self._is_low_value_helper(sym):
                text = self._render_symbol(sym, available, compress=True)
            else:
                text = self._render_symbol(sym, available, compress=False)

            if total + len(text) > budget:
                chars_left = budget - total
                if chars_left > 100:
                    text = text[:chars_left] + "\n# ... [budget exceeded]\n"
                    truncated = True
                else:
                    truncated = True
                    break

            rendered_parts.append(text)
            included_ids.append(sym.symbol_id)
            total += len(text)

        if not rendered_parts:
            return "", [], truncated

        section_text = "\n".join(rendered_parts)
        return section_text, included_ids, truncated

    def _render_symbol(self, sym: CodeSymbol, budget: int, compress: bool = False) -> str:
        """Render a single symbol for context inclusion."""
        header = (
            f"# [{sym.symbol_type.value.upper()}] {sym.qualified_name}"
            f" — {sym.file_path}:{sym.line_start}\n"
        )

        source = sym.source.strip() if sym.source else ""

        if not source:
            content = header
            if sym.docstring:
                content += f'"""{sym.docstring[:200]}"""\n'
            return content

        if compress:
            source = self._compress_source(source, sym.docstring)

        full_text = header + source + "\n"

        if len(full_text) <= budget:
            return full_text

        # Truncation fallback: keep signature + docstring + lines that fit
        lines = source.split("\n")
        result_lines: list[str] = [header.rstrip()]
        chars_used = len(header)
        omitted = 0

        for i, line in enumerate(lines):
            line_cost = len(line) + 1
            if chars_used + line_cost + 40 > budget:
                omitted = len(lines) - i
                break
            result_lines.append(line)
            chars_used += line_cost

        if omitted > 0:
            result_lines.append(f"# ... [{omitted} lines truncated]")

        return "\n".join(result_lines) + "\n"

    def _build_imports_section(
        self,
        results: list[RetrievalResult],
        included_ids: list[str],
    ) -> str:
        """Build a compact imports section from key source files."""
        import_lines_by_file: dict[str, set[str]] = {}
        included_set = set(included_ids)

        for result in results:
            sym = result.symbol
            if sym.symbol_id not in included_set:
                continue
            if not sym.source:
                continue

            file_path = sym.file_path
            if file_path not in import_lines_by_file:
                import_lines_by_file[file_path] = set()

            for line in sym.source.split("\n"):
                stripped = line.strip()
                if stripped.startswith("import ") or stripped.startswith("from "):
                    import_lines_by_file[file_path].add(stripped)

        if not import_lines_by_file:
            return ""

        parts: list[str] = []
        for file_path, imports in import_lines_by_file.items():
            if not imports:
                continue
            import_list = sorted(imports)[:8]
            parts.append(f"# {file_path}")
            parts.extend(import_list)
            parts.append("")

        if not parts:
            return ""

        return "\n".join(parts)

    def _deduplicate(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Deduplicate by symbol_id, keeping highest score."""
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            sid = r.symbol.symbol_id
            if sid not in seen or r.score > seen[sid].score:
                seen[sid] = r
        return sorted(seen.values(), key=lambda r: r.score, reverse=True)

    def _join_sections(self, sections: dict[str, str]) -> str:
        """Join section texts with headers."""
        parts: list[str] = []
        section_order = [
            SECTION_PROJECT_PURPOSE,
            SECTION_ENTRY_POINTS,
            SECTION_HIGH_LEVEL_FLOW,
            SECTION_ORCHESTRATORS,
            SECTION_DEPENDENCIES,
            SECTION_IMPLEMENTATION_DETAILS,
            SECTION_IMPORTS,
        ]
        for name in section_order:
            if name in sections and sections[name].strip():
                parts.append(f"\n### [{name}]\n")
                parts.append(sections[name])

        return "\n".join(parts)
