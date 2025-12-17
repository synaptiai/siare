"""ASCII graph visualization for SOP role graphs."""

from __future__ import annotations

from collections import defaultdict
from html import escape as html_escape
from typing import Any, cast


def _build_ascii_adjacency(
    edges: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], set[str], set[str]]:
    """Build adjacency list and track all nodes.

    Args:
        edges: List of edge dicts with 'from_' and 'to' keys

    Returns:
        Tuple of (adjacency dict, all_nodes set, has_incoming set)
    """
    adjacency: dict[str, list[str]] = defaultdict(list)
    all_nodes: set[str] = set()
    has_incoming: set[str] = set()

    for edge in edges:
        from_node = edge.get("from_") or edge.get("from", "")
        to_node = edge.get("to", "")
        if isinstance(from_node, list):
            from_list = cast("list[Any]", from_node)
            from_node = ",".join(sorted(str(n) for n in from_list))
        if from_node and to_node:
            adjacency[from_node].append(to_node)
            all_nodes.add(from_node)
            all_nodes.add(to_node)
            has_incoming.add(to_node)

    return adjacency, all_nodes, has_incoming


def _find_ascii_roots(
    edges: list[dict[str, Any]], all_nodes: set[str], has_incoming: set[str]
) -> set[str]:
    """Find root nodes (nodes with no incoming edges).

    Args:
        edges: Original edges list for fallback
        all_nodes: Set of all node names
        has_incoming: Set of nodes that have incoming edges

    Returns:
        Set of root node names
    """
    roots = all_nodes - has_incoming
    if not roots:
        roots = {edges[0].get("from_") or edges[0].get("from", "start")}
    return roots


def _format_ascii_node(node: str, role_tools: dict[str, list[str]]) -> str:
    """Format a node with its tools if available.

    Args:
        node: Node name
        role_tools: Lookup dict mapping node names to tool lists

    Returns:
        Formatted node string with tools
    """
    tools = role_tools.get(node, [])
    if tools:
        return f"{node} [{', '.join(tools)}]"
    return node


def _render_ascii_subtree(
    node: str,
    adjacency: dict[str, list[str]],
    role_tools: dict[str, list[str]],
    visited: set[str],
    lines: list[str],
    prefix: str = "",
    is_last: bool = True,
) -> None:
    """Render a node and its children recursively.

    Args:
        node: Current node to render
        adjacency: Graph adjacency dict
        role_tools: Lookup dict for node tools
        visited: Set of already visited nodes (modified in place)
        lines: Output lines list (modified in place)
        prefix: Current line prefix for indentation
        is_last: Whether this is the last child of its parent
    """
    if node in visited:
        lines.append(f"{prefix}{_format_ascii_node(node, role_tools)} (cycle)")
        return

    visited.add(node)
    lines.append(f"{prefix}{_format_ascii_node(node, role_tools)}")

    children = adjacency.get(node, [])
    for i, child in enumerate(children):
        is_child_last = i == len(children) - 1
        child_prefix = prefix + ("    " if is_last else "|   ")
        lines.append(f"{child_prefix}|")
        lines.append(f"{child_prefix}v")
        _render_ascii_subtree(
            child, adjacency, role_tools, visited, lines, child_prefix, is_child_last
        )


def generate_ascii_graph(
    edges: list[dict[str, Any]],
    roles: list[dict[str, Any]] | None = None,
) -> str:
    """Generate ASCII representation of role graph.

    Args:
        edges: List of edge dicts with 'from_' and 'to' keys
        roles: Optional list of role dicts with 'id' and 'tools' keys

    Returns:
        ASCII string representation of the graph

    Example output:
        user_input
            |
            v
        retriever [vector_search]
            |
            v
        answerer
    """
    if not edges:
        return "(empty graph)"

    adjacency, all_nodes, has_incoming = _build_ascii_adjacency(edges)
    roots = _find_ascii_roots(edges, all_nodes, has_incoming)
    role_tools = _build_role_tools_lookup(roles)

    lines: list[str] = []
    visited: set[str] = set()

    for i, root in enumerate(sorted(roots)):
        if i > 0:
            lines.append("")
        _render_ascii_subtree(root, adjacency, role_tools, visited, lines)

    return "\n".join(lines)


def generate_html_graph(
    edges: list[dict[str, Any]],
    roles: list[dict[str, Any]] | None = None,
) -> str:
    """Generate HTML representation of role graph.

    Args:
        edges: List of edge dicts with 'from_' and 'to' keys
        roles: Optional list of role dicts with 'id' and 'tools' keys

    Returns:
        HTML string with styled graph
    """
    ascii_graph = generate_ascii_graph(edges, roles)
    escaped = html_escape(ascii_graph)
    return f'<pre class="graph-visual">{escaped}</pre>'


def _build_role_tools_lookup(roles: list[dict[str, Any]] | None) -> dict[str, list[str]]:
    """Build a lookup dict mapping role IDs to their tools.

    Args:
        roles: Optional list of role dicts with 'id' and 'tools' keys

    Returns:
        Dict mapping role_id to list of tool names
    """
    if not roles:
        return {}

    role_tools: dict[str, list[str]] = {}
    for role in roles:
        role_id = role.get("id", "")
        tools = role.get("tools", [])
        if role_id:
            role_tools[role_id] = tools if tools else []
    return role_tools


def _normalize_from_node(from_node: str | list[Any]) -> str:
    """Normalize from_node, handling list inputs.

    Args:
        from_node: Either a string node name or list of node names

    Returns:
        Normalized string representation
    """
    if isinstance(from_node, list):
        return ",".join(sorted(str(n) for n in from_node))
    return from_node


def _add_node_label_if_new(
    node: str,
    seen_nodes: set[str],
    role_tools: dict[str, list[str]],
    lines: list[str],
) -> None:
    """Add node label with tools to lines if not already seen.

    Args:
        node: Node name to add
        seen_nodes: Set of already-seen nodes (modified in place)
        role_tools: Lookup dict for node tools
        lines: Output lines list (modified in place)
    """
    if node in seen_nodes:
        return

    tools = role_tools.get(node, [])
    if tools:
        lines.append(f"    {node}[{node}<br/>{', '.join(tools)}]")
    seen_nodes.add(node)


def _format_edge_line(from_node: str, to_node: str, condition: str | None) -> str:
    """Format a Mermaid edge line with optional condition.

    Args:
        from_node: Source node name
        to_node: Target node name
        condition: Optional condition label

    Returns:
        Formatted Mermaid edge line
    """
    if condition:
        return f"    {from_node} -->|{condition}| {to_node}"
    return f"    {from_node} --> {to_node}"


def generate_mermaid_graph(
    edges: list[dict[str, Any]],
    roles: list[dict[str, Any]] | None = None,
) -> str:
    """Generate Mermaid diagram syntax for role graph.

    Args:
        edges: List of edge dicts with 'from_' and 'to' keys
        roles: Optional list of role dicts with 'id' and 'tools' keys

    Returns:
        Mermaid diagram syntax string

    Example output:
        ```mermaid
        flowchart TD
            user_input --> retriever
            retriever --> answerer
        ```
    """
    if not edges:
        return "```mermaid\nflowchart TD\n    empty[No edges]\n```"

    lines = ["```mermaid", "flowchart TD"]
    role_tools = _build_role_tools_lookup(roles)
    seen_nodes: set[str] = set()

    for edge in edges:
        from_node = _normalize_from_node(edge.get("from_") or edge.get("from", ""))
        to_node = edge.get("to", "")

        if from_node and to_node:
            _add_node_label_if_new(from_node, seen_nodes, role_tools, lines)
            _add_node_label_if_new(to_node, seen_nodes, role_tools, lines)
            lines.append(_format_edge_line(from_node, to_node, edge.get("condition")))

    lines.append("```")
    return "\n".join(lines)


def summarize_graph_changes(
    old_edges: list[dict[str, Any]],
    new_edges: list[dict[str, Any]],
) -> dict[str, list[tuple[str, str]]]:
    """Summarize changes between two graphs.

    Args:
        old_edges: Previous graph edges
        new_edges: New graph edges

    Returns:
        Dict with 'added' and 'removed' edge tuples
    """
    old_set = {
        (e.get("from_") or e.get("from", ""), e.get("to", "")) for e in old_edges
    }
    new_set = {
        (e.get("from_") or e.get("from", ""), e.get("to", "")) for e in new_edges
    }

    return {
        "added": list(new_set - old_set),
        "removed": list(old_set - new_set),
    }
