"""Diff utilities for comparing prompts and configurations."""

from __future__ import annotations

import difflib
from html import escape as html_escape
from typing import TypedDict


class DiffLine(TypedDict):
    """A single line in a diff."""

    type: str  # "add", "remove", "unchanged", "context"
    line_num_old: int | None
    line_num_new: int | None
    content: str


def compute_prompt_diff(old_prompt: str, new_prompt: str) -> list[DiffLine]:
    """Compute line-by-line diff between prompts.

    Args:
        old_prompt: The original prompt text
        new_prompt: The new prompt text

    Returns:
        List of DiffLine dicts with type, line numbers, and content
    """
    old_lines = old_prompt.splitlines(keepends=False)
    new_lines = new_prompt.splitlines(keepends=False)

    diff_result: list[DiffLine] = []

    # Use SequenceMatcher for more intelligent diff
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)

    old_line_num = 1
    new_line_num = 1

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Unchanged lines - show a few for context
            lines_to_show = old_lines[i1:i2]
            # Only show first and last context lines if too many
            if len(lines_to_show) > 3:
                # Show first line
                diff_result.append(
                    DiffLine(
                        type="context",
                        line_num_old=old_line_num,
                        line_num_new=new_line_num,
                        content=lines_to_show[0],
                    )
                )
                # Show ellipsis
                diff_result.append(
                    DiffLine(
                        type="context",
                        line_num_old=None,
                        line_num_new=None,
                        content=f"... ({len(lines_to_show) - 2} unchanged lines) ...",
                    )
                )
                # Show last line
                diff_result.append(
                    DiffLine(
                        type="context",
                        line_num_old=old_line_num + len(lines_to_show) - 1,
                        line_num_new=new_line_num + len(lines_to_show) - 1,
                        content=lines_to_show[-1],
                    )
                )
            else:
                for line in lines_to_show:
                    diff_result.append(
                        DiffLine(
                            type="context",
                            line_num_old=old_line_num,
                            line_num_new=new_line_num,
                            content=line,
                        )
                    )
                    old_line_num += 1
                    new_line_num += 1
            old_line_num = i2 + 1
            new_line_num = j2 + 1

        elif tag == "delete":
            # Lines removed
            for line in old_lines[i1:i2]:
                diff_result.append(
                    DiffLine(
                        type="remove",
                        line_num_old=old_line_num,
                        line_num_new=None,
                        content=line,
                    )
                )
                old_line_num += 1

        elif tag == "insert":
            # Lines added
            for line in new_lines[j1:j2]:
                diff_result.append(
                    DiffLine(
                        type="add",
                        line_num_old=None,
                        line_num_new=new_line_num,
                        content=line,
                    )
                )
                new_line_num += 1

        elif tag == "replace":
            # Lines changed - show as remove then add
            for line in old_lines[i1:i2]:
                diff_result.append(
                    DiffLine(
                        type="remove",
                        line_num_old=old_line_num,
                        line_num_new=None,
                        content=line,
                    )
                )
                old_line_num += 1
            for line in new_lines[j1:j2]:
                diff_result.append(
                    DiffLine(
                        type="add",
                        line_num_old=None,
                        line_num_new=new_line_num,
                        content=line,
                    )
                )
                new_line_num += 1

    return diff_result


def format_diff_for_markdown(diff: list[DiffLine]) -> str:
    """Format diff as Markdown-compatible diff block with +/- syntax.

    Args:
        diff: List of DiffLine objects from compute_prompt_diff

    Returns:
        String formatted for markdown ```diff code block
    """
    lines: list[str] = []
    for item in diff:
        if item["type"] == "add":
            lines.append(f"+ {item['content']}")
        elif item["type"] == "remove":
            lines.append(f"- {item['content']}")
        elif item["type"] == "context":
            lines.append(f"  {item['content']}")

    return "\n".join(lines)


def format_diff_for_html(diff: list[DiffLine]) -> str:
    """Format diff as styled HTML with red/green highlighting.

    Args:
        diff: List of DiffLine objects from compute_prompt_diff

    Returns:
        HTML string with styled diff lines
    """
    lines: list[str] = []
    for item in diff:
        content = html_escape(item["content"])
        if item["type"] == "add":
            lines.append(f'<div class="diff-add">+ {content}</div>')
        elif item["type"] == "remove":
            lines.append(f'<div class="diff-remove">- {content}</div>')
        elif item["type"] == "context":
            lines.append(f'<div class="diff-context">  {content}</div>')

    return "\n".join(lines)


def has_changes(diff: list[DiffLine]) -> bool:
    """Check if diff contains any actual changes.

    Args:
        diff: List of DiffLine objects

    Returns:
        True if there are add/remove lines, False if only context
    """
    return any(item["type"] in ("add", "remove") for item in diff)
