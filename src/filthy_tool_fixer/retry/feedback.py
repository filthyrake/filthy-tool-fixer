"""Error feedback message construction for retry loop."""

from __future__ import annotations

from filthy_tool_fixer.models import ChatMessage
from filthy_tool_fixer.validation.schema import ValidationError, ValidationResult


def build_feedback_message(result: ValidationResult) -> ChatMessage:
    """Build a clear, actionable error feedback message for the LLM.

    Prioritizes the most critical error and keeps the message concise
    to avoid confusing the model with too much information.
    """
    if result.valid or not result.errors:
        raise ValueError("Cannot build feedback for valid result")

    worst = result.worst_error
    assert worst is not None

    # If there's only one error, just report it
    if len(result.errors) == 1:
        content = (
            f"Your tool call had an error: {worst.message}\n"
            "Please fix this and try again."
        )
    else:
        # Report the worst error prominently, mention the count of others
        other_count = len(result.errors) - 1
        other_word = "error" if other_count == 1 else "errors"
        content = (
            f"Your tool call had {len(result.errors)} errors. "
            f"Most critical: {worst.message}\n"
        )
        # Add one more error for context if there are just a few
        if other_count <= 3:
            others = [e for e in result.errors if e is not worst]
            content += "Other issues:\n"
            for e in others:
                content += f"- {e.message}\n"
        else:
            content += f"Plus {other_count} other {other_word}.\n"

        content += "Please fix all issues and try again."

    return ChatMessage(role="user", content=content)
