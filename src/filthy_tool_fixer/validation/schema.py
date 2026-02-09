"""Validate tool calls against tool definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from difflib import get_close_matches

import jsonschema

from filthy_tool_fixer.models import ToolCall, ToolDefinition

_MAX_TOOL_CALLS = 50
_MAX_ARGUMENTS_SIZE = 1_000_000  # 1MB


@dataclass
class ValidationError:
    tool_call_index: int
    error_type: str  # "unknown_tool", "invalid_json", "missing_required", "wrong_type", "extra_field"
    message: str
    severity: int = 0  # Higher = worse. unknown_tool=3, wrong_type=2, extra_field=1


@dataclass
class ValidationResult:
    valid: bool
    errors: list[ValidationError] = field(default_factory=list)

    @property
    def worst_error(self) -> ValidationError | None:
        if not self.errors:
            return None
        return max(self.errors, key=lambda e: e.severity)


def validate_tool_calls(
    tool_calls: list[ToolCall],
    tool_definitions: list[ToolDefinition],
) -> ValidationResult:
    """Validate a list of tool calls against the provided tool definitions.

    For parallel tool calls, rejects entire set if any single call is invalid.
    """
    if not tool_calls:
        return ValidationResult(valid=True)

    if len(tool_calls) > _MAX_TOOL_CALLS:
        return ValidationResult(valid=False, errors=[
            ValidationError(0, "too_many_calls", f"Too many tool calls ({len(tool_calls)}), max {_MAX_TOOL_CALLS}.", severity=3),
        ])

    tool_map: dict[str, ToolDefinition] = {t.function.name: t for t in tool_definitions}
    tool_names = list(tool_map.keys())
    errors: list[ValidationError] = []

    for idx, call in enumerate(tool_calls):
        if len(call.function.arguments) > _MAX_ARGUMENTS_SIZE:
            errors.append(ValidationError(
                idx, "arguments_too_large",
                f"Tool '{call.function.name}' arguments exceed {_MAX_ARGUMENTS_SIZE} bytes.",
                severity=3,
            ))
            continue
        call_errors = _validate_single_call(idx, call, tool_map, tool_names)
        errors.extend(call_errors)

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def _validate_single_call(
    idx: int,
    call: ToolCall,
    tool_map: dict[str, ToolDefinition],
    tool_names: list[str],
) -> list[ValidationError]:
    """Validate a single tool call."""
    errors: list[ValidationError] = []
    func_name = call.function.name

    # Check tool name exists
    if func_name not in tool_map:
        suggestion = ""
        matches = get_close_matches(func_name, tool_names, n=1, cutoff=0.6)
        if matches:
            suggestion = f" Did you mean '{matches[0]}'?"
        errors.append(
            ValidationError(
                tool_call_index=idx,
                error_type="unknown_tool",
                message=f"Tool '{func_name}' does not exist.{suggestion}",
                severity=3,
            )
        )
        return errors  # Can't validate args for unknown tool

    tool_def = tool_map[func_name]
    schema = tool_def.function.parameters

    # Parse arguments JSON
    try:
        args = json.loads(call.function.arguments)
    except (json.JSONDecodeError, TypeError) as e:
        errors.append(
            ValidationError(
                tool_call_index=idx,
                error_type="invalid_json",
                message=f"Tool '{func_name}' arguments are not valid JSON: {e}",
                severity=3,
            )
        )
        return errors  # Can't validate further

    if not isinstance(args, dict):
        errors.append(
            ValidationError(
                tool_call_index=idx,
                error_type="invalid_json",
                message=f"Tool '{func_name}' arguments must be a JSON object, got {type(args).__name__}.",
                severity=3,
            )
        )
        return errors

    # Check for extra fields (hallucinated parameters)
    if schema and "properties" in schema:
        allowed = set(schema["properties"].keys())
        extra = set(args.keys()) - allowed
        for field_name in extra:
            errors.append(
                ValidationError(
                    tool_call_index=idx,
                    error_type="extra_field",
                    message=(
                        f"Tool '{func_name}' does not accept parameter '{field_name}'. "
                        f"Valid parameters: {sorted(allowed)}"
                    ),
                    severity=1,
                )
            )

    # Check required arguments
    required = schema.get("required", []) if schema else []
    for req_field in required:
        if req_field not in args:
            errors.append(
                ValidationError(
                    tool_call_index=idx,
                    error_type="missing_required",
                    message=f"Tool '{func_name}' requires parameter '{req_field}'.",
                    severity=2,
                )
            )

    # Validate types via jsonschema
    if schema:
        validator = jsonschema.Draft7Validator(schema)
        for error in validator.iter_errors(args):
            # Skip errors we already reported (required, additionalProperties)
            if error.validator in ("required", "additionalProperties"):
                continue
            errors.append(
                ValidationError(
                    tool_call_index=idx,
                    error_type="wrong_type",
                    message=f"Tool '{func_name}': {error.message}",
                    severity=2,
                )
            )

    return errors
