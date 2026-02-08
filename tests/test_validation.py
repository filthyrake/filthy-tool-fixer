"""Tests for schema validation."""

from __future__ import annotations

import json

import pytest

from filthy_tool_fixer.models import FunctionCall, ToolCall
from filthy_tool_fixer.validation.schema import validate_tool_calls

from tests.helpers import make_tool_call


class TestToolNameValidation:
    def test_valid_tool_name(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": "test"})]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid

    def test_unknown_tool_name(self, sample_tools):
        calls = [make_tool_call("serch_files", {"query": "test"})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "unknown_tool"
        assert "search_files" in result.errors[0].message  # fuzzy match suggestion

    def test_completely_wrong_tool_name(self, sample_tools):
        calls = [make_tool_call("do_magic", {"query": "test"})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "unknown_tool"

    def test_empty_tool_calls(self, sample_tools):
        result = validate_tool_calls([], sample_tools)
        assert result.valid


class TestArgumentValidation:
    def test_valid_required_args(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": "test"})]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid

    def test_valid_all_args(self, sample_tools):
        calls = [
            make_tool_call(
                "search_files",
                {"query": "test", "max_results": 10, "include_hidden": True},
            )
        ]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid

    def test_missing_required_arg(self, sample_tools):
        calls = [make_tool_call("search_files", {"max_results": 10})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert any(e.error_type == "missing_required" for e in result.errors)

    def test_wrong_type_arg(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": "test", "max_results": "ten"})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert any(e.error_type == "wrong_type" for e in result.errors)

    def test_extra_hallucinated_field(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": "test", "recursive": True})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert any(e.error_type == "extra_field" for e in result.errors)
        assert "recursive" in result.errors[0].message

    def test_multiple_required_args(self, sample_tools):
        calls = [make_tool_call("write_file", {"path": "/tmp/test.txt", "content": "hello"})]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid

    def test_missing_multiple_required_args(self, sample_tools):
        calls = [make_tool_call("write_file", {})]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        missing = [e for e in result.errors if e.error_type == "missing_required"]
        assert len(missing) == 2


class TestJSONParsing:
    def test_invalid_json(self, sample_tools):
        calls = [make_tool_call("search_files", "not json at all")]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "invalid_json"

    def test_json_array_instead_of_object(self, sample_tools):
        calls = [make_tool_call("search_files", '["query", "test"]')]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "invalid_json"

    def test_empty_json_object(self, sample_tools):
        # search_files requires "query", so empty object should fail
        calls = [make_tool_call("search_files", "{}")]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid


class TestParallelToolCalls:
    def test_all_valid_parallel(self, sample_tools):
        calls = [
            make_tool_call("search_files", {"query": "test"}, "call_1"),
            make_tool_call("write_file", {"path": "/tmp/x", "content": "y"}, "call_2"),
        ]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid

    def test_one_invalid_rejects_all(self, sample_tools):
        calls = [
            make_tool_call("search_files", {"query": "test"}, "call_1"),
            make_tool_call("write_file", {"path": "/tmp/x"}, "call_2"),  # missing content
        ]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        # The error is on the second call
        assert result.errors[0].tool_call_index == 1


class TestInputLimits:
    def test_rejects_too_many_tool_calls(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": f"test{i}"}, call_id=f"call_{i}")
                 for i in range(51)]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "too_many_calls"

    def test_rejects_oversized_arguments(self, sample_tools):
        huge_args = json.dumps({"query": "x" * 1_000_001})
        calls = [ToolCall(id="call_1", type="function",
                          function=FunctionCall(name="search_files", arguments=huge_args))]
        result = validate_tool_calls(calls, sample_tools)
        assert not result.valid
        assert result.errors[0].error_type == "arguments_too_large"

    def test_accepts_max_tool_calls(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": f"test{i}"}, call_id=f"call_{i}")
                 for i in range(50)]
        result = validate_tool_calls(calls, sample_tools)
        assert result.valid


class TestErrorSeverity:
    def test_unknown_tool_is_worst(self, sample_tools):
        calls = [make_tool_call("fake_tool", {"query": "test"})]
        result = validate_tool_calls(calls, sample_tools)
        assert result.worst_error is not None
        assert result.worst_error.severity == 3

    def test_missing_required_is_medium(self, sample_tools):
        calls = [make_tool_call("search_files", {})]
        result = validate_tool_calls(calls, sample_tools)
        worst = result.worst_error
        assert worst is not None
        assert worst.severity == 2

    def test_extra_field_is_low(self, sample_tools):
        calls = [make_tool_call("search_files", {"query": "test", "bogus": 1})]
        result = validate_tool_calls(calls, sample_tools)
        worst = result.worst_error
        assert worst is not None
        assert worst.severity == 1
