"""Tests for the schema validator."""

import pytest

from launchpromptly._internal.schema_validator import (
    validate_schema,
    validate_output_schema,
    OutputSchemaOptions,
    SchemaValidationError,
)


# ── Type validation ──────────────────────────────────────────────────────────


class TestTypeValidation:
    def test_string_type(self):
        assert len(validate_schema("hello", {"type": "string"})) == 0
        assert len(validate_schema(42, {"type": "string"})) == 1

    def test_number_type(self):
        assert len(validate_schema(42, {"type": "number"})) == 0
        assert len(validate_schema(3.14, {"type": "number"})) == 0
        assert len(validate_schema("42", {"type": "number"})) == 1

    def test_integer_type(self):
        assert len(validate_schema(42, {"type": "integer"})) == 0
        assert len(validate_schema(3.14, {"type": "integer"})) == 1

    def test_boolean_type(self):
        assert len(validate_schema(True, {"type": "boolean"})) == 0
        assert len(validate_schema("true", {"type": "boolean"})) == 1

    def test_array_type(self):
        assert len(validate_schema([1, 2], {"type": "array"})) == 0
        assert len(validate_schema("not array", {"type": "array"})) == 1

    def test_object_type(self):
        assert len(validate_schema({"a": 1}, {"type": "object"})) == 0
        assert len(validate_schema([1], {"type": "object"})) == 1

    def test_null_type(self):
        assert len(validate_schema(None, {"type": "null"})) == 0
        assert len(validate_schema("null", {"type": "null"})) == 1

    def test_union_types(self):
        assert len(validate_schema("hello", {"type": ["string", "null"]})) == 0
        assert len(validate_schema(None, {"type": ["string", "null"]})) == 0
        assert len(validate_schema(42, {"type": ["string", "null"]})) == 1


# ── Required fields ──────────────────────────────────────────────────────────


class TestRequiredFields:
    def test_missing_required(self):
        schema = {
            "type": "object",
            "required": ["name", "score"],
            "properties": {
                "name": {"type": "string"},
                "score": {"type": "number"},
            },
        }
        errors = validate_schema({"name": "test"}, schema)
        assert len(errors) == 1
        assert errors[0].path == "/score"
        assert "Missing required" in errors[0].message

    def test_all_required_present(self):
        schema = {
            "type": "object",
            "required": ["name"],
            "properties": {"name": {"type": "string"}},
        }
        assert len(validate_schema({"name": "test"}, schema)) == 0


# ── Nested objects ───────────────────────────────────────────────────────────


class TestNestedObjects:
    def test_nested_property_types(self):
        schema = {
            "type": "object",
            "properties": {
                "user": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
            },
        }
        errors = validate_schema({"user": {"name": "test"}}, schema)
        assert len(errors) == 1
        assert errors[0].path == "/user/id"


# ── Enum validation ──────────────────────────────────────────────────────────


class TestEnumValidation:
    def test_valid_enum(self):
        assert len(validate_schema("red", {"enum": ["red", "green", "blue"]})) == 0

    def test_invalid_enum(self):
        errors = validate_schema("yellow", {"enum": ["red", "green", "blue"]})
        assert len(errors) == 1
        assert "must be one of" in errors[0].message


# ── String constraints ───────────────────────────────────────────────────────


class TestStringConstraints:
    def test_min_length(self):
        assert len(validate_schema("ab", {"type": "string", "minLength": 3})) == 1
        assert len(validate_schema("abc", {"type": "string", "minLength": 3})) == 0

    def test_max_length(self):
        assert len(validate_schema("abcd", {"type": "string", "maxLength": 3})) == 1
        assert len(validate_schema("abc", {"type": "string", "maxLength": 3})) == 0

    def test_pattern(self):
        assert len(validate_schema("abc123", {"type": "string", "pattern": "^[a-z]+$"})) == 1
        assert len(validate_schema("abc", {"type": "string", "pattern": "^[a-z]+$"})) == 0


# ── Number constraints ───────────────────────────────────────────────────────


class TestNumberConstraints:
    def test_minimum(self):
        assert len(validate_schema(5, {"type": "number", "minimum": 10})) == 1
        assert len(validate_schema(10, {"type": "number", "minimum": 10})) == 0

    def test_maximum(self):
        assert len(validate_schema(15, {"type": "number", "maximum": 10})) == 1
        assert len(validate_schema(10, {"type": "number", "maximum": 10})) == 0


# ── Array validation ─────────────────────────────────────────────────────────


class TestArrayValidation:
    def test_items_schema(self):
        schema = {"type": "array", "items": {"type": "string"}}
        assert len(validate_schema(["a", "b"], schema)) == 0
        assert len(validate_schema(["a", 42], schema)) == 1

    def test_min_items(self):
        assert len(validate_schema([], {"type": "array", "minItems": 1})) == 1
        assert len(validate_schema([1], {"type": "array", "minItems": 1})) == 0

    def test_max_items(self):
        assert len(validate_schema([1, 2, 3], {"type": "array", "maxItems": 2})) == 1
        assert len(validate_schema([1, 2], {"type": "array", "maxItems": 2})) == 0


# ── Additional properties ────────────────────────────────────────────────────


class TestAdditionalProperties:
    def test_blocks_extra_when_false(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        assert len(validate_schema({"name": "test", "extra": 1}, schema)) == 1
        assert len(validate_schema({"name": "test"}, schema)) == 0


# ── Combinators ──────────────────────────────────────────────────────────────


class TestCombinators:
    def test_any_of(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "number"}]}
        assert len(validate_schema("hello", schema)) == 0
        assert len(validate_schema(42, schema)) == 0
        assert len(validate_schema(True, schema)) == 1

    def test_all_of(self):
        schema = {
            "allOf": [
                {"type": "object", "required": ["name"]},
                {"type": "object", "required": ["score"]},
            ],
        }
        assert len(validate_schema({"name": "a", "score": 1}, schema)) == 0
        assert len(validate_schema({"name": "a"}, schema)) == 1


# ── validateOutputSchema ─────────────────────────────────────────────────────


class TestValidateOutputSchema:
    def test_valid_json(self):
        opts = OutputSchemaOptions(
            schema={
                "type": "object",
                "required": ["name", "score"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number", "minimum": 0, "maximum": 100},
                },
            }
        )
        result = validate_output_schema('{"name": "test", "score": 85}', opts)
        assert result.valid is True
        assert len(result.errors) == 0
        assert result.parsed == {"name": "test", "score": 85}

    def test_invalid_json(self):
        result = validate_output_schema(
            "not json at all",
            OutputSchemaOptions(schema={"type": "object"}),
        )
        assert result.valid is False
        assert "Invalid JSON" in result.errors[0].message

    def test_schema_violations(self):
        opts = OutputSchemaOptions(
            schema={
                "type": "object",
                "required": ["name", "score"],
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
            }
        )
        result = validate_output_schema('{"name": "test"}', opts)
        assert result.valid is False
        assert len(result.errors) == 1

    def test_on_invalid_callback(self):
        captured: list = []
        opts = OutputSchemaOptions(
            schema={"type": "object", "required": ["id"]},
            on_invalid=lambda e: captured.extend(e),
        )
        validate_output_schema("{}", opts)
        assert len(captured) > 0

    def test_no_callback_on_success(self):
        called = []
        opts = OutputSchemaOptions(
            schema={"type": "object"},
            on_invalid=lambda e: called.append(True),
        )
        validate_output_schema("{}", opts)
        assert len(called) == 0
