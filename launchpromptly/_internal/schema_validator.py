"""Zero-dependency JSON schema validator (draft-07 subset).

Validates LLM output against a user-provided schema.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union


@dataclass
class SchemaValidationError:
    """A single validation error."""

    path: str
    message: str
    expected: Optional[str] = None
    actual: Any = None


@dataclass
class OutputSchemaOptions:
    """Configuration for output schema validation."""

    schema: Dict[str, Any]
    block_on_invalid: bool = False
    on_invalid: Optional[Callable[[List[SchemaValidationError]], None]] = None


@dataclass
class OutputValidationResult:
    """Result of schema validation."""

    valid: bool
    errors: List[SchemaValidationError] = field(default_factory=list)
    parsed: Any = None


def validate_schema(
    value: Any,
    schema: Dict[str, Any],
    path: str = "",
) -> List[SchemaValidationError]:
    """Validate a value against a JSON schema.

    Returns a list of validation errors (empty = valid).
    """
    errors: List[SchemaValidationError] = []

    # Type validation
    schema_type = schema.get("type")
    if schema_type is not None:
        types = schema_type if isinstance(schema_type, list) else [schema_type]
        actual_type = _get_json_type(value)
        # In JSON Schema, "number" accepts both integers and floats
        type_matches = actual_type in types or (actual_type == "integer" and "number" in types)
        if not type_matches:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Expected type {' | '.join(types)}, got {actual_type}",
                expected=" | ".join(types),
                actual=actual_type,
            ))
            return errors  # Type mismatch — skip further validation

    # Const validation
    if "const" in schema:
        if not _deep_equal(value, schema["const"]):
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Expected constant {json.dumps(schema['const'])}, got {json.dumps(value)}",
                expected=json.dumps(schema["const"]),
                actual=value,
            ))

    # Enum validation
    if "enum" in schema:
        if not any(_deep_equal(e, value) for e in schema["enum"]):
            enum_str = ", ".join(json.dumps(e) for e in schema["enum"])
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Value must be one of: {enum_str}",
                expected=" | ".join(json.dumps(e) for e in schema["enum"]),
                actual=value,
            ))

    # String validations
    if isinstance(value, str):
        min_len = schema.get("minLength")
        if min_len is not None and len(value) < min_len:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"String length {len(value)} is less than minimum {min_len}",
                expected=f"minLength: {min_len}",
                actual=len(value),
            ))
        max_len = schema.get("maxLength")
        if max_len is not None and len(value) > max_len:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"String length {len(value)} exceeds maximum {max_len}",
                expected=f"maxLength: {max_len}",
                actual=len(value),
            ))
        pattern = schema.get("pattern")
        if pattern is not None:
            try:
                if not re.search(pattern, value):
                    errors.append(SchemaValidationError(
                        path=path or "/",
                        message=f"String does not match pattern: {pattern}",
                        expected=f"pattern: {pattern}",
                        actual=value,
                    ))
            except re.error:
                pass  # Invalid regex in schema — skip

    # Number validations
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        if minimum is not None and value < minimum:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Value {value} is less than minimum {minimum}",
                expected=f"minimum: {minimum}",
                actual=value,
            ))
        maximum = schema.get("maximum")
        if maximum is not None and value > maximum:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Value {value} exceeds maximum {maximum}",
                expected=f"maximum: {maximum}",
                actual=value,
            ))

    # Object validations
    if isinstance(value, dict):
        # Required fields
        required = schema.get("required")
        if required:
            for key in required:
                if key not in value:
                    errors.append(SchemaValidationError(
                        path=f"{path}/{key}",
                        message=f"Missing required field: {key}",
                        expected="required",
                    ))

        # Properties
        properties = schema.get("properties")
        if properties:
            for key, prop_schema in properties.items():
                if key in value:
                    errors.extend(validate_schema(value[key], prop_schema, f"{path}/{key}"))

        # Additional properties
        additional = schema.get("additionalProperties")
        if additional is not None and properties:
            allowed_keys = set(properties.keys())
            for key in value:
                if key not in allowed_keys:
                    if additional is False:
                        errors.append(SchemaValidationError(
                            path=f"{path}/{key}",
                            message=f"Additional property not allowed: {key}",
                        ))
                    elif isinstance(additional, dict):
                        errors.extend(validate_schema(value[key], additional, f"{path}/{key}"))

    # Array validations
    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Array length {len(value)} is less than minimum {min_items}",
                expected=f"minItems: {min_items}",
                actual=len(value),
            ))
        max_items = schema.get("maxItems")
        if max_items is not None and len(value) > max_items:
            errors.append(SchemaValidationError(
                path=path or "/",
                message=f"Array length {len(value)} exceeds maximum {max_items}",
                expected=f"maxItems: {max_items}",
                actual=len(value),
            ))
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(value):
                errors.extend(validate_schema(item, items_schema, f"{path}/{i}"))

    # Combinators
    all_of = schema.get("allOf")
    if all_of:
        for sub in all_of:
            errors.extend(validate_schema(value, sub, path))

    any_of = schema.get("anyOf")
    if any_of:
        any_valid = any(len(validate_schema(value, sub, path)) == 0 for sub in any_of)
        if not any_valid:
            errors.append(SchemaValidationError(
                path=path or "/",
                message="Value does not match any of the anyOf schemas",
            ))

    one_of = schema.get("oneOf")
    if one_of:
        matching = [sub for sub in one_of if len(validate_schema(value, sub, path)) == 0]
        if len(matching) != 1:
            msg = (
                "Value does not match any oneOf schema"
                if len(matching) == 0
                else f"Value matches {len(matching)} oneOf schemas (expected exactly 1)"
            )
            errors.append(SchemaValidationError(path=path or "/", message=msg))

    not_schema = schema.get("not")
    if not_schema:
        if len(validate_schema(value, not_schema, path)) == 0:
            errors.append(SchemaValidationError(
                path=path or "/",
                message='Value should not match the "not" schema',
            ))

    return errors


def validate_output_schema(
    response_text: str,
    options: OutputSchemaOptions,
) -> OutputValidationResult:
    """Validate an LLM response string as JSON against a schema.

    First parses the JSON, then validates.
    """
    try:
        parsed = json.loads(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        errs = [SchemaValidationError(path="/", message=f"Invalid JSON: {e}")]
        if options.on_invalid:
            options.on_invalid(errs)
        return OutputValidationResult(valid=False, errors=errs)

    errs = validate_schema(parsed, options.schema)
    if errs and options.on_invalid:
        options.on_invalid(errs)

    return OutputValidationResult(valid=len(errs) == 0, errors=errs, parsed=parsed)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_json_type(value: Any) -> str:
    """Get the JSON type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _deep_equal(a: Any, b: Any) -> bool:
    """Deep equality comparison."""
    return a == b
