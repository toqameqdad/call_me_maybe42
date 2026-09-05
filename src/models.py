"""Pydantic data models for the function-calling project."""

from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


class ParameterSpec(BaseModel):
    """Type specification for one parameter or return value."""

    type: Literal["string", "number", "integer", "boolean"]


class FunctionDefinition(BaseModel):
    """One callable function the model may choose to invoke."""

    name: str
    description: str = ""
    parameters: dict[str, ParameterSpec] = Field(default_factory=dict)
    returns: ParameterSpec | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        """Reject empty or whitespace-only function names.

        A blank name is silently unreachable in the NAME decoding
        phase (it can never win a character comparison against a
        non-empty candidate), which causes the model to fall back to
        a different, wrong function with no visible error — so this
        must be caught at load time instead.
        """
        if not value.strip():
            raise ValueError("Function name must not be empty or blank.")
        return value


class TestPrompt(BaseModel):
    """A single natural-language request to process."""

    prompt: str


class FunctionCall(BaseModel):
    """The structured result produced for one prompt."""

    prompt: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)


def validate_function_list(functions: list[FunctionDefinition]) -> None:
    """Check cross-item constraints that a single FunctionDefinition can't.

    Raises:
        ValueError: If the list is empty or contains duplicate names.
    """
    if not functions:
        raise ValueError("No function definitions were provided.")
    names = [fn.name for fn in functions]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"Duplicate function name(s): {sorted(duplicates)}")


def validate_call_parameters(
    function: FunctionDefinition,
    parameters: dict[str, Any],
) -> None:
    """Validate generated parameters against function schema."""

    expected = function.parameters

    if set(parameters.keys()) != set(expected.keys()):
        raise ValueError("Generated parameters do not match function schema.")

    for name, spec in expected.items():
        value = parameters[name]

        if spec.type == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")

        elif spec.type == "number":
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")

        elif spec.type == "string":
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")

        elif spec.type == "boolean":
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
