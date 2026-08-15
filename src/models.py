"""Pydantic data models for the call_me_maybe function calling project.

These models mirror the exact JSON shapes described in the project spec:
- functions_definition.json  -> list[FunctionDefinition]
- function_calling_tests.json -> list[TestPrompt]
- function_calling_results.json -> list[FunctionCallResult]
"""

from typing import Any, Dict

from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    """Describes a single parameter of a function (its declared type).

    Example (from functions_definition.json):
        "a": {"type": "number"}
    """

    type: str = Field(
        ...,
        description=(
            "Declared type of the parameter, "
            "e.g. 'number', 'string', 'boolean'."
        ),
    )


class ReturnDefinition(BaseModel):
    """Describes the return type of a function.

    Example (from functions_definition.json):
        "returns": {"type": "number"}
    """

    type: str = Field(
        ...,
        description="Declared return type of the function.",
    )


class FunctionDefinition(BaseModel):
    """A single function entry as found in functions_definition.json.

    Example:
        {
            "name": "fn_add_numbers",
            "description": "Add two numbers together and return their sum",
            "parameters": {
                "a": {"type": "number"},
                "b": {"type": "number"}
            },
            "returns": {"type": "number"}
        }
    """

    name: str = Field(
        ...,
        description="Name of the callable function.",
    )
    description: str = Field(
        ...,
        description="Human-readable description of the function.",
    )
    parameters: Dict[str, ParameterDefinition] = Field(
        ...,
        description="Mapping of parameter name to its type.",
    )
    returns: ReturnDefinition = Field(
        ...,
        description="Type definition of the return value.",
    )


class TestPrompt(BaseModel):
    """A single entry from function_calling_tests.json.

    Example:
        {"prompt": "What is the sum of 2 and 3?"}
    """

    prompt: str = Field(
        ...,
        description="Natural language request to process.",
    )


class FunctionCallResult(BaseModel):
    """A single entry to write into function_calling_results.json.

    Must contain exactly these three keys per the spec
    (no extra keys, no prose):
        {
            "prompt": "What is the sum of 2 and 3?",
            "name": "fn_add_numbers",
            "parameters": {"a": 2, "b": 3}
        }
    """

    prompt: str = Field(
        ...,
        description="The original natural-language request.",
    )
    name: str = Field(
        ...,
        description="Name of the function chosen by the LLM.",
    )
    parameters: Dict[str, Any] = Field(
        ...,
        description="Resolved arguments, matching the definition.",
    )
