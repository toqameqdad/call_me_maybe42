"""Pydantic models used by the Call Me Maybe project."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ParameterDefinition(BaseModel):
    """Describe one function parameter."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="JSON value type of the parameter.",
    )


class ReturnDefinition(BaseModel):
    """Describe a function return value."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(
        ...,
        description="JSON value type returned by the function.",
    )


class FunctionDefinition(BaseModel):
    """Describe one callable function."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        ...,
        description="Name of the function.",
    )

    description: str = Field(
        ...,
        description="Human-readable description of the function.",
    )

    parameters: dict[str, ParameterDefinition] = Field(
        default_factory=dict,
        description="Function parameter definitions.",
    )

    returns: ReturnDefinition = Field(
        ...,
        description="Function return type definition.",
    )


class TestPrompt(BaseModel):
    """Represent one input prompt."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        description="Natural-language user request.",
    )


class FunctionCallResult(BaseModel):
    """Represent one final function-calling result.

    The serialized object contains exactly:

        {
            "prompt": "...",
            "name": "...",
            "parameters": {...}
        }
    """

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        ...,
        description="Original user prompt.",
    )

    name: str = Field(
        ...,
        description="Function selected by the LLM.",
    )

    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments generated for the function.",
    )
