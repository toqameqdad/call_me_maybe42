"""JSON input/output helpers for the Call Me Maybe project."""

import json
from pathlib import Path
from typing import cast

from pydantic import TypeAdapter, ValidationError

from .models import (
    FunctionCallResult,
    FunctionDefinition,
    TestPrompt,
)


def _read_json(path: str | Path) -> object:
    """Read and parse a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON value.

    Raises:
        RuntimeError: If the file is missing or contains invalid JSON.
    """
    file_path = Path(path)

    try:
        text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            f"File not found: {file_path}"
        ) from None
    except OSError as error:
        raise RuntimeError(
            f"Unable to read file {file_path}: {error}"
        ) from None

    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON in {file_path}: {error}"
        ) from None


def load_functions_definitions(
    path: str | Path,
) -> list[FunctionDefinition]:
    """Load and validate function definitions.

    Args:
        path: Path to functions_definition.json.

    Returns:
        Validated function definitions.

    Raises:
        RuntimeError: If the file does not match the expected format.
    """
    raw_data = _read_json(path)

    try:
        adapter: TypeAdapter[list[FunctionDefinition]] = TypeAdapter(
            list[FunctionDefinition]
        )
        validated = adapter.validate_python(raw_data)

        return cast(
            list[FunctionDefinition],
            validated,
        )
    except ValidationError as error:
        raise RuntimeError(
            f"Invalid function definitions in {path}: {error}"
        ) from None


def load_test_prompts(
    path: str | Path,
) -> list[TestPrompt]:
    """Load and validate test prompts.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        Validated test prompts.

    Raises:
        RuntimeError: If the file does not match the expected format.
    """
    raw_data = _read_json(path)

    try:
        adapter: TypeAdapter[list[TestPrompt]] = TypeAdapter(
            list[TestPrompt]
        )
        validated = adapter.validate_python(raw_data)

        return cast(
            list[TestPrompt],
            validated,
        )
    except ValidationError as error:
        raise RuntimeError(
            f"Invalid test prompts in {path}: {error}"
        ) from None


def save_results(
    path: str | Path,
    results: list[FunctionCallResult],
) -> None:
    """Save function-calling results as JSON.

    Args:
        path: Output JSON path.
        results: Results to save.

    Raises:
        RuntimeError: If the output file cannot be written.
    """
    file_path = Path(path)

    try:
        file_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_path.write_text(
            json.dumps(
                [result.model_dump() for result in results],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except OSError as error:
        raise RuntimeError(
            f"Unable to write results to {file_path}: {error}"
        ) from None
