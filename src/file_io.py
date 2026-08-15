"""File loading and saving helpers.

Each loader follows the same three-step pipeline:

    1. Open the file
    2. Parse it as JSON
    3. Validate the parsed data with Pydantic

Every failure is caught and re-raised as a clear, human-readable
error so the caller (the CLI) can print something useful instead
of a raw traceback.
"""

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from src.models import (
    FunctionCallResult,
    FunctionDefinition,
    TestPrompt,
)


def _read_json(path: str) -> object:
    """Open a file and parse it as JSON.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        RuntimeError: If the file does not exist or contains invalid JSON.
    """
    file_path = Path(path)

    try:
        raw_text = file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise RuntimeError(
            f"File not found: {file_path}"
        ) from None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON in {file_path}: {error}"
        ) from None


def load_functions_definitions(
    path: str,
) -> list[FunctionDefinition]:
    """Load and validate function definitions.

    Args:
        path: Path to functions_definition.json.

    Returns:
        A list of validated FunctionDefinition objects.

    Raises:
        RuntimeError: If the JSON does not match the expected schema.
    """
    raw_data = _read_json(path)

    try:
        adapter = TypeAdapter(list[FunctionDefinition])
        return adapter.validate_python(raw_data)
    except ValidationError as error:
        raise RuntimeError(
            f"File does not match the expected schema: "
            f"{path} ({error})"
        ) from None


def load_test_prompts(
    path: str,
) -> list[TestPrompt]:
    """Load and validate test prompts.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        A list of validated TestPrompt objects.

    Raises:
        RuntimeError: If the JSON does not match the expected schema.
    """
    raw_data = _read_json(path)

    try:
        adapter = TypeAdapter(list[TestPrompt])
        return adapter.validate_python(raw_data)
    except ValidationError as error:
        raise RuntimeError(
            f"File does not match the expected schema: "
            f"{path} ({error})"
        ) from None


def save_results(
    path: str,
    results: list[FunctionCallResult],
) -> None:
    """Save function call results to a JSON file.

    Args:
        path: Path where the output JSON should be written.
        results: Validated function call results.
    """
    data = [result.model_dump() for result in results]

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    file_path.write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
