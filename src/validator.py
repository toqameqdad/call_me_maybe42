"""Validation utilities for generated function arguments."""

from typing import Any

from .models import FunctionDefinition


def validate_parameters(
    parameters: dict[str, Any],
    function: FunctionDefinition,
) -> dict[str, Any]:
    """Validate and convert generated parameters."""

    expected = function.parameters

    for name in expected:

        if name not in parameters:
            raise ValueError(
                f"Missing parameter: {name}"
            )

    for name in parameters:

        if name not in expected:
            raise ValueError(
                f"Unexpected parameter: {name}"
            )

    for name, definition in expected.items():

        parameters[name] = _convert_type(
            name,
            parameters[name],
            definition.type,
        )

    return parameters


def _convert_type(
    name: str,
    value: Any,
    expected_type: str,
) -> Any:
    """Convert and validate one parameter."""

    if expected_type == "string":

        return str(value)


    if expected_type == "number":

        if isinstance(value, str):

            try:
                number = float(value)

                if number.is_integer():
                    return int(number)

                return number

            except ValueError:

                raise ValueError(
                    f"{name} must be number"
                )


        if isinstance(value, (int, float)):

            return value


        raise ValueError(
            f"{name} must be number"
        )


    if expected_type == "boolean":

        if isinstance(value, str):

            return value.lower() == "true"

        return bool(value)


    if expected_type == "null":

        if value is not None:
            raise ValueError(
                f"{name} must be null"
            )

        return None


    raise ValueError(
        f"Unsupported type: {expected_type}"
    )
