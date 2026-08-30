"""Translate natural-language prompts into function calls."""

import json
import re

from .llm_decoder import LLMDecoder
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    TestPrompt,
)
from .validator import validate_parameters


class FunctionCaller:
    """Convert user prompts into structured function calls."""

    def __init__(
        self,
        decoder: LLMDecoder,
        functions: list[FunctionDefinition],
    ) -> None:

        if not functions:
            raise ValueError(
                "At least one function is required."
            )

        self.decoder = decoder

        self.functions = {
            function.name: function
            for function in functions
        }


    def call(
        self,
        test_prompt: TestPrompt,
    ) -> FunctionCallResult:
        """Translate one prompt into a function call."""

        prompt = test_prompt.prompt

        function = self._select_function(
            prompt
        )

        parameters = self._extract_parameters(
            prompt,
            function,
        )

        validate_parameters(
            parameters,
            function,
        )

        return FunctionCallResult(
            prompt=prompt,
            name=function.name,
            parameters=parameters,
        )


    def _select_function(
        self,
        prompt: str,
    ) -> FunctionDefinition:
        """Select the correct function."""

        functions_text = "\n".join(
            [
                (
                    f"- {function.name}: "
                    f"{function.description}"
                )
                for function in self.functions.values()
            ]
        )

        allowed_names = "\n".join(
            self.functions.keys()
        )

        request = f"""
You are a strict function router.

Your ONLY task is to select the function.
Do not extract parameters.
Do not solve the user request.
Do not execute anything.

Available functions:

{functions_text}

Allowed function names:

{allowed_names}

Rules:
- Select based on the intent of the request.
- Ignore words that are only input values.
- Text manipulation requests should select string functions.
- Arithmetic requests should select math functions.

User request:

{prompt}

Return ONLY the function name.
"""

        result = self.decoder.generate(
            request,
            max_tokens=32,
        )

        matches = re.findall(
            r"fn_[a-zA-Z0-9_]+",
            result,
        )

        if not matches:
            raise ValueError(
                f"Could not extract function name: {result}"
            )

        name = matches[0]

        if name not in self.functions:
            raise ValueError(
                f"Unknown function: {name}"
            )

        return self.functions[name]


    def _extract_parameters(
        self,
        prompt: str,
        function: FunctionDefinition,
    ) -> dict:
        """Extract function parameters."""

        schema = {
            name: parameter.type
            for name, parameter
            in function.parameters.items()
        }

        request = f"""
You extract JSON arguments for a function call.

Your ONLY task is to extract input parameters.
Do NOT solve the task.
Do NOT produce the function output.

Function:
{function.name}

Function description:
{function.description}

Parameters:
{json.dumps(schema)}

Required keys:
{", ".join(schema.keys())}

User request:
{prompt}

Rules:
- Return ONLY JSON.
- Include ALL required parameters.
- Do not add extra parameters.
- Extract only the values needed as function inputs.
- Do not execute the function.
- Do not return the function output.
- Use the function description and parameter names to understand the expected values.
- Always extract the original input values from the user request.
- Never transform, calculate, format, uppercase, lowercase, reverse, or modify values.
- The parameters are inputs to the function, not the function output.

For regex parameters:
- Return only valid regex syntax.
- Never return descriptive words.
- Words like digit, number, space, vowel are not valid regex values.
- Convert the concept into a regex pattern.
- The regex must match only the part being replaced.

For replacement parameters:
- Return only the characters that will be inserted.
- Never return descriptive names of symbols.
- Example: underscore means _

JSON:
"""

        result = self.decoder.generate_json(
            request,
            max_tokens=128,
        )

        return result


    def call_all(
        self,
        prompts: list[TestPrompt],
    ) -> list[FunctionCallResult]:
        """Process multiple prompts."""

        results = []

        for prompt in prompts:

            try:
                results.append(
                    self.call(prompt)
                )

            except ValueError as error:

                print(
                    f"Skipped: {error}"
                )

        return results
