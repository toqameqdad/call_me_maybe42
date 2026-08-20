"""Translate natural-language prompts into function calls."""

import json

from .constrained_decoder import ConstrainedDecoder
from .models import (
    FunctionCallResult,
    FunctionDefinition,
    TestPrompt,
)

DEFAULT_MAX_SELECTION_TOKENS = 64
DEFAULT_MAX_ARGUMENT_TOKENS = 256


class FunctionCaller:
    """Select functions and generate their arguments via the LLM."""

    def __init__(
        self,
        decoder: ConstrainedDecoder,
        functions: list[FunctionDefinition],
        max_selection_tokens: int = DEFAULT_MAX_SELECTION_TOKENS,
        max_argument_tokens: int = DEFAULT_MAX_ARGUMENT_TOKENS,
    ) -> None:
        """Initialize the function caller.

        Args:
            decoder: Constrained decoder used for LLM generation.
            functions: Available function definitions.
            max_selection_tokens: Token budget for choosing which
                function to call.
            max_argument_tokens: Token budget for generating the
                chosen function's arguments.

        Raises:
            ValueError: If no function definitions are provided.
        """
        if not functions:
            raise ValueError(
                "At least one function definition is required."
            )

        self.decoder = decoder
        self.functions = {
            function.name: function for function in functions
        }
        self.max_selection_tokens = max_selection_tokens
        self.max_argument_tokens = max_argument_tokens

    def call(self, test_prompt: TestPrompt) -> FunctionCallResult:
        """Translate one prompt into a structured function call.

        Args:
            test_prompt: Natural-language user request.

        Returns:
            The selected function name and generated arguments.

        Raises:
            ValueError: If function selection or argument
                generation fails.
        """
        prompt = test_prompt.prompt
        function = self._select_function(prompt)

        schema = {
            name: parameter.type
            for name, parameter in function.parameters.items()
        }

        decoded = self.decoder.generate(
            prompt=prompt,
            schema=schema,
            max_tokens=self.max_argument_tokens,
        )

        parameters = json.loads(decoded.text)

        return FunctionCallResult(
            prompt=prompt,
            name=function.name,
            parameters=parameters,
        )

    def call_all(
        self,
        test_prompts: list[TestPrompt],
    ) -> list[FunctionCallResult]:
        """Translate every prompt into a structured function call.

        Prompts that fail are skipped rather than aborting the
        whole batch, so one bad prompt cannot break the run.

        Args:
            test_prompts: Natural-language user requests.

        Returns:
            Structured function-calling results, in order.
        """
        results: list[FunctionCallResult] = []

        for test_prompt in test_prompts:
            try:
                results.append(self.call(test_prompt))
            except ValueError:
                continue

        return results

    def _select_function(self, prompt: str) -> FunctionDefinition:
        """Choose which function to call using the LLM.

        Args:
            prompt: Natural-language user request.

        Returns:
            The selected function definition.

        Raises:
            ValueError: If the LLM selects an unknown function
                or no function can be selected.
        """
        selection_prompt = _build_selection_prompt(
            prompt,
            list(self.functions.values()),
        )

        input_ids = self.decoder.encode_prompt(selection_prompt)

        options = [
            json.dumps(name) for name in self.functions
        ]

        selected = self.decoder.select_option(
            input_ids=input_ids,
            options=options,
            max_tokens=self.max_selection_tokens,
        )

        name = json.loads(selected)

        if name not in self.functions:
            raise ValueError(
                f"LLM selected an unknown function: {name!r}"
            )

        return self.functions[name]


def _build_selection_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt used to pick a function via the LLM.

    Args:
        prompt: Natural-language user request.
        functions: Available function definitions.

    Returns:
        Prompt text listing the candidate functions.
    """
    lines = [
        "You must choose exactly one function that can "
        "fulfill the user request below.",
        "",
        "Available functions:",
    ]

    for function in functions:
        lines.append(f"- {function.name}: {function.description}")

    lines.extend(
        [
            "",
            f'User request: "{prompt}"',
            "",
            "Respond with the chosen function name only, as a "
            "JSON string.",
            "Function name:",
        ]
    )

    return "\n".join(lines)
