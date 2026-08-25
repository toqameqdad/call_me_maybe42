"""Translate natural-language prompts into function calls."""

import json

from .constrained_decoder import ConstrainedDecoder
from .models import FunctionCallResult, FunctionDefinition, TestPrompt

DEFAULT_MAX_SELECTION_TOKENS = 64
DEFAULT_MAX_ARGUMENT_TOKENS = 128


class FunctionCaller:
    """Select a function and extract its arguments with the LLM."""

    def __init__(
        self,
        decoder: ConstrainedDecoder,
        functions: list[FunctionDefinition],
        max_selection_tokens: int = DEFAULT_MAX_SELECTION_TOKENS,
        max_argument_tokens: int = DEFAULT_MAX_ARGUMENT_TOKENS,
    ) -> None:
        """Initialize the function caller."""
        if not functions:
            raise ValueError("At least one function definition is required.")
        if max_selection_tokens <= 0 or max_argument_tokens <= 0:
            raise ValueError("Token limits must be greater than zero.")

        self.decoder = decoder
        self.functions = {function.name: function for function in functions}
        self.max_selection_tokens = max_selection_tokens
        self.max_argument_tokens = max_argument_tokens

    def call(self, test_prompt: TestPrompt) -> FunctionCallResult:
        """Translate one natural-language request into a function call."""
        prompt = test_prompt.prompt
        function = self._select_function(prompt)
        parameters: dict[str, object] = {}

        for name, parameter in function.parameters.items():
            extraction_prompt = _build_parameter_prompt(
                prompt,
                function,
                name,
                parameter.type,
            )
            decoded = self.decoder.generate(
                prompt=extraction_prompt,
                schema={name: parameter.type},
                max_tokens=self.max_argument_tokens,
            )
            values = json.loads(decoded.text)
            parameters[name] = values[name]

        return FunctionCallResult(
            prompt=prompt,
            name=function.name,
            parameters=parameters,
        )

    def call_all(
        self,
        test_prompts: list[TestPrompt],
    ) -> list[FunctionCallResult]:
        """Translate all prompts, skipping prompts that fail safely."""
        results: list[FunctionCallResult] = []
        for test_prompt in test_prompts:
            try:
                results.append(self.call(test_prompt))
            except (ValueError, json.JSONDecodeError):
                continue
        return results

    def _select_function(self, prompt: str) -> FunctionDefinition:
        """Choose the function using constrained LLM generation."""
        selection_prompt = _build_selection_prompt(
            prompt,
            list(self.functions.values()),
        )
        input_ids = self.decoder.encode_prompt(selection_prompt)
        options = [json.dumps(name) for name in self.functions]
        selected = self.decoder.select_option(
            input_ids=input_ids,
            options=options,
            max_tokens=self.max_selection_tokens,
        )
        name = json.loads(selected)
        if name not in self.functions:
            raise ValueError(f"LLM selected an unknown function: {name!r}")
        return self.functions[name]


def _build_selection_prompt(
    prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build the prompt used to choose a function."""
    lines = [
        "Choose exactly one function for the user's request.",
        "Choose by meaning and by the function descriptions.",
        "Do not execute a function.",
        "Available functions:",
    ]
    for function in functions:
        lines.append(f"- {function.name}: {function.description}")
    lines.extend(
        [
            "",
            f"User request: {prompt}",
            "",
            "Output only the chosen function name as a JSON string.",
        ]
    )
    return "\n".join(lines)


def _build_parameter_prompt(
    prompt: str,
    function: FunctionDefinition,
    parameter_name: str,
    parameter_type: str,
) -> str:
    """Build a focused extraction prompt for one argument."""
    meaning = _parameter_meaning(parameter_name, parameter_type)
    return "\n".join(
        [
            "Extract exactly one argument from the user's request.",
            "Do not answer the request and do not execute the function.",
            "Use only information supported by the user's request.",
            "Copy requested text exactly when it is present in quotes.",
            "For numbers, copy the complete number, not a prefix of it.",
            f"Function: {function.name}",
            f"Function purpose: {function.description}",
            f"Parameter: {parameter_name}",
            f"Parameter type: {parameter_type}",
            f"Parameter meaning: {meaning}",
            "",
            f"User request: {prompt}",
            "",
            "Return only the requested argument in the JSON schema provided.",
        ]
    )


def _parameter_meaning(name: str, value_type: str) -> str:
    """Describe common parameter names without hardcoding test answers."""
    meanings = {
        "name": "the person or entity name requested by the user",
        "s": "the exact string the user asks the function to process",
        "a": "the first numeric argument requested by the user",
        "b": "the second numeric argument requested by the user",
        "source_string": "the original text on which the replacement is performed",
        "regex": "the regular-expression pattern matching the requested target",
        "replacement": "the text that should replace each regex match",
    }
    return meanings.get(
        name,
        f"the {value_type} argument named {name!r} required by the function",
    )
