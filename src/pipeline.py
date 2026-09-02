"""Generation pipeline."""

from __future__ import annotations
import json
from collections.abc import Sequence
from llm_sdk import Small_LLM_Model
from .decoder import ConstrainedDecoder, StepFn
from .models import (
    FunctionCall,
    FunctionDefinition,
    validate_call_parameters,
)
from .vocabulary import Vocabulary


def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """Build the steering prompt shown to the model."""
    lines = [
        "You convert requests into function calls.",
        "Select a function only if it directly matches the request.",
        "Do not force a function choice for unrelated requests.",
        "Do not invent values for missing information.",
        "If no function matches the request, output NO_FUNCTION.",
        "Available functions:",
    ]
    for fn in functions:
        lines.append(f"- {fn.name}: {fn.description}")
    lines.append(f'Request: "{prompt}"')
    lines.append("Output:")
    return "\n".join(lines)


class Pipeline:
    """Load the model once and generate function calls per prompt."""

    def __init__(self, functions: list[FunctionDefinition]) -> None:
        """Load the model and build the constrained decoder.

        Args:
            functions: The available function definitions.

        Raises:
            ValueError: If no function definitions are available.
        """
        if not functions:
            raise ValueError("No function definitions were provided.")
        self._functions = functions
        self._model = Small_LLM_Model()
        self._vocab = Vocabulary(self._model.get_path_to_vocab_file())
        self._decoder = ConstrainedDecoder(
            functions, self._vocab, self._logits, self._encode
        )

    def _encode(self, text: str) -> list[int]:
        """Encode prompt text into a flat list of token ids."""
        return list(self._model.encode(text).tolist()[0])

    def _logits(self, ids: list[int]) -> Sequence[float]:
        """Return next-token logits for a token-id context."""
        return list(self._model.get_logits_from_input_ids(ids))

    def run(self, prompt: str, on_step: StepFn | None = None) -> FunctionCall:
        text = build_prompt(prompt, self._functions)

        raw = self._decoder.decode(
            text,
            on_step=on_step,
        )

        data = json.loads(raw)

        name = data["name"]
        parameters = data["parameters"]

        if name == "NO_FUNCTION":
            raise ValueError(
                "No suitable function found for this request"
            )

        function = next(
            (
                fn
                for fn in self._functions
                if fn.name == name
            ),
            None,
        )

        if function is None:
            raise ValueError(
                f"Unknown function generated: {name}"
            )

        validate_call_parameters(
            function,
            parameters,
        )

        return FunctionCall(
            prompt=prompt,
            name=name,
            parameters=parameters,
        )
    
    
