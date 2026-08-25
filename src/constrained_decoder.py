"""Schema-aware constrained decoding for JSON function calls."""

import heapq
import json
import math
import re
from typing import Callable, Iterable

from pydantic import BaseModel, ConfigDict

from llm_sdk import Small_LLM_Model

from .state_machine import (
    Constraint,
    Kind,
    advance,
    get_allowed,
    initial_state,
)
from .vocabulary import Vocabulary


class DecodeResult(BaseModel):
    """Represent the result of constrained generation."""

    model_config = ConfigDict(frozen=True)

    text: str
    token_ids: list[int]


class ConstrainedDecoder:
    """Generate JSON with schema-aware token constraints.

    The implementation uses only the public ``llm_sdk`` API.  Token
    filtering is performed from cached single-token decoded text, while
    the selected token is always appended to the real generation context.
    This avoids decoding every candidate sequence at every generation step.
    """

    _TOP_K = 4096

    def __init__(
        self,
        model: Small_LLM_Model,
        vocabulary: Vocabulary,
    ) -> None:
        """Initialize the decoder."""
        self.model = model
        self.vocabulary = vocabulary
        self._token_text_cache: dict[int, str] = {}

    def _token_text(self, token_id: int) -> str:
        """Decode one vocabulary token and cache its text."""
        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.model.decode([token_id])
        return self._token_text_cache[token_id]

    def _decode(self, token_ids: list[int]) -> str:
        """Decode a complete generated sequence."""
        return self.model.decode(token_ids)

    def encode_prompt(self, prompt: str) -> list[int]:
        """Encode a natural-language prompt into model input IDs."""
        return _extract_input_ids(self.model.encode(prompt))

    def select_option(
        self,
        input_ids: list[int],
        options: list[str],
        max_tokens: int,
    ) -> str:
        """Select one literal option using constrained decoding."""
        if not options:
            raise ValueError("No options provided for selection.")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        progress_ids: list[int] = []
        prefix = ""

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(
                [*input_ids, *progress_ids]
            )
            token_id, candidate = self._choose_candidate(
                logits,
                progress_ids,
                prefix,
                lambda text: any(
                    option.startswith(text) for option in options
                ),
            )
            progress_ids.append(token_id)
            prefix = candidate

            if prefix in options:
                return prefix

        raise ValueError(
            f"Could not select an option within {max_tokens} tokens."
        )

    def _choose_candidate(
        self,
        logits: list[float],
        progress_ids: list[int],
        prefix: str,
        validator: Callable[[str], bool],
    ) -> tuple[int, str]:
        """Choose the highest-logit token satisfying a prefix validator."""
        del progress_ids  # Kept in the API for generation-state clarity.

        ranked = _top_logit_ids(logits, self._TOP_K)
        result = self._best_from_ids(ranked, logits, prefix, validator)
        if result is not None:
            return result

        # A valid token can have a low logit early in generation.  Fall back
        # to the full vocabulary only when the cheap top-k pass found none.
        all_ids = range(len(logits))
        result = self._best_from_ids(all_ids, logits, prefix, validator)
        if result is not None:
            return result

        raise ValueError("No token can continue the current constraint.")

    def _best_from_ids(
        self,
        token_ids: Iterable[int],
        logits: list[float],
        prefix: str,
        validator: Callable[[str], bool],
    ) -> tuple[int, str] | None:
        """Return the best valid token from a supplied ID sequence."""
        best_id: int | None = None
        best_text = ""
        best_score = float("-inf")

        for token_id in token_ids:
            if token_id < 0 or token_id >= len(logits):
                continue
            score = logits[token_id]
            if not math.isfinite(score) or score <= best_score:
                continue

            token_text = self._token_text(token_id)
            if not token_text:
                continue

            candidate = prefix + token_text
            if validator(candidate):
                best_id = token_id
                best_text = candidate
                best_score = score

        if best_id is None:
            return None
        return best_id, best_text

    def _generate_constraint(
        self,
        input_ids: list[int],
        constraint: Constraint,
        max_tokens: int,
    ) -> tuple[list[int], str]:
        """Generate one complete schema constraint."""
        constraint_ids: list[int] = []
        prefix = ""

        for _ in range(max_tokens):
            logits = self.model.get_logits_from_input_ids(
                [*input_ids, *constraint_ids]
            )

            if constraint.literal:
                validator = lambda text, literal=constraint.literal: (
                    _literal_prefix_is_valid(text, literal)
                )
            else:
                validator = lambda text, value_type=constraint.value_type: (
                    _value_prefix_is_valid(text, value_type)
                )

            token_id, candidate = self._choose_candidate(
                logits,
                constraint_ids,
                prefix,
                validator,
            )
            constraint_ids.append(token_id)
            prefix = candidate

            if _constraint_is_complete(prefix, constraint):
                return constraint_ids, prefix

        raise ValueError(
            "Constraint could not be completed within "
            f"{max_tokens} tokens: {constraint}"
        )

    def generate(
        self,
        prompt: str,
        schema: dict[str, str],
        max_tokens: int = 256,
    ) -> DecodeResult:
        """Generate a JSON object satisfying the supplied schema."""
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        state = initial_state(schema)
        input_ids = self.encode_prompt(prompt)
        generated_ids: list[int] = []
        generated_text = ""

        while state.kind != Kind.END:
            remaining = max_tokens - len(generated_ids)
            if remaining <= 0:
                raise ValueError("Maximum generation token limit reached.")

            allowed = get_allowed(state)
            if not allowed:
                raise ValueError(f"No constraints available in state {state.kind}.")

            constraint = self._select_constraint(
                [*input_ids, *generated_ids],
                allowed,
            )
            ids, text = self._generate_constraint(
                [*input_ids, *generated_ids],
                constraint,
                remaining,
            )
            generated_ids.extend(ids)
            generated_text += text
            state = advance(state, constraint)

        # Re-decode once as a final guard against tokenizer composition issues.
        final_text = self._decode(generated_ids)
        _validate_json(final_text, schema)
        return DecodeResult(text=final_text, token_ids=generated_ids)

    def _select_constraint(
        self,
        input_ids: list[int],
        constraints: frozenset[Constraint],
    ) -> Constraint:
        """Select the highest-scoring constraint that can start."""
        if len(constraints) == 1:
            return next(iter(constraints))

        logits = self.model.get_logits_from_input_ids(input_ids)
        best: Constraint | None = None
        best_score = float("-inf")

        for constraint in constraints:
            try:
                token_id, _ = self._choose_candidate(
                    logits,
                    [],
                    "",
                    lambda text, c=constraint: _constraint_prefix_is_valid(
                        text, c
                    ),
                )
            except ValueError:
                continue

            score = logits[token_id]
            if score > best_score:
                best_score = score
                best = constraint

        if best is None:
            raise ValueError("No valid constraint can be selected.")
        return best


def _top_logit_ids(logits: list[float], limit: int) -> Iterable[int]:
    """Yield the highest-scoring finite token IDs."""
    ranked = heapq.nlargest(
        min(limit, len(logits)),
        (
            (score, token_id)
            for token_id, score in enumerate(logits)
            if math.isfinite(score)
        ),
    )
    return (token_id for _, token_id in ranked)


def _constraint_prefix_is_valid(candidate: str, constraint: Constraint) -> bool:
    """Return whether candidate is a valid constraint prefix."""
    if constraint.literal:
        return _literal_prefix_is_valid(candidate, constraint.literal)
    return _value_prefix_is_valid(candidate, constraint.value_type)


def _literal_prefix_is_valid(candidate: str, literal: str) -> bool:
    """Return whether candidate is a prefix of a literal."""
    return literal.startswith(candidate)


def _value_prefix_is_valid(candidate: str, value_type: str) -> bool:
    """Return whether candidate can be a JSON value prefix."""
    if value_type == "string":
        return _valid_string_prefix(candidate)
    if value_type == "number":
        return _valid_number_prefix(candidate)
    if value_type == "boolean":
        return _valid_boolean_prefix(candidate)
    if value_type == "null":
        return "null".startswith(candidate)
    raise ValueError(f"Unsupported value type: {value_type!r}")


def _constraint_is_complete(value: str, constraint: Constraint) -> bool:
    """Return whether a constraint is fully generated."""
    if constraint.literal:
        return value == constraint.literal
    return _value_is_complete(value, constraint.value_type)


def _value_is_complete(value: str, value_type: str) -> bool:
    """Return whether a JSON value is complete."""
    if value_type == "string":
        if not _valid_string_prefix(value):
            return False
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        # Avoid accepting an empty string immediately when the model has
        # been asked to extract a concrete argument from natural language.
        return isinstance(parsed, str) and parsed != ""

    if value_type == "number":
        if not _valid_number_prefix(value):
            return False
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False
        return isinstance(parsed, (int, float)) and not isinstance(parsed, bool)

    if value_type == "boolean":
        return value in {"true", "false"}
    if value_type == "null":
        return value == "null"
    return False


def _valid_boolean_prefix(value: str) -> bool:
    """Return whether value prefixes true or false."""
    return "true".startswith(value) or "false".startswith(value)


def _valid_number_prefix(value: str) -> bool:
    """Return whether value is a valid JSON number prefix."""
    if value == "":
        return True

    pattern = re.compile(
        r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?(?:[eE][+-]?[0-9]*)?$"
    )
    if value in {"-", "-.", ".", "-e", "-E"}:
        return False
    if pattern.fullmatch(value):
        return True

    return bool(
        re.fullmatch(
            r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?[eE][+-]?$",
            value,
        )
    )


def _valid_string_prefix(value: str) -> bool:
    """Return whether value is a valid JSON string prefix."""
    if not value:
        return True
    if not value.startswith('"'):
        return False

    escaped = False
    index = 1
    while index < len(value):
        char = value[index]
        if escaped:
            if char not in '"\\/bfnrtu':
                return False
            if char == "u":
                remaining = value[index + 1:index + 5]
                if len(remaining) < 4:
                    return True
                if not re.fullmatch(r"[0-9a-fA-F]{4}", remaining):
                    return False
                index += 4
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            if index != len(value) - 1:
                return False
        elif ord(char) < 0x20:
            return False
        index += 1
    return True


def _extract_input_ids(encoded: object) -> list[int]:
    """Extract input IDs from the SDK tensor."""
    if not hasattr(encoded, "tolist"):
        raise ValueError("The result of encode() does not support tolist().")
    values = encoded.tolist()
    if not isinstance(values, list):
        raise ValueError("Unexpected result returned by encode().")
    if values and isinstance(values[0], list):
        values = values[0]
    if not all(isinstance(value, int) for value in values):
        raise ValueError("Encoded input IDs must be integers.")
    return values


def _validate_json(text: str, schema: dict[str, str]) -> None:
    """Validate generated JSON against the expected schema."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Generated output is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Generated output must be a JSON object.")
    if set(data.keys()) != set(schema.keys()):
        raise ValueError("Generated keys do not match the schema.")

    for key, value_type in schema.items():
        value = data[key]
        if value_type == "string":
            valid = isinstance(value, str)
        elif value_type == "number":
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif value_type == "boolean":
            valid = isinstance(value, bool)
        elif value_type == "null":
            valid = value is None
        else:
            raise ValueError(f"Unsupported schema type: {value_type!r}")
        if not valid:
            raise ValueError(
                f"Invalid type for {key!r}: expected {value_type!r}."
            )
