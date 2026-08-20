"""Constrained decoding for schema-aware JSON generation."""

import json
import math
import re

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
    """Generate JSON while respecting logical constraints."""

    def __init__(
        self,
        model: Small_LLM_Model,
        vocabulary: Vocabulary,
    ) -> None:
        """Initialize the decoder.

        Args:
            model: LLM SDK model.
            vocabulary: Model vocabulary.
        """
        self.model = model
        self.vocabulary = vocabulary

    def _decode(
        self,
        token_ids: list[int],
    ) -> str:
        """Decode a complete token sequence.

        Args:
            token_ids: Token IDs to decode.

        Returns:
            Decoded text.
        """
        return self.model.decode(token_ids)

    def encode_prompt(self, prompt: str) -> list[int]:
        """Encode a natural-language prompt into input IDs.

        Args:
            prompt: Natural-language prompt.

        Returns:
            Token IDs representing the prompt.
        """
        encoded = self.model.encode(prompt)
        return _extract_input_ids(encoded)

    def select_option(
        self,
        input_ids: list[int],
        options: list[str],
        max_tokens: int,
    ) -> str:
        """Select one candidate string using the LLM.

        This mirrors constraint generation: at each step, only
        tokens that keep at least one candidate option reachable
        are allowed, and the highest-logit valid token is chosen.
        Generation stops as soon as one candidate is fully formed.

        Args:
            input_ids: Full model input sequence so far.
            options: Candidate literal strings to choose between
                (e.g. quoted JSON strings such as '"fn_greet"').
            max_tokens: Maximum tokens available for selection.

        Returns:
            The candidate option that was fully generated.

        Raises:
            ValueError: If no option is provided, or no option
                can be selected within the token budget.
        """
        if not options:
            raise ValueError(
                "No options provided for selection."
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        remaining = list(options)
        progress_ids: list[int] = []

        for _ in range(max_tokens):
            candidates = self._candidate_ids_for_options(
                progress_ids,
                remaining,
            )

            if not candidates:
                raise ValueError(
                    "No token can continue any candidate "
                    "option."
                )

            full_input = [
                *input_ids,
                *progress_ids,
            ]

            logits = self.model.get_logits_from_input_ids(
                full_input
            )

            token_id = self._choose_token(
                logits,
                candidates,
            )

            progress_ids.append(token_id)
            candidate_text = self._decode(progress_ids)

            remaining = [
                option
                for option in remaining
                if option.startswith(candidate_text)
            ]

            if candidate_text in remaining:
                return candidate_text

        raise ValueError(
            "Could not select an option within "
            f"{max_tokens} tokens."
        )

    def _candidate_ids_for_options(
        self,
        progress_ids: list[int],
        options: list[str],
    ) -> list[int]:
        """Find tokens that can continue at least one option.

        Args:
            progress_ids: Tokens already generated so far.
            options: Candidate strings still in the running.

        Returns:
            Valid candidate token IDs.
        """
        candidates: list[int] = []

        for token_id in self.vocabulary.id_to_token:
            candidate_ids = [
                *progress_ids,
                token_id,
            ]

            candidate_text = self._decode(candidate_ids)

            if any(
                option.startswith(candidate_text)
                for option in options
            ):
                candidates.append(token_id)

        return candidates

    def _candidate_token_ids(
        self,
        constraint: Constraint,
        progress_ids: list[int],
    ) -> list[int]:
        """Find tokens that can continue a constraint.

        Args:
            constraint: Logical constraint.
            progress_ids: Tokens already generated for this constraint.

        Returns:
            Valid candidate token IDs.
        """
        candidates: list[int] = []

        for token_id in self.vocabulary.id_to_token:
            candidate_ids = [
                *progress_ids,
                token_id,
            ]

            candidate_text = self._decode(candidate_ids)

            if constraint.literal:
                if _literal_prefix_is_valid(
                    candidate_text,
                    constraint.literal,
                ):
                    candidates.append(token_id)

            elif constraint.value_type:
                if _value_prefix_is_valid(
                    candidate_text,
                    constraint.value_type,
                ):
                    candidates.append(token_id)

        return candidates

    def _choose_token(
        self,
        logits: list[float],
        candidates: list[int],
    ) -> int:
        """Choose the highest-logit candidate.

        Args:
            logits: Model logits.
            candidates: Allowed token IDs.

        Returns:
            Selected token ID.

        Raises:
            ValueError: If no valid candidate exists.
        """
        valid = [
            token_id
            for token_id in candidates
            if 0 <= token_id < len(logits)
            and math.isfinite(logits[token_id])
        ]

        if not valid:
            raise ValueError(
                "No valid token has a usable model logit."
            )

        return max(
            valid,
            key=lambda token_id: logits[token_id],
        )

    def _generate_constraint(
        self,
        input_ids: list[int],
        constraint: Constraint,
        max_tokens: int,
    ) -> tuple[list[int], str]:
        """Generate tokens until one constraint is complete.

        Args:
            input_ids: Full model input sequence.
            constraint: Constraint to satisfy.
            max_tokens: Maximum tokens available.

        Returns:
            Generated token IDs and decoded constraint text.

        Raises:
            ValueError: If the constraint cannot be completed.
        """
        constraint_ids: list[int] = []

        for _ in range(max_tokens):
            candidates = self._candidate_token_ids(
                constraint,
                constraint_ids,
            )

            if not candidates:
                raise ValueError(
                    "No token can continue constraint "
                    f"{constraint}."
                )

            full_input = [
                *input_ids,
                *constraint_ids,
            ]

            logits = self.model.get_logits_from_input_ids(
                full_input
            )

            token_id = self._choose_token(
                logits,
                candidates,
            )

            candidate_ids = [
                *constraint_ids,
                token_id,
            ]

            candidate_text = self._decode(candidate_ids)

            constraint_ids.append(token_id)

            if _constraint_is_complete(
                candidate_text,
                constraint,
            ):
                return constraint_ids, candidate_text

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
        """Generate schema-constrained JSON.

        Args:
            prompt: Natural-language prompt.
            schema: JSON field names and their expected types.
            max_tokens: Maximum generated token count.

        Returns:
            Generated text and token IDs.

        Raises:
            ValueError: If constrained generation fails.
        """
        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than zero."
            )

        state = initial_state(schema)

        input_ids = self.encode_prompt(prompt)

        generated_ids: list[int] = []
        generated_text = ""

        while state.kind != Kind.END:
            remaining = max_tokens - len(generated_ids)

            if remaining <= 0:
                raise ValueError(
                    "Maximum generation token limit reached."
                )

            allowed = get_allowed(state)

            if not allowed:
                raise ValueError(
                    f"No constraints available in state "
                    f"{state.kind}."
                )

            constraint = self._select_constraint(
                input_ids=[
                    *input_ids,
                    *generated_ids,
                ],
                constraints=allowed,
            )

            constraint_ids, constraint_text = (
                self._generate_constraint(
                    input_ids=[
                        *input_ids,
                        *generated_ids,
                    ],
                    constraint=constraint,
                    max_tokens=remaining,
                )
            )

            generated_ids.extend(constraint_ids)
            generated_text += constraint_text

            state = advance(
                state,
                constraint,
            )

        _validate_json(
            generated_text,
            schema,
        )

        return DecodeResult(
            text=generated_text,
            token_ids=generated_ids,
        )

    def _select_constraint(
        self,
        input_ids: list[int],
        constraints: frozenset[Constraint],
    ) -> Constraint:
        """Select the highest-scoring valid constraint.

        Args:
            input_ids: Current model input IDs.
            constraints: Constraints allowed by the state machine.

        Returns:
            Selected constraint.

        Raises:
            ValueError: If no constraint can be continued.
        """
        if len(constraints) == 1:
            return next(iter(constraints))

        logits = self.model.get_logits_from_input_ids(
            input_ids
        )

        best_constraint: Constraint | None = None
        best_score = float("-inf")

        for constraint in constraints:
            candidates = self._candidate_token_ids(
                constraint,
                [],
            )

            if not candidates:
                continue

            token_id = self._choose_token(
                logits,
                candidates,
            )

            score = logits[token_id]

            if score > best_score:
                best_score = score
                best_constraint = constraint

        if best_constraint is None:
            raise ValueError(
                "No valid constraint can be selected."
            )

        return best_constraint


def _literal_prefix_is_valid(
    candidate: str,
    literal: str,
) -> bool:
    """Return whether candidate is a literal prefix."""
    return literal.startswith(candidate)


def _value_prefix_is_valid(
    candidate: str,
    value_type: str,
) -> bool:
    """Return whether candidate can be a JSON value prefix."""
    if value_type == "string":
        return _valid_string_prefix(candidate)

    if value_type == "number":
        return _valid_number_prefix(candidate)

    if value_type == "boolean":
        return _valid_boolean_prefix(candidate)

    if value_type == "null":
        return "null".startswith(candidate)

    raise ValueError(
        f"Unsupported value type: {value_type!r}"
    )


def _constraint_is_complete(
    value: str,
    constraint: Constraint,
) -> bool:
    """Return whether a constraint is fully generated."""
    if constraint.literal:
        return value == constraint.literal

    return _value_is_complete(
        value,
        constraint.value_type,
    )


def _value_is_complete(
    value: str,
    value_type: str,
) -> bool:
    """Return whether a JSON value is complete."""
    if value_type == "string":
        if not _valid_string_prefix(value):
            return False

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False

        return isinstance(parsed, str)

    if value_type == "number":
        if not _valid_number_prefix(value):
            return False

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False

        return (
            isinstance(parsed, (int, float))
            and not isinstance(parsed, bool)
        )

    if value_type == "boolean":
        return value in {"true", "false"}

    if value_type == "null":
        return value == "null"

    return False


def _valid_boolean_prefix(value: str) -> bool:
    """Return whether value prefixes true or false."""
    return (
        "true".startswith(value)
        or "false".startswith(value)
    )


def _valid_number_prefix(value: str) -> bool:
    """Return whether value is a valid JSON number prefix."""
    if value == "":
        return True

    pattern = re.compile(
        r"^-?(?:0|[1-9][0-9]*)"
        r"(?:\.[0-9]*)?"
        r"(?:[eE][+-]?[0-9]*)?$"
    )

    if value in {"-", "-.", ".", "-e", "-E"}:
        return False

    if pattern.fullmatch(value):
        return True

    if value in {"0.", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."}:
        return True

    exponent_match = re.fullmatch(
        r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?[eE][+-]?$",
        value,
    )

    return exponent_match is not None


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

                if not re.fullmatch(
                    r"[0-9a-fA-F]{4}",
                    remaining,
                ):
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
        raise ValueError(
            "The result of encode() does not support tolist()."
        )

    values = encoded.tolist()

    if not isinstance(values, list):
        raise ValueError(
            "Unexpected result returned by encode()."
        )

    if values and isinstance(values[0], list):
        values = values[0]

    if not all(isinstance(value, int) for value in values):
        raise ValueError(
            "Encoded input IDs must be integers."
        )

    return values


def _validate_json(
    text: str,
    schema: dict[str, str],
) -> None:
    """Validate generated JSON against the expected schema."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Generated output is not valid JSON: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            "Generated output must be a JSON object."
        )

    if set(data.keys()) != set(schema.keys()):
        raise ValueError(
            "Generated keys do not match the schema."
        )

    for key, value_type in schema.items():
        value = data[key]

        if value_type == "string":
            valid = isinstance(value, str)

        elif value_type == "number":
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        elif value_type == "boolean":
            valid = isinstance(value, bool)

        elif value_type == "null":
            valid = value is None

        else:
            raise ValueError(
                f"Unsupported value type: {value_type!r}"
            )

        if not valid:
            raise ValueError(
                f"Invalid type for field {key!r}: "
                f"expected {value_type!r}."
            )
