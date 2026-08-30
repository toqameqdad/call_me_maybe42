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
    """Generate JSON with schema-aware token constraints."""

    _TOP_K = 4096

    def __init__(
        self,
        model: Small_LLM_Model,
        vocabulary: Vocabulary,
    ) -> None:
        self.model = model
        self.vocabulary = vocabulary
        self._token_text_cache: dict[int, str] = {}

    def _token_text(self, token_id: int) -> str:
        """Decode one token and cache its text."""

        if token_id not in self._token_text_cache:
            self._token_text_cache[token_id] = self.model.decode(
                [token_id]
            )

        return self._token_text_cache[token_id]

    def _decode(self, token_ids: list[int]) -> str:
        return self.model.decode(token_ids)

    def encode_prompt(self, prompt: str) -> list[int]:
        encoded = self.model.encode(prompt)

        values = encoded.tolist()

        if values and isinstance(values[0], list):
            values = values[0]

        return values

    def select_option(
        self,
        input_ids: list[int],
        options: list[str],
        max_tokens: int,
    ) -> str:

        if not options:
            raise ValueError("No options provided.")

        progress_ids: list[int] = []
        prefix = ""

        for _ in range(max_tokens):

            logits = self.model.get_logits_from_input_ids(
                [
                    *input_ids,
                    *progress_ids,
                ]
            )

            token_id, candidate = self._choose_candidate(
                logits,
                prefix,
                lambda text: any(
                    option.startswith(text)
                    for option in options
                ),
            )

            progress_ids.append(token_id)
            prefix = candidate

            if prefix in options:
                return prefix

        raise ValueError(
            "Could not select option."
        )

    def _choose_candidate(
        self,
        logits: list[float],
        prefix: str,
        validator: Callable[[str], bool],
    ) -> tuple[int, str]:

        ranked = _top_logit_ids(
            logits,
            self._TOP_K,
        )

        result = self._best_from_ids(
            ranked,
            logits,
            prefix,
            validator,
        )

        if result:
            return result

        result = self._best_from_ids(
            range(len(logits)),
            logits,
            prefix,
            validator,
        )

        if result:
            return result

        raise ValueError(
            "No token can continue the current constraint."
        )

    def _best_from_ids(
        self,
        token_ids: Iterable[int],
        logits: list[float],
        prefix: str,
        validator: Callable[[str], bool],
    ):

        best_id = None
        best_text = ""
        best_score = float("-inf")

        for token_id in token_ids:

            if token_id >= len(logits):
                continue

            score = logits[token_id]

            if (
                not math.isfinite(score)
                or score <= best_score
            ):
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

    def generate(
        self,
        prompt: str,
        schema: dict[str, str],
        max_tokens: int = 256,
    ) -> DecodeResult:

        state = initial_state(schema)

        input_ids = self.encode_prompt(prompt)

        generated_ids: list[int] = []

        while state.kind != Kind.END:

            allowed = get_allowed(state)

            constraint = self._select_constraint(
                [
                    *input_ids,
                    *generated_ids,
                ],
                allowed,
            )

            ids, _ = self._generate_constraint(
                [
                    *input_ids,
                    *generated_ids,
                ],
                constraint,
                max_tokens,
            )

            generated_ids.extend(ids)

            state = advance(
                state,
                constraint,
            )

        text = self._decode(generated_ids)

        _validate_json(
            text,
            schema,
        )

        return DecodeResult(
            text=text,
            token_ids=generated_ids,
        )

    def _generate_constraint(
        self,
        input_ids: list[int],
        constraint: Constraint,
        max_tokens: int,
    ):

        ids = []
        prefix = ""

        for _ in range(max_tokens):

            logits = self.model.get_logits_from_input_ids(
                [
                    *input_ids,
                    *ids,
                ]
            )

            validator = (
                lambda text: _literal_prefix_is_valid(
                    text,
                    constraint.literal,
                )
                if constraint.literal
                else _value_prefix_is_valid(
                    text,
                    constraint.value_type,
                )
            )

            token_id, candidate = self._choose_candidate(
                logits,
                prefix,
                validator,
            )

            ids.append(token_id)
            prefix = candidate

            if _constraint_is_complete(
                prefix,
                constraint,
            ):
                return ids, prefix

        raise ValueError(
            f"Constraint could not be completed: {constraint}"
        )

    def _select_constraint(
        self,
        input_ids,
        constraints,
    ):

        if len(constraints) == 1:
            return next(iter(constraints))

        logits = self.model.get_logits_from_input_ids(
            input_ids
        )

        best = None
        best_score = float("-inf")

        for constraint in constraints:

            try:
                token_id, _ = self._choose_candidate(
                    logits,
                    "",
                    lambda text, c=constraint:
                        _constraint_prefix_is_valid(
                            text,
                            c,
                        ),
                )

            except ValueError:
                continue

            if logits[token_id] > best_score:
                best_score = logits[token_id]
                best = constraint

        if best is None:
            raise ValueError(
                "No valid constraint."
            )

        return best


def _top_logit_ids(
    logits: list[float],
    limit: int,
):
    ranked = heapq.nlargest(
        limit,
        enumerate(logits),
        key=lambda x: x[1],
    )

    return (
        token_id
        for token_id, _ in ranked
    )


def _constraint_prefix_is_valid(
    candidate: str,
    constraint: Constraint,
):

    if constraint.literal:
        return _literal_prefix_is_valid(
            candidate,
            constraint.literal,
        )

    return _value_prefix_is_valid(
        candidate,
        constraint.value_type,
    )


def _literal_prefix_is_valid(
    candidate: str,
    literal: str,
):

    return literal.startswith(candidate)


def _value_prefix_is_valid(
    candidate: str,
    value_type: str,
):

    if value_type == "string":
        return _valid_string_prefix(candidate)

    if value_type == "number":
        return _valid_number_prefix(candidate)

    if value_type == "boolean":
        return "true".startswith(candidate) or "false".startswith(candidate)

    if value_type == "null":
        return "null".startswith(candidate)

    raise ValueError(
        f"Unsupported type {value_type}"
    )


def _constraint_is_complete(
    value: str,
    constraint: Constraint,
):

    if constraint.literal:
        return value == constraint.literal

    return _value_is_complete(
        value,
        constraint.value_type,
    )


def _value_is_complete(
    value: str,
    value_type: str,
):

    if value_type == "string":

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False

        return isinstance(parsed, str) and parsed != ""

    if value_type == "number":

        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return False

        return (
            isinstance(parsed, (int, float))
            and not isinstance(parsed, bool)
        )

    if value_type == "boolean":
        return value in {
            "true",
            "false",
        }

    return value == "null"


def _valid_string_prefix(value: str):

    if not value:
        return True

    if not value.startswith('"'):
        return False

    escaped = False

    for char in value[1:]:

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True

        elif ord(char) < 0x20:
            return False

    return True


def _valid_number_prefix(value: str):

    if value == "":
        return True

    return bool(
        re.fullmatch(
            r"-?\d*(\.\d*)?([eE][+-]?\d*)?",
            value,
        )
    )


def _validate_json(
    text: str,
    schema: dict[str, str],
):

    data = json.loads(text)

    if set(data.keys()) != set(schema.keys()):
        raise ValueError(
            "Generated keys do not match schema."
        )
