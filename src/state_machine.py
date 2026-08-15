"""Schema-aware JSON object state machine.

Unlike the earlier toy version, this does NOT hardcode field names
like "name" or "age". It reads a schema (a dict of field name ->
type name, e.g. {"a": "number", "b": "number"}) at runtime and
drives the same state sequence for any schema:

    START -> KEY -> COLON -> VALUE -> COMMA_OR_END -> ... -> END

This machine only knows LOGICAL constraints (a literal character,
or "a value of type X is expected here"). It does NOT know about
vocabulary tokens or token IDs -- translating a logical constraint
into allowed token IDs is the job of constrained_decoder.py.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, FrozenSet, Tuple


class Kind(Enum):
    START = auto()
    KEY = auto()
    COLON = auto()
    VALUE = auto()
    COMMA_OR_END = auto()
    END = auto()


@dataclass(frozen=True)
class Constraint:
    """A single logical thing that is allowed right now.

    Exactly one of the two fields is set:
      - literal: an exact required string, e.g. "{" or '"a"'
      - value_type: the type a value must satisfy, e.g. "number"
    """

    literal: str = ""
    value_type: str = ""


@dataclass(frozen=True)
class DecoderState:
    """Where we are in building the JSON object, and what schema
    we're building it against.
    """

    kind: Kind
    schema: Dict[str, str]
    remaining_keys: Tuple[str, ...]
    current_key: str = ""


def initial_state(schema: Dict[str, str]) -> DecoderState:
    """Start state for a given schema, e.g. {"a": "number"}."""
    return DecoderState(
        kind=Kind.START,
        schema=schema,
        remaining_keys=tuple(schema.keys()),
    )


def get_allowed(state: DecoderState) -> FrozenSet[Constraint]:
    """Return the logical constraint(s) legal in this state."""
    if state.kind == Kind.START:
        return frozenset({Constraint(literal="{")})

    if state.kind == Kind.KEY:
        next_key = state.remaining_keys[0]
        return frozenset({Constraint(literal=f'"{next_key}"')})

    if state.kind == Kind.COLON:
        return frozenset({Constraint(literal=":")})

    if state.kind == Kind.VALUE:
        expected_type = state.schema[state.current_key]
        return frozenset({Constraint(value_type=expected_type)})

    if state.kind == Kind.COMMA_OR_END:
        if state.remaining_keys:
            return frozenset({Constraint(literal=",")})
        return frozenset({Constraint(literal="}")})

    return frozenset()  # END: nothing more is allowed


def advance(state: DecoderState, chosen: Constraint) -> DecoderState:
    """Consume `chosen` and return the resulting next state.

    Raises ValueError if `chosen` isn't currently allowed.
    """
    allowed = get_allowed(state)
    if chosen not in allowed:
        raise ValueError(f"{chosen} is not allowed in state {state.kind}")

    if state.kind == Kind.START:
        return DecoderState(
            kind=Kind.KEY,
            schema=state.schema,
            remaining_keys=state.remaining_keys,
        )

    if state.kind == Kind.KEY:
        key = state.remaining_keys[0]
        return DecoderState(
            kind=Kind.COLON,
            schema=state.schema,
            remaining_keys=state.remaining_keys[1:],
            current_key=key,
        )

    if state.kind == Kind.COLON:
        return DecoderState(
            kind=Kind.VALUE,
            schema=state.schema,
            remaining_keys=state.remaining_keys,
            current_key=state.current_key,
        )

    if state.kind == Kind.VALUE:
        return DecoderState(
            kind=Kind.COMMA_OR_END,
            schema=state.schema,
            remaining_keys=state.remaining_keys,
        )

    if state.kind == Kind.COMMA_OR_END:
        if chosen.literal == ",":
            return DecoderState(
                kind=Kind.KEY,
                schema=state.schema,
                remaining_keys=state.remaining_keys,
            )
        return DecoderState(
            kind=Kind.END,
            schema=state.schema,
            remaining_keys=state.remaining_keys,
        )

    raise ValueError(f"Cannot advance from terminal state {state}")
