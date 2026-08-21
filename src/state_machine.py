"""Schema-aware JSON object state machine."""

from enum import Enum, auto

from pydantic import BaseModel, ConfigDict, Field


class Kind(Enum):
    """Represent the current position in JSON generation."""

    START = auto()
    KEY = auto()
    COLON = auto()
    VALUE = auto()
    COMMA_OR_END = auto()
    END = auto()


class Constraint(BaseModel):
    """Represent one logical JSON constraint.

    Exactly one of ``literal`` or ``value_type`` must be provided.
    """

    model_config = ConfigDict(frozen=True)

    literal: str = ""
    value_type: str = ""

    def model_post_init(self, __context: object) -> None:
        """Validate the constraint contents."""
        has_literal = bool(self.literal)
        has_value_type = bool(self.value_type)

        if has_literal == has_value_type:
            raise ValueError(
                "Constraint must contain exactly one of "
                "'literal' or 'value_type'."
            )


class DecoderState(BaseModel):
    """Represent the current JSON generation state."""

    model_config = ConfigDict(frozen=True)

    kind: Kind
    schema: dict[str, str] = Field(default_factory=dict)
    remaining_keys: tuple[str, ...] = ()
    current_key: str = ""


def _copy_state(
    state: DecoderState,
    *,
    kind: Kind | None = None,
    remaining_keys: tuple[str, ...] | None = None,
    current_key: str | None = None,
) -> DecoderState:
    """Return a typed copy of a decoder state."""
    return DecoderState(
        kind=state.kind if kind is None else kind,
        schema=state.schema,
        remaining_keys=(
            state.remaining_keys
            if remaining_keys is None
            else remaining_keys
        ),
        current_key=(
            state.current_key
            if current_key is None
            else current_key
        ),
    )


def initial_state(schema: dict[str, str]) -> DecoderState:
    """Create the initial state for a JSON object.

    Args:
        schema: Mapping of field names to JSON value types.

    Returns:
        Initial decoder state.

    Raises:
        ValueError: If the schema format is invalid.
    """
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary.")

    for key, value_type in schema.items():
        if not isinstance(key, str):
            raise ValueError("Schema keys must be strings.")

        if not isinstance(value_type, str):
            raise ValueError(f"Type for field {key!r} must be a string.")

    return DecoderState(
        kind=Kind.START,
        schema=dict(schema),
        remaining_keys=tuple(schema.keys()),
    )


def get_allowed(
    state: DecoderState,
) -> frozenset[Constraint]:
    """Return constraints allowed in the current state.

    Args:
        state: Current decoder state.

    Returns:
        Logical constraints allowed by the state machine.
    """
    if state.kind == Kind.START:
        return frozenset({Constraint(literal="{")})

    if state.kind == Kind.KEY:
        if not state.remaining_keys:
            return frozenset({Constraint(literal="}")})

        key = state.remaining_keys[0]

        return frozenset({Constraint(literal=f'"{key}"')})

    if state.kind == Kind.COLON:
        return frozenset({Constraint(literal=":")})

    if state.kind == Kind.VALUE:
        value_type = state.schema.get(state.current_key)

        if value_type is None:
            raise ValueError(f"No type found for key {state.current_key!r}.")

        return frozenset({Constraint(value_type=value_type)})

    if state.kind == Kind.COMMA_OR_END:
        if state.remaining_keys:
            return frozenset({Constraint(literal=",")})

        return frozenset({Constraint(literal="}")})

    return frozenset()


def advance(
    state: DecoderState,
    chosen: Constraint,
) -> DecoderState:
    """Advance the state after completing a constraint.

    Args:
        state: Current state.
        chosen: Completed constraint.

    Returns:
        New decoder state.

    Raises:
        ValueError: If the constraint is not allowed.
    """
    if chosen not in get_allowed(state):
        raise ValueError(f"{chosen} is not allowed in state {state.kind}.")

    if state.kind == Kind.START:
        return _copy_state(state, kind=Kind.KEY)

    if state.kind == Kind.KEY:
        if not state.remaining_keys:
            raise ValueError("No key available.")

        return _copy_state(
            state,
            kind=Kind.COLON,
            current_key=state.remaining_keys[0],
        )

    if state.kind == Kind.COLON:
        return _copy_state(state, kind=Kind.VALUE)

    if state.kind == Kind.VALUE:
        return _copy_state(
            state,
            kind=Kind.COMMA_OR_END,
            remaining_keys=state.remaining_keys[1:],
            current_key="",
        )

    if state.kind == Kind.COMMA_OR_END:
        if chosen.literal == ",":
            return _copy_state(state, kind=Kind.KEY)

        return _copy_state(state, kind=Kind.END)

    raise ValueError(f"Cannot advance from terminal state {state.kind}.")
