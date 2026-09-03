"""Constrained decoding loop."""

from __future__ import annotations
from collections.abc import Callable, Sequence
import numpy as np
import numpy.typing as npt
from .constraints import FunctionCallConstraint
from .models import FunctionDefinition
from .vocabulary import Vocabulary

LogitsFn = Callable[[list[int]], Sequence[float]]
EncodeFn = Callable[[str], list[int]]
StepFn = Callable[[int, str, int, int, str, str], None]


class DecodingError(Exception):
    """Raised when constrained decoding cannot produce a valid call."""


class ConstrainedDecoder:
    """Generate a schema-valid function call via constrained decoding."""

    def __init__(
        self,
        functions: list[FunctionDefinition],
        vocabulary: Vocabulary,
        logits_fn: LogitsFn,
        encode_fn: EncodeFn,
        max_steps: int = 256,
        repetition_penalty: float = 2.0,
    ) -> None:
        """Set up the decoder for a fixed set of functions.

        Args:
            functions: The available function definitions.
            vocabulary: Token-id to text mapping.
            logits_fn: Returns next-token logits for a token-id context.
            encode_fn: Encodes prompt text into initial token ids.
            max_steps: Safety cap on generated tokens.
            repetition_penalty: Amount subtracted from a token's logit
                for each prior time it was chosen this decode. Without
                this, an unusual prompt can leave every legal option
                unappealing to the model except the one it just
                produced, so it keeps re-picking that same digit or
                character until ``max_steps`` is hit. The penalty
                grows with each repeat, so any such loop is eventually
                broken in favour of the next-best legal token (e.g.
                the terminator), instead of stalling for the full
                step budget.
        """
        self._functions = functions
        self._vocab = vocabulary
        self._logits_fn = logits_fn
        self._encode = encode_fn
        self._max_steps = max_steps
        self._repetition_penalty = repetition_penalty
        self._id_to_text = [
            vocabulary.token_text(i) for i in range(len(vocabulary))
        ]

    def _legal_token_ids(
        self, constraint: FunctionCallConstraint
    ) -> list[int]:
        """Return token ids whose full text the constraint accepts."""
        allowed_first = constraint.allowed_next()
        legal: list[int] = []
        for tid, text in enumerate(self._id_to_text):
            if not text or text[0] not in allowed_first:
                continue
            if constraint.accepts(text):
                legal.append(tid)
        return legal

    def _select(
        self,
        logits: Sequence[float],
        legal: list[int],
        counts: dict[int, int],
    ) -> int:
        """Pick the highest-logit legal token (illegal -> -inf).

        Args:
            logits: Raw next-token logits.
            legal: Token ids currently allowed by the constraint.
            counts: Number of prior times each token id was chosen
                this decode, used to penalise repeats.
        """
        arr: npt.NDArray = np.asarray(logits, dtype=np.float64)
        masked: npt.NDArray = np.full(arr.shape, -np.inf, dtype=np.float64)
        idx: npt.NDArray = np.asarray(legal, dtype=np.int64)
        masked[idx] = arr[idx]
        for tid, count in counts.items():
            if count and masked[tid] != -np.inf:
                masked[tid] -= self._repetition_penalty * count
        return int(np.argmax(masked))

    def decode(self, prompt: str, on_step: StepFn | None = None) -> str:
        """Generate the JSON function call for a prompt.

        Args:
            prompt: The full prompt text to seed the model.
            on_step: Optional per-token callback with
                (step, phase, num_legal, vocab_size, chosen, output).

        Returns:
            A complete, schema-valid JSON string.

        Raises:
            DecodingError: If no legal token exists or the step cap is
                exceeded before completion.
        """
        constraint = FunctionCallConstraint(self._functions, prompt=prompt)
        input_ids = list(self._encode(prompt))
        generated = ""
        steps = 0
        counts: dict[int, int] = {}
        last_phase = constraint.phase
        while not constraint.is_complete():
            if steps >= self._max_steps:
                raise DecodingError("exceeded max decoding steps")
            steps += 1
            if constraint.phase != last_phase:
                # Repetition counts must not leak across fields: a
                # phase change means we've moved on to a new literal,
                # key, or value, so prior counts (e.g. from copying
                # source_string) must not bias the next field's
                # (e.g. regex) token choice.
                counts = {}
                last_phase = constraint.phase
            logits = self._logits_fn(input_ids)
            legal = self._legal_token_ids(constraint)
            if not legal:
                raise DecodingError("no legal token at this step")
            best = self._select(logits, legal, counts)
            counts[best] = counts.get(best, 0) + 1
            text = self._id_to_text[best]
            for ch in text:
                constraint.advance(ch)
            if constraint.phase != last_phase:
                counts = {}
                last_phase = constraint.phase
            generated += text
            input_ids.append(best)
            if on_step is not None:
                on_step(
                    steps,
                    constraint.phase.name,
                    len(legal),
                    len(self._id_to_text),
                    text,
                    generated,
                )
        return generated
