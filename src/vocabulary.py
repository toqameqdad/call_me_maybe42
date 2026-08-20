"""Vocabulary loading and token/ID lookup utilities."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class Vocabulary(BaseModel):
    """Represent the model vocabulary.

    The vocabulary provides lookup in both directions between token
    strings and integer token IDs.
    """

    model_config = ConfigDict(frozen=True)

    token_to_id: dict[str, int] = Field(default_factory=dict)
    id_to_token: dict[int, str] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a JSON file.

        Args:
            path: Path to the vocabulary JSON file.

        Returns:
            A populated Vocabulary instance.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is invalid JSON.
            ValueError: If the vocabulary format is invalid.
        """
        vocabulary_path = Path(path)

        if not vocabulary_path.is_file():
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocabulary_path}"
            )

        with vocabulary_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Vocabulary must be a JSON object.")

        token_to_id: dict[str, int] = {}
        seen_ids: set[int] = set()

        for token, token_id in data.items():
            if not isinstance(token, str):
                raise ValueError(
                    "Every vocabulary token must be a string."
                )

            if isinstance(token_id, bool) or not isinstance(
                token_id, int
            ):
                raise ValueError(
                    f"Token ID for {token!r} must be an integer."
                )

            if token_id < 0:
                raise ValueError(
                    f"Token ID cannot be negative: {token_id}"
                )

            if token_id in seen_ids:
                raise ValueError(
                    f"Duplicate token ID found: {token_id}"
                )

            seen_ids.add(token_id)
            token_to_id[token] = token_id

        id_to_token = {
            token_id: token
            for token, token_id in token_to_id.items()
        }

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )

    @property
    def size(self) -> int:
        """Return the number of tokens."""
        return len(self.token_to_id)

    def get_token_id(self, token: str) -> int:
        """Return the ID associated with a token.

        Args:
            token: Token string.

        Returns:
            The corresponding token ID.

        Raises:
            KeyError: If the token does not exist.
        """
        return self.token_to_id[token]

    def get_token(self, token_id: int) -> str:
        """Return the token associated with an ID.

        Args:
            token_id: Token ID.

        Returns:
            The corresponding token string.

        Raises:
            KeyError: If the ID does not exist.
        """
        return self.id_to_token[token_id]

    def has_token(self, token: str) -> bool:
        """Return whether a token exists."""
        return token in self.token_to_id

    def has_token_id(self, token_id: int) -> bool:
        """Return whether a token ID exists."""
        return token_id in self.id_to_token
