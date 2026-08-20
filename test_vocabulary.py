"""Tests for the Vocabulary class."""

from src.vocabulary import Vocabulary


VOCAB_PATH = "PUT_YOUR_VOCAB_PATH_HERE"


def test_load_vocabulary() -> None:
    """Test loading the vocabulary."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    assert vocabulary.size == 151643


def test_token_to_id() -> None:
    """Test token to ID conversion."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    assert vocabulary.get_token_id("!") == 0


def test_id_to_token() -> None:
    """Test ID to token conversion."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    assert vocabulary.get_token(0) == "!"


def test_bidirectional_mapping() -> None:
    """Test both vocabulary directions."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    token = "!"
    token_id = vocabulary.get_token_id(token)

    assert vocabulary.get_token(token_id) == token


def test_token_existence() -> None:
    """Test token existence checks."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    assert vocabulary.has_token("!")
    assert not vocabulary.has_token("__definitely_not_a_token__")


def test_id_existence() -> None:
    """Test token ID existence checks."""
    vocabulary = Vocabulary.from_file(VOCAB_PATH)

    assert vocabulary.has_token_id(0)
    assert not vocabulary.has_token_id(-1)